import json
import math
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

import mosim_env as env_module
from mosim_controls import InputGuard, game_title_matches
from mosim_debug import REWARD_LABELS, RewardLedger, RewardLog
from mosim_overlay import paint
from mosim_vision import Detection, DetectionWorker, ObjectDetector, color_candidates
from tests.scenario import ACTIONS, fake_env, run_scenario


def parts(**kwargs):
    values = dict.fromkeys(REWARD_LABELS, 0.0)
    values.update(kwargs)
    return values


def eventually(predicate, timeout=1.5):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Background task did not reach expected state")


class AccountingTests(unittest.TestCase):
    def test_every_term_and_both_signs_reconcile(self):
        ledger = RewardLedger()
        for tick in range(100):
            values = parts(time=-0.001, shaping=0.2 if tick % 2 else -0.3,
                           score=20 if tick == 50 else 0)
            self.assertAlmostEqual(ledger.record(values, tick/10), sum(values.values()))
        self.assertAlmostEqual(ledger.total, sum(ledger.totals.values()))
        self.assertAlmostEqual(ledger.total, ledger.positive + ledger.negative)
        self.assertEqual(ledger.last_score["amount"], 20)

    def test_repeats_group_but_sign_changes_and_gaps_do_not(self):
        ledger = RewardLedger()
        ledger.record(parts(shaping=0.01), 0.1)
        ledger.record(parts(shaping=0.02), 0.2)
        ledger.record(parts(shaping=-0.01), 0.3)
        ledger.record(parts(), 0.4)
        ledger.record(parts(shaping=-0.01), 0.5)
        self.assertEqual(len(ledger.events), 3)
        self.assertEqual(ledger.events[0]["count"], 2)
        self.assertAlmostEqual(ledger.events[0]["amount"], 0.03)
        frozen = ledger.snapshot()
        ledger.record(parts(shaping=-0.02), 0.6)
        self.assertEqual(frozen["events"][-1]["count"], 1)

    def test_reset_clears_ledger_and_history(self):
        ledger = RewardLedger()
        ledger.record(parts(score=10), 1)
        ledger.reset()
        self.assertEqual(ledger.total, 0)
        self.assertFalse(ledger.events)
        self.assertIsNone(ledger.last_score)

    def test_missing_and_nonfinite_terms_rejected(self):
        ledger = RewardLedger()
        with self.assertRaises(ValueError):
            ledger.record({"score": 1}, 0)
        with self.assertRaises(ValueError):
            ledger.record(parts(time=float("nan")), 0)

    def test_jsonl_keeps_each_tick_without_ui_grouping(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            log = RewardLog(directory)
            for tick in range(100):
                log.write({"type": "step", "step": tick, "components": parts(time=-0.001)})
            log.close()
            rows = [json.loads(line) for line in log.path.read_text().splitlines()]
            self.assertEqual(len(rows), 101)
            self.assertEqual(rows[0]["reward_units"], "raw_before_normalization")
            self.assertEqual([r["step"] for r in rows[1:]], list(range(100)))


class EnvironmentTests(unittest.TestCase):
    def test_cached_capture_returns_without_waiting_and_keeps_timestamp(self):
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        camera = unittest.mock.Mock()
        camera.grab.side_effect = [frame, None]
        grabber = env_module.Grabber.__new__(env_module.Grabber)
        grabber.camera = camera
        grabber._last = None
        grabber._last_capture_time = 0.0
        grabber._capture_lock = threading.Lock()
        with patch.object(env_module.time, "sleep", side_effect=AssertionError("Cached capture waited")):
            first, captured = grabber.grab_timed()
            cached, cached_at = grabber.grab_timed()
        self.assertIs(first, frame)
        self.assertIs(cached, frame)
        self.assertEqual(captured, cached_at)
        self.assertEqual(camera.grab.call_count, 2)

    def test_concurrent_vision_and_reward_capture_are_serialized(self):
        class Camera:
            active = 0
            maximum = 0

            def grab(self):
                self.active += 1
                self.maximum = max(self.maximum, self.active)
                time.sleep(0.02)
                self.active -= 1
                return np.zeros((20, 30, 3), dtype=np.uint8)

        grabber = env_module.Grabber.__new__(env_module.Grabber)
        grabber.camera = Camera()
        grabber._last = None
        grabber._last_capture_time = 0.0
        grabber._capture_lock = threading.Lock()
        threads = [threading.Thread(target=grabber.grab_timed) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=1)
            self.assertFalse(thread.is_alive())
        self.assertEqual(grabber.camera.maximum, 1)

    def test_v4_observations_and_other_rewards_preserved_without_movement_bonus(self):
        baseline = json.loads((Path(__file__).parent / "v4_reward_baseline.json").read_text())
        with patch.object(env_module, "CFG", env_module.Config({**env_module.CONFIG_DEFAULTS, **baseline["config"]})):
            actual = run_scenario(env_module)
        expected_steps = baseline["steps"]
        self.assertEqual(len(actual), len(expected_steps))
        for tick, (found, expected) in enumerate(zip(actual, expected_steps)):
            with self.subTest(tick=tick):
                self.assertEqual(found["hub_active"], expected["hub_active"])
                for name in ("motion", "fill"):
                    self.assertAlmostEqual(found[name], expected[name], places=10)
                previous = expected_steps[tick-1]["breakdown"]["smooth"] if tick else 0.0
                removed_bonus = expected["breakdown"]["smooth"]-previous
                self.assertAlmostEqual(found["reward"], expected["reward"]-removed_bonus, places=10)
                self.assertNotIn("smooth", found["breakdown"])
                np.testing.assert_allclose(found["aux"], expected["aux"], atol=1e-7)
                for name in REWARD_LABELS:
                    self.assertAlmostEqual(found["breakdown"][name], expected["breakdown"][name], places=10)

    def test_watch_never_sends_keys_and_log_matches_returned_rewards(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            with fake_env(env_module, directory, inject=False) as (env, pdi):
                env.reset()
                self.assertEqual(env._ledger.episode, 1)
                total = 0
                for index, keys in enumerate(ACTIONS):
                    env._elapsed = lambda i=index: (i+1)*0.1
                    action = [int(key in keys) for key in env_module.ACTION_KEYS]
                    _, reward, _, _, info = env.step(action)
                    total += reward
                    self.assertAlmostEqual(reward, math.fsum(info["reward_components"].values()))
                    self.assertAlmostEqual(total, info["episode_return"])
                path = env._reward_log.path
                self.assertFalse(pdi.events, "Watch mode injected a game key")
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            ticks = [r for r in rows if r["type"] == "step"]
            self.assertEqual(len(ticks), len(ACTIONS))
            self.assertTrue(all(r["mode"] == "watch" for r in ticks))
            self.assertAlmostEqual(total, sum(r["reward"] for r in ticks))
            self.assertTrue(all("phi_before" in r["diagnostics"] for r in ticks))

    def test_invalid_actions_fail_before_input(self):
        with fake_env(env_module, inject=True) as (env, pdi):
            env.reset()
            pdi.events.clear()
            for action in ([1]*9, [1]*11, [0.5]*10, [256]*10):
                with self.assertRaises(ValueError):
                    env.step(action)
            self.assertFalse(pdi.events)

    def test_dry_run_suppresses_resets_and_all_taps(self):
        with fake_env(env_module, inject=True, dry_run=True) as (env, pdi):
            env.reset()
            env.step(np.ones(10, dtype=np.int8))
            self.assertEqual(env._mode, "dry_run")
            self.assertFalse(pdi.events)

    def test_capture_that_lost_focus_is_discarded(self):
        with fake_env(env_module, inject=False) as (env, pdi):
            env.reset()
            checks = iter([False, True])
            env._guard.check_input = lambda: next(checks)
            frame = env._grab_focused()
            # The first frame (brightness 60) was captured with the wrong focus.
            self.assertEqual(int(frame[0, 0, 0]), 85)


class VisionTests(unittest.TestCase):
    def test_live_source_keeps_updating_without_reward_submissions(self):
        class Detector:
            status = "test"
            calls = 0

            def detect(self, frame):
                self.calls += 1
                return [Detection(str(int(frame[0, 0, 0])), (0, 0, 1, 1))]

        enabled = threading.Event()
        value = [1]

        def source():
            return np.full((1, 1, 3), value[0], dtype=np.uint8), time.perf_counter()

        detector = Detector()
        worker = DetectionWorker(detector, fps=40, frame_source=source, enabled=enabled.is_set)
        try:
            time.sleep(0.03)
            self.assertEqual(detector.calls, 0)
            enabled.set()
            eventually(lambda: detector.calls >= 1)
            value[0] = 7
            # No env.step / submit / OCR completion is needed for new positions.
            eventually(lambda: worker.latest()["boxes"] and worker.latest()["boxes"][0]["name"] == "7")
            before = detector.calls
            eventually(lambda: detector.calls >= before + 3)
            worker.submit(np.full((1, 1, 3), 99, np.uint8))
            self.assertIsNone(worker.pending, "Reward frames must not queue behind live capture")
            self.assertLess(time.perf_counter() - worker.latest()["captured_at"], 0.2)
        finally:
            worker.close()

    def test_no_weights_uses_explicitly_labeled_candidates(self):
        config = dict(env_module.CFG)
        frame = np.zeros((200, 400, 3), np.uint8)
        frame[30:50, 30:50] = [0, 255, 255]
        found = color_candidates(frame, config)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].source, "color_candidate")
        self.assertIsNone(found[0].confidence)
        self.assertNotIn(found[0].name, ("hub", "blue_robot", "red_robot"))

    def test_nonexistent_explicit_weights_fail_without_downloading(self):
        with self.assertRaises(FileNotFoundError):
            ObjectDetector(env_module.CFG, weights="this_model_does_not_exist.pt")

    def test_latest_frame_wins_and_reset_discards_inflight_result(self):
        started, release = threading.Event(), threading.Event()

        class Detector:
            status = "test"
            calls = []

            def detect(self, frame):
                value = int(frame[0, 0, 0])
                self.calls.append(value)
                if value == 1:
                    started.set()
                    release.wait(1)
                return [Detection(str(value), (0, 0, 1, 1))]

        detector = Detector()
        worker = DetectionWorker(detector, fps=1000)
        try:
            worker.submit(np.full((1, 1, 3), 1, np.uint8))
            self.assertTrue(started.wait(1))
            worker.submit(np.full((1, 1, 3), 2, np.uint8))
            worker.reset()
            worker.submit(np.full((1, 1, 3), 3, np.uint8))
            release.set()
            eventually(lambda: worker.latest()["boxes"] and worker.latest()["boxes"][0]["name"] == "3")
            self.assertEqual(detector.calls, [1, 3])
        finally:
            release.set()
            worker.close()


class FakeUser:
    def __init__(self):
        self.title = "MoSimulator"
        self.keys = {}
        self.initial_read = threading.Event()

    def GetAsyncKeyState(self, key):
        self.initial_read.set()
        return self.keys.get(key, 0)

    def GetForegroundWindow(self):
        return 1

    def GetWindowTextLengthW(self, hwnd):
        return len(self.title)

    def GetWindowTextW(self, hwnd, buffer, size):
        buffer.value = self.title


class SafetyTests(unittest.TestCase):
    def test_both_game_names_match_and_script_console_does_not(self):
        for expected in ("MoSim", "MoSimulator"):
            for actual in ("MoSim", "MoSimulator", "MoSim v26.4.1", "MoSimulator - REBUILT"):
                with self.subTest(expected=expected, actual=actual):
                    self.assertTrue(game_title_matches(expected, actual))
            for actual in ("Windows PowerShell", "mosim_env.py watch", "mosim_vision.py collect"):
                self.assertFalse(game_title_matches(expected, actual))

    def test_current_mosim_window_can_arm_with_right_bracket(self):
        user = FakeUser()
        user.title = "MoSim"
        guard = InputGuard(lambda: None, "MoSimulator", user=user)
        try:
            self.assertTrue(user.initial_read.wait(1))
            user.keys[0xDD] = 0x8000
            eventually(lambda: not guard.paused)
            self.assertTrue(guard.check_input())
            self.assertEqual(guard.status, "LIVE")
        finally:
            guard.close()

    def test_blocked_start_identifies_wrong_foreground_window(self):
        user = FakeUser()
        user.title = "Windows PowerShell"
        guard = InputGuard(lambda: None, "MoSim", user=user)
        try:
            self.assertFalse(guard.check_input())
            self.assertIn("Windows PowerShell", guard.status)
            self.assertTrue(guard.paused)
        finally:
            guard.close()

    def test_focus_loss_releases_and_requires_resume_and_stop_is_global(self):
        user = FakeUser()
        releases = []
        guard = InputGuard(lambda: releases.append(True), "MoSimulator", user=user)
        try:
            self.assertTrue(user.initial_read.wait(1))
            self.assertTrue(guard.paused)
            user.keys[0xDD] = 0x8000
            eventually(lambda: not guard.paused)
            user.keys[0xDD] = 0
            user.title = "Other app"
            eventually(lambda: guard.paused)
            self.assertTrue(releases)
            user.title = "MoSimulator"
            time.sleep(0.03)
            self.assertTrue(guard.paused)
            user.title = "Other app"
            user.keys[0xDB] = 0x8000
            self.assertTrue(guard.stopped.wait(1))
            with self.assertRaises(KeyboardInterrupt):
                guard.wait()
        finally:
            guard.close()


class RendererTests(unittest.TestCase):
    class Painter:
        def __init__(self):
            self.rects = []
            self.texts = []

        def rect(self, box, fill=None, outline=None):
            self.rects.append((box, fill, outline))

        def text(self, *args):
            self.texts.append(str(args[2]))

        def line(self, *args):
            pass

    def test_stale_or_paused_detections_are_hidden(self):
        from mosim_overlay import RED
        vision = {"boxes": [{"name": "red_robot", "box": [0.1, 0.1, 0.2, 0.2]}],
                  "captured_at": 10}
        for now, status, expected in ((10.2, "LIVE", 1), (11, "LIVE", 0), (10.2, "PAUSED", 0)):
            painter = self.Painter()
            paint(painter, 1920, 1080, {}, vision, status, now=now)
            self.assertEqual(sum(r[2] == RED for r in painter.rects), expected)

    def test_feature_agent_labels_score_credit_and_sensor_warning_are_visible(self):
        from crosim_agent.rewards import LABELS
        ledger = RewardLedger(labels=LABELS)
        painter = self.Painter()
        state = {"ledger": ledger.snapshot(), "points": 15, "credited_points": 4,
                 "policy_action": "collect_forward", "sensor_warning": "SCORE UNREADABLE"}
        paint(painter, 1920, 1080, state, {}, "LIVE", now=10)
        for label in LABELS.values():
            self.assertIn(label, painter.texts)
        self.assertIn("REWARDED / ALLIANCE", painter.texts)
        self.assertIn("4 / 15", painter.texts)
        self.assertTrue(any("SCORE UNREADABLE" in text for text in painter.texts))
        self.assertTrue(any("collect_forward" in text for text in painter.texts))

    def test_agent_target_crosshairs_expire_and_hide_when_paused(self):
        state = {"feature_captured_at": 10,
                 "agent_targets": [{"x": 0.3, "y": 0.4, "label": "Agent fuel target"}]}
        for now, status, expected in ((10.1, "LIVE", True), (10.4, "LIVE", False), (10.1, "PAUSED", False)):
            painter = self.Painter()
            paint(painter, 1920, 1080, state, {}, status, now=now)
            self.assertEqual("Agent fuel target" in painter.texts, expected)

    def test_fast_fuel_boxes_expire_after_quarter_second(self):
        from mosim_overlay import AMBER
        vision = {"boxes": [{"name": "fuel? color", "box": [0.1, 0.1, 0.2, 0.2]}],
                  "captured_at": 10, "max_age": 0.25}
        for now, expected in ((10.1, 1), (10.3, 0)):
            painter = self.Painter()
            paint(painter, 1920, 1080, {}, vision, "LIVE", now=now)
            self.assertEqual(sum(r[2] == AMBER for r in painter.rects), expected)

    def test_excluded_fuel_is_gray_and_the_ignored_count_is_visible(self):
        from mosim_overlay import MUTED, AMBER
        vision = {"captured_at": 10,"max_age":0.25,"fuel_zones":{"excluded_regions":8},
                  "boxes":[{"name":"fuel? color","box":[0.1,0.1,0.2,0.2],"collectible":False}]}
        painter = self.Painter()
        paint(painter,1920,1080,{},vision,"LIVE",now=10.1)
        self.assertEqual(sum(row[2] == MUTED for row in painter.rects),1)
        self.assertEqual(sum(row[2] == AMBER for row in painter.rects),0)
        self.assertTrue(any("ignored 8" in text for text in painter.texts))

    def test_panel_fits_720p_in_all_corners(self):
        for corner in ("top-left", "top-right", "bottom-left", "bottom-right"):
            painter = self.Painter()
            paint(painter, 1280, 720, {}, {}, corner=corner)
            for box, _, _ in painter.rects:
                self.assertGreaterEqual(min(box), 0)
                self.assertLessEqual(box[2], 1280)
                self.assertLessEqual(box[3], 720)

    def test_objective_panel_fits_720p_and_explains_its_goal(self):
        state = {"objective": {"goal": "return", "target_kind": "blue hub marker",
                               "reason": "Hopper 86%; return to blue hub, ignore fuel",
                               "allowed_actions": ["aim_right", "return_right"],
                               "turn_budget": 0.5, "no_progress_seconds": 1.0}}
        for corner in ("top-left", "top-right", "bottom-left", "bottom-right"):
            painter = self.Painter()
            paint(painter, 1280, 720, state, {}, corner=corner)
            self.assertTrue(any("Advice: RETURN" in text for text in painter.texts))
            self.assertTrue(any("Hopper 86%" in text for text in painter.texts))
            for box, _, _ in painter.rects:
                self.assertGreaterEqual(min(box), 0)
                self.assertLessEqual(box[2], 1280)
                self.assertLessEqual(box[3], 720)


if __name__ == "__main__":
    unittest.main()
