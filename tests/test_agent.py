from dataclasses import replace
from contextlib import contextmanager, ExitStack
import json
import math
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from crosim_agent.actions import ACTIONS, action_bits, tick_bits
from crosim_agent.features import (Scene, FeatureExtractor, FeatureHistory, OBS_DIM,
                                   FRAME_DIM, FEATURE_INDEX, pack_features, feature_value)
from crosim_agent.rewards import CycleReward, RewardConfig, RewardState, LABELS
from crosim_agent.proxy import PracticeEnv
from crosim_agent.checkpoints import save_checkpoint, resolve_checkpoint, schema, fingerprint
from crosim_agent.sensors import AsyncScoreReader
from mosim_debug import RewardLedger
from mosim_landmarks import HubTagDetector, strongest_hub
import mosim_env


TEST_ROOT = Path(__file__).resolve().parent


def eventually(predicate, seconds=2):
    end = time.perf_counter()+seconds
    while time.perf_counter() < end:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Background condition did not become true")


class ActionTests(unittest.TestCase):
    def test_actions_preserve_real_key_mapping_without_conflicts(self):
        for index in range(len(ACTIONS)):
            bits = action_bits(index)
            self.assertEqual(bits.shape, (10,))
            self.assertEqual(mosim_env.count_conflicts(bits), 0)
            self.assertEqual(bool(bits[9]),ACTIONS[index].name == "camera_swap")
        for bad in (-1, len(ACTIONS), 0.5):
            with self.assertRaises(ValueError):
                action_bits(bad)

    def test_fine_aim_releases_after_one_tick(self):
        self.assertTrue(tick_bits(12, 0)[4])
        self.assertFalse(tick_bits(12, 1).any())
        self.assertTrue(tick_bits(1, 1)[0])


class FeatureTests(unittest.TestCase):
    def test_fixed_slots_history_and_reset(self):
        scene = Scene(fuel_visible=True, fuel_x=-0.4, hub_visible=False)
        first = pack_features(scene, fill=0.6, previous_action=11)
        history = FeatureHistory()
        obs = history.reset(first)
        self.assertEqual(obs.shape, (OBS_DIM,))
        self.assertAlmostEqual(feature_value(obs, "fill"), 0.6, places=6)
        self.assertEqual(feature_value(obs, "previous_shoot"), 1)
        scene.fuel_x = 0.7
        history.push(pack_features(scene, fill=0.2))
        self.assertAlmostEqual(feature_value(history.observation(), "fuel_x"), 0.7, places=6)
        history.reset(pack_features(Scene()))
        self.assertEqual(feature_value(history.observation(), "fuel_visible"), 0)

    def test_full_screen_hud_and_hopper_yellow_are_excluded(self):
        frame = np.zeros((300, 600, 3), np.uint8)
        left, top, right, bottom = mosim_env.CFG.hopper_region
        frame[int(top*300):int(bottom*300), int(left*600):int(right*600)] = [0, 255, 255]
        frame[260:] = [0, 255, 255]
        extractor = FeatureExtractor(mosim_env.CFG, landmarks=False,zones=False)
        scene = extractor.analyze(frame)
        self.assertFalse(scene.fuel_visible)
        frame[120:135, 100:115] = [0, 255, 255]
        scene = extractor.analyze(frame)
        self.assertTrue(scene.fuel_visible)
        self.assertLess(scene.fuel_x, 0)

    def test_observation_rejects_nan(self):
        with self.assertRaises(ValueError):
            pack_features(Scene(), fill=float("nan"))


class LandmarkTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(cv2, "aruco"), "OpenCV aruco is optional")
    def test_hub_tag_decodes_and_wall_tag_is_not_a_hub(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        frame = np.full((300, 550, 3), 255, np.uint8)
        for marker, x in ((25, 40), (1, 300)):
            image = cv2.aruco.generateImageMarker(dictionary, marker, 160)
            frame[50:210, x:x+160] = image[:, :, None]
        detector = HubTagDetector(interval=0)
        anchors = detector.detect(frame, captured_at=10.0)
        self.assertEqual([a.tag_id for a in anchors], [25])
        self.assertIsNotNone(strongest_hub(anchors, "blue", now=10.1))
        self.assertIsNone(strongest_hub(anchors, "red", now=10.1))
        self.assertIsNone(strongest_hub(anchors, "blue", now=11))


class RewardTests(unittest.TestCase):
    def test_all_confirmed_alliance_points_pay(self):
        engine = CycleReward()
        engine.reset(RewardState(fill=0.8))
        parts, _, diag = engine.update(RewardState(fill=0.8, elapsed=1), points=10)
        self.assertEqual(parts["score"], 36)
        self.assertEqual(diag["score_uncredited"], 0)

    def test_delayed_shoot_correlated_points_are_delivered_once(self):
        engine = CycleReward()
        engine.reset(RewardState(fill=0.8))
        engine.update(RewardState(fill=0.6, shooting=True, elapsed=1))
        parts, _, _ = engine.update(RewardState(fill=0, elapsed=1.8), points=4)
        self.assertEqual(parts["score"], 4*engine.config.score_per_point)
        parts, _, _ = engine.update(RewardState(fill=0, elapsed=4), points=4)
        self.assertEqual(parts["score"], 14.4)
        self.assertEqual(engine.credited_points, 8)

    def test_confirmed_score_after_space_outweighs_noisy_fill_or_hub_estimates(self):
        for state in (RewardState(fill=0, shooting=True, elapsed=1),
                      RewardState(fill=0.8, shooting=True, hub_active=False, elapsed=1)):
            engine = CycleReward()
            engine.reset(replace(state, elapsed=0))
            parts, _, _ = engine.update(state, points=5)
            self.assertEqual(parts["score"], 5*engine.config.score_per_point)
            self.assertLess(parts["wasted_shot"], 0)

    def test_unknown_hub_does_not_create_false_inactive_penalty(self):
        engine = CycleReward()
        state = RewardState(fill=0.8, hub_active=False, hub_trusted=False, shooting=True)
        engine.reset(state)
        parts, _, _ = engine.update(state)
        self.assertEqual(parts["wasted_shot"], 0)

    def test_potential_cannot_be_farmed_by_fill_drain_or_sensor_gaps(self):
        cfg = RewardConfig()
        engine = CycleReward(cfg)
        engine.reset(RewardState())
        discounted = 0.0
        transitions = [RewardState(fill=1, fuel_visible=True, fuel_y=0.8),
                       RewardState(fill=0, fill_valid=False, settling=True),
                       RewardState(fill=0), RewardState(fill=1), RewardState(fill=0)]
        for index, state in enumerate(transitions):
            parts, _, _ = engine.update(state, terminal=index == len(transitions)-1)
            discounted += cfg.gamma**index*(parts["fuel"]+parts["approach"])
        self.assertAlmostEqual(discounted, 0.0, places=10)
        self.assertEqual(engine.phi_fuel, 0)

    def test_full_initial_hopper_does_not_get_fake_first_reward(self):
        engine = CycleReward()
        state = RewardState(fill=1)
        engine.reset(state)
        parts, _, _ = engine.update(state)
        self.assertLess(parts["fuel"], 0)

    def test_steady_movement_has_no_arbitrary_positive_bonus(self):
        engine = CycleReward()
        state = RewardState(motion=10, movement_requested=True)
        engine.reset(state)
        total = 0
        for _ in range(100):
            parts, _, _ = engine.update(state)
            total += sum(parts.values())
        self.assertLess(total, 0)

    def test_revised_reward_ledger_reconciles(self):
        engine = CycleReward()
        ledger = RewardLedger(labels=LABELS)
        parts, reasons, _ = engine.update(RewardState(fill=0.5, fuel_visible=True, fuel_y=0.7))
        returned = ledger.record(parts, 0.1, reasons)
        self.assertAlmostEqual(returned, sum(parts.values()))
        self.assertEqual(ledger.snapshot()["labels"], LABELS)


class AsyncScoreTests(unittest.TestCase):
    def test_factory_failure_is_reported_and_can_recover(self):
        calls = [0]

        class Reader:
            confirmed = 0
            _pending = None
            ocr_failures = 0

            def update(self, frame, now):
                self.confirmed = 2
                return 0

        def factory():
            calls[0] += 1
            if calls[0] == 1:
                raise RuntimeError("temporary initialization failure")
            return Reader()

        reader = AsyncScoreReader(factory, interval=0.005)
        try:
            reader.update(np.zeros((1, 1, 3), np.uint8), 1)
            eventually(lambda: reader.snapshot.error is not None)
            self.assertTrue(reader.thread.is_alive())
            self.assertIsNone(reader.confirmed)
            reader.update(np.zeros((1, 1, 3), np.uint8), 2)
            eventually(lambda: reader.confirmed == 2)
            self.assertIsNone(reader.snapshot.error)
        finally:
            reader.close()

    def test_crop_path_queues_only_the_score_image(self):
        shapes = []

        class Reader:
            confirmed = 0
            _pending = None
            ocr_failures = 0

            def update_crop(self, frame, now):
                shapes.append(frame.shape)
                return 0

        reader = AsyncScoreReader(Reader, frame_transform=lambda frame: frame[10:20, 20:40], crop_reader=True)
        try:
            reader.update(np.zeros((100, 200, 3), np.uint8), 1)
            eventually(lambda: bool(shapes))
            self.assertEqual(shapes[0], (10, 20, 3))
        finally:
            reader.close()

    def test_real_score_confirmation_and_timeout_contract(self):
        reader = mosim_env.ScoreReader(ocr_timeout=0.75)
        crop = np.full((20, 50, 3), 200, np.uint8)
        with patch("mosim_env.ocr_score_crop", side_effect=[0, 0, 3, 3]) as ocr:
            self.assertEqual(reader.update_crop(crop, 0), 0)
            self.assertEqual(reader.update_crop(crop, 0.1), 0)
            self.assertEqual(reader.confirmed, 0)
            self.assertEqual(reader.ocr_failures, 0)
            self.assertEqual(reader.update_crop(crop, 3), 0)
            self.assertEqual(reader.update_crop(crop, 3.1), 3)
            self.assertEqual(reader.update_crop(crop, 3.2), 0)
            self.assertEqual(ocr.call_args.kwargs["timeout"], 0.75)

    def test_zero_and_unchanged_score_remain_a_valid_baseline(self):
        reader = mosim_env.ScoreReader(ocr_timeout=0.75)
        crop = np.full((20, 50, 3), 200, np.uint8)
        with patch("mosim_env.ocr_score_crop", return_value=0):
            for now in (0, 0.1, 3, 6, 9, 12, 40, 80):
                self.assertEqual(reader.update_crop(crop, now), 0)
            self.assertEqual(reader.confirmed, 0)
            self.assertEqual(reader.ocr_failures, 0)

    def test_blank_line_read_uses_word_fallback_with_shared_timeout(self):
        crop = np.zeros((40, 90, 3), np.uint8)
        cv2.putText(crop, "0", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        with patch("mosim_env.ocr_read", side_effect=[None, 0]) as read:
            self.assertEqual(mosim_env.ocr_score_crop(crop, timeout=0.75), 0)
            self.assertEqual(read.call_count, 2)
            self.assertEqual(read.call_args.kwargs["psm"], 8)
            self.assertGreater(read.call_args.kwargs["timeout"], 0)
            self.assertLessEqual(read.call_args.kwargs["timeout"], 0.75)

    def test_ocr_parser_preserves_zero_and_rejects_phantom_leading_zero(self):
        from types import SimpleNamespace
        binary = np.full((20, 50), 255, np.uint8)
        for text, expected in (("0\n", 0), ("9", 9), ("12", 12), ("07", None), ("", None)):
            with patch("mosim_env.need_pytesseract", return_value=SimpleNamespace(image_to_string=lambda *a, **kw: text)):
                self.assertEqual(mosim_env.ocr_read(binary), expected)

    def test_blank_read_is_not_invented_as_zero_and_zero_needs_no_fallback(self):
        crop = np.full((20, 50, 3), 200, np.uint8)
        with patch("mosim_env.ocr_read", return_value=None):
            self.assertIsNone(mosim_env.ocr_score_crop(crop, timeout=0.75))
        with patch("mosim_env.ocr_read", return_value=0) as read:
            self.assertEqual(mosim_env.ocr_score_crop(crop, timeout=0.75), 0)
            self.assertEqual(read.call_count, 1)

    def test_completed_points_are_drained_once(self):
        class Reader:
            confirmed = 0
            _pending = None
            ocr_failures = 0

            def update(self, frame, now):
                value = int(frame[0, 0, 0])
                delta = max(0, value-self.confirmed)
                self.confirmed = value
                return delta

        reader = AsyncScoreReader(Reader, interval=0.005)
        try:
            self.assertEqual(reader.update(np.full((1, 1, 3), 3, np.uint8), 1), 0)
            eventually(lambda: reader.confirmed == 3)
            self.assertEqual(reader.update(np.full((1, 1, 3), 3, np.uint8), 2), 3)
            self.assertEqual(reader.update(np.full((1, 1, 3), 3, np.uint8), 3), 0)
        finally:
            reader.close()

    def test_reset_discards_inflight_score(self):
        started, release = threading.Event(), threading.Event()

        class Reader:
            confirmed = None
            _pending = None
            ocr_failures = 0

            def update(self, frame, now):
                started.set()
                release.wait(1)
                self.confirmed = 20
                return 20

        reader = AsyncScoreReader(Reader)
        try:
            reader.update(np.zeros((1, 1, 3), np.uint8), 1)
            self.assertTrue(started.wait(1))
            reader.reset()
            release.set()
            time.sleep(0.03)
            self.assertIsNone(reader.confirmed)
            self.assertEqual(reader.points, 0)
        finally:
            release.set()
            reader.close()


class ProxyTests(unittest.TestCase):
    def test_gym_contract_seed_and_episode_end(self):
        env = PracticeEnv(duration=0.4)
        from stable_baselines3.common.env_checker import check_env
        check_env(env, warn=True)
        first, _ = env.reset(seed=42)
        second, _ = env.reset(seed=42)
        np.testing.assert_array_equal(first, second)
        for _ in range(2):
            obs, reward, terminal, truncated, info = env.step(1)
            self.assertTrue(env.observation_space.contains(obs))
            self.assertAlmostEqual(reward, sum(info["reward_components"].values()))
        self.assertTrue(terminal)
        self.assertFalse(truncated)
        with self.assertRaises(RuntimeError):
            env.step(1)


@contextmanager
def live_fixture(dry_run=False, inject_inputs=True):
    from crosim_agent.live import LiveFeatureEnv
    from tests.scenario import FakePDI

    class Camera:
        bounds = (0, 0, 640, 360)

        def grab(self):
            frame = np.zeros((360, 640, 3), np.uint8)
            frame[150:170, 100:120] = [0, 255, 255]
            return frame

        def release(self):
            pass

    class Guard:
        status = "LIVE"
        paused = False

        def __init__(self, release, title, input_lock=None):
            self.release = release
            self.input_lock = input_lock or threading.RLock()
            self.stopped, self.closed = threading.Event(), threading.Event()

        def wait(self):
            return 0

        def check_input(self):
            return True

        def focused(self):
            return True

        def close(self):
            self.closed.set()
            self.release()

    class Score:
        confirmed = 0
        _pending = None
        ocr_failures = 0

        def __init__(self, *args, **kwargs):
            self.closed = False
            from types import SimpleNamespace
            self.snapshot = SimpleNamespace(error=None)

        def update(self, frame, now):
            return 0

        def reset(self):
            pass

        def close(self):
            self.closed = True

    pdi = FakePDI()
    with ExitStack() as patches:
        patches.enter_context(patch("mosim_env.Grabber", Camera))
        patches.enter_context(patch("mosim_controls.InputGuard", Guard))
        patches.enter_context(patch("mosim_env.need_pydirectinput", return_value=pdi))
        patches.enter_context(patch("crosim_agent.live.AsyncScoreReader", Score))
        patches.enter_context(patch("mosim_env.time.sleep", return_value=None))
        env = LiveFeatureEnv(debug_options={"overlay": False, "log_rewards": False}, dry_run=dry_run,
                             inject_inputs=inject_inputs)
        env._wait_for_tick = lambda: None
        try:
            yield env, pdi
        finally:
            env.close()


class LiveIntegrationTests(unittest.TestCase):
    def test_stale_capture_releases_inputs_and_still_reaches_watchdog_reset(self):
        with live_fixture() as (env, pdi):
            env.reset()
            frame = env._latest_frame.copy()
            env.grabber.grab_timed = lambda: (frame, time.perf_counter()-10)
            env._live_start = time.perf_counter()-166
            _,_,terminal,_,info = env.step(1)
            self.assertTrue(terminal)
            self.assertIn("watchdog",info["reason"])
            self.assertFalse(env._held_keys)
            env.reset()
            self.assertEqual(pdi.events.count(("down","r")),1)

    def test_optional_robot_features_use_only_fresh_worker_results(self):
        from types import SimpleNamespace
        with live_fixture() as (env, pdi):
            env.reset()
            snapshot = {"captured_at": time.perf_counter(), "max_age": 0.8,
                        "boxes": [{"name": "red_robot", "box": [0.2, 0.2, 0.4, 0.5]}]}
            with patch.object(env, "_vision", SimpleNamespace(latest=lambda: snapshot)):
                self.assertTrue(env._analyze_frame(env._latest_frame).opponent_visible)
                snapshot["captured_at"] -= 2
                self.assertFalse(env._analyze_frame(env._latest_frame).opponent_visible)

    def test_unreadable_score_warns_but_keeps_training_without_fake_points(self):
        with live_fixture() as (env, pdi):
            env.reset()
            env._async_score.confirmed = None
            env._async_score.ocr_failures = 20
            self.assertIn("training continues", env._debug_state()["sensor_warning"])
            _, reward, terminal, truncated, info = env.step(1)
            self.assertFalse(terminal or truncated)
            self.assertTrue(math.isfinite(reward))
            self.assertEqual(info["reward_components"]["score"], 0)
            self.assertIsNone(env._async_score.confirmed)
            env.close()
            self.assertFalse(env._held_keys)
            self.assertTrue(env._async_score.closed)

    def test_confirmed_zero_is_not_reported_as_missing(self):
        with live_fixture() as (env, pdi):
            env.reset()
            env._async_score.confirmed = 0
            env._async_score.ocr_failures = 20
            self.assertNotIn("OCR", env._debug_state()["sensor_warning"] or "")
            self.assertFalse(env.step(1)[2])

    def test_human_camera_change_has_one_cost_per_press(self):
        with live_fixture(inject_inputs=False) as (env, pdi):
            env.reset()
            keys = np.zeros(10, np.int8)
            keys[9] = 1
            first = env.watch_step(keys)[4]["reward_components"]
            held = env.watch_step(keys)[4]["reward_components"]
            self.assertEqual(first["sensor"], -env.reward_config.sensor_cost)
            self.assertEqual(held["sensor"], 0)
            self.assertFalse(pdi.events)

    def test_human_watch_uses_actual_keys_and_never_injects(self):
        with live_fixture(inject_inputs=False) as (env, pdi):
            env.reset()
            keys = np.array([1, 0, 0, 0, 0, 0, 1, 1, 0, 0], np.int8)
            _, reward, _, _, info = env.watch_step(keys)
            self.assertAlmostEqual(reward, sum(info["reward_components"].values()))
            self.assertEqual(set(info["reward_components"]), set(LABELS))
            self.assertFalse(pdi.events)

    def test_live_wiring_uses_feature_observation_and_new_rewards(self):
        with live_fixture() as (env, pdi):
            obs, _ = env.reset(seed=1)
            self.assertTrue(env.observation_space.contains(obs))
            obs, reward, terminal, truncated, info = env.step(1)
            self.assertTrue(env.observation_space.contains(obs))
            self.assertEqual(set(info["reward_components"]), set(LABELS))
            self.assertAlmostEqual(reward, sum(info["reward_components"].values()))
            self.assertEqual(info["domain"], "live_screen")
            self.assertFalse(terminal or truncated)
            self.assertTrue(any(event == ("down", "w") for event in pdi.events))

    def test_live_dry_run_never_sends_resets_or_actions(self):
        with live_fixture(dry_run=True) as (env, pdi):
            env.reset()
            env.step(1)
            env.step(14)
            self.assertFalse(pdi.events)

    def test_long_control_gap_truncates_before_sending_stale_action(self):
        with live_fixture() as (env, pdi):
            env.reset()
            pdi.events.clear()
            env._last_control_end = time.perf_counter()-10
            _, reward, terminal, truncated, info = env.step(1)
            self.assertFalse(terminal)
            self.assertTrue(truncated)
            self.assertTrue(info["interrupted"])
            self.assertEqual(reward, 0)
            self.assertFalse(pdi.events)

    def test_natural_match_end_terminates_and_settles_potential(self):
        with live_fixture() as (env, pdi):
            env.reset()
            now = time.perf_counter()
            for at in (now-1.1,now-.5,now):
                env.match_clock.observe(0,at,now=now)
            _, _, terminal, truncated, info = env.step(1)
            self.assertTrue(terminal)
            self.assertFalse(truncated)
            self.assertTrue(info["reward_diagnostics"]["potential_terminal"])
            self.assertEqual(env.reward_engine.phi_fuel, 0)
            self.assertEqual(env.reward_engine.phi_target, 0)
            self.assertFalse(env._held_keys)

    def test_live_clock_does_not_hide_optimizer_or_pause_time(self):
        with live_fixture() as (env, pdi):
            env._live_start = 100
            env._lost_seconds = 50
            with patch("crosim_agent.live.time.perf_counter", return_value=105.5):
                self.assertEqual(env._elapsed(), 5.5)

    def test_fine_aim_does_not_leave_rotation_held(self):
        with live_fixture() as (env, pdi):
            env.reset()
            pdi.events.clear()
            env.step(12)
            self.assertIn(("down", "j"), pdi.events)
            self.assertIn(("up", "j"), pdi.events)
            self.assertFalse(env._held_keys)


class CheckpointTests(unittest.TestCase):
    def test_unpromoted_candidate_does_not_replace_latest(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as directory:
            first = save_checkpoint(self.Model(), directory, domain="test")
            candidate = save_checkpoint(self.Model(), directory, domain="candidate", promote=False)
            self.assertTrue(candidate.is_file())
            self.assertEqual(resolve_checkpoint(directory)[0], first)

    class Model:
        num_timesteps = 123

        def save(self, filename):
            Path(filename).write_bytes(b"test-model")

    def test_atomic_save_roundtrip_and_schema_validation(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as directory:
            path = save_checkpoint(self.Model(), directory, domain="test")
            found, metadata = resolve_checkpoint(directory)
            self.assertEqual(found, path)
            self.assertEqual(metadata["fingerprint"], fingerprint(schema()))
            with self.assertRaisesRegex(ValueError, "different"):
                resolve_checkpoint(directory, reward_config=RewardConfig(score_per_point=3))

    def test_corrupt_weights_and_escaping_pointer_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as directory:
            path = save_checkpoint(self.Model(), directory, domain="test")
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "integrity"):
                resolve_checkpoint(directory)
            (Path(directory)/"latest.json").write_text(json.dumps({"model": "../outside.zip"}))
            with self.assertRaisesRegex(ValueError, "escapes"):
                resolve_checkpoint(directory)

    def test_failed_save_keeps_previous_latest(self):
        class Broken:
            num_timesteps = 124

            def save(self, filename):
                Path(filename).write_bytes(b"partial")
                raise OSError("simulated disk failure")

        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as directory:
            first = save_checkpoint(self.Model(), directory, domain="test")
            with self.assertRaises(OSError):
                save_checkpoint(Broken(), directory, domain="test")
            self.assertEqual(resolve_checkpoint(directory)[0], first)
            self.assertFalse(list(Path(directory).glob("pending*")))

    def test_real_ppo_update_save_load_and_prediction(self):
        from stable_baselines3 import PPO
        from crosim_agent.training import make_model
        env = PracticeEnv(duration=4)
        model = make_model(env)
        try:
            model.learn(total_timesteps=128)
            obs, _ = env.reset(seed=2)
            before, _ = model.predict(obs, deterministic=True)
            with tempfile.TemporaryDirectory(dir=TEST_ROOT) as directory:
                path = save_checkpoint(model, directory, domain="test")
                loaded = PPO.load(str(resolve_checkpoint(directory)[0]), device="cpu")
                after, _ = loaded.predict(obs, deterministic=True)
                self.assertEqual(before.item(), after.item())
        finally:
            env.close()


class CliFailureTests(unittest.TestCase):
    def test_live_training_pipeline_updates_saves_and_releases_with_fake_hardware(self):
        from crosim_agent import cli
        from crosim_agent.training import make_model
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as directory, live_fixture() as (env, pdi):
            model = make_model(PracticeEnv(duration=1), env.reward_config)
            args = cli.parser().parse_args(["train", "--models", directory, "--steps", "128"])
            with patch.object(cli, "require_ready"), \
                    patch.object(cli, "load_or_bootstrap", return_value=(model, {"domain": "bootstrap_proxy"})), \
                    patch("crosim_agent.live.LiveFeatureEnv", return_value=env):
                cli.train(args)
            path, metadata = resolve_checkpoint(Path(directory)/"live", reward_config=env.reward_config)
            self.assertEqual(metadata["domain"], "live_screen")
            self.assertEqual(metadata["metrics"]["session_decisions"], 128)
            self.assertGreater(metadata["metrics"]["updates_performed"], 0)
            self.assertFalse(env._held_keys)
            self.assertTrue(env._async_score.closed)

    def test_evaluation_reports_zero_score_failures_and_is_repeatable(self):
        from crosim_agent.teacher import evaluate

        class Coast:
            def predict(self, observation, deterministic=True):
                return 0, None

        env = PracticeEnv(duration=0.4)
        try:
            first = evaluate(Coast(), env, (71, 72))
            self.assertEqual(first, evaluate(Coast(), env, (71, 72)))
            self.assertEqual(first["scoring_episodes"], 0)
            self.assertEqual(first["decisions"], [2, 2])
            self.assertFalse(first["real_game_skill_validated"])
            with self.assertRaises(ValueError):
                evaluate(Coast(), env, ())
        finally:
            env.close()

    def test_failed_native_start_does_not_claim_live_training(self):
        from crosim_agent import cli

        class Model:
            num_timesteps = 0

            def get_env(self):
                return None

        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as directory:
            args = cli.parser().parse_args(["train", "--models", directory])
            with patch.object(cli, "require_ready"), \
                    patch.object(cli, "load_or_bootstrap", return_value=(Model(), {"domain": "bootstrap_proxy"})), \
                    patch("crosim_agent.live.LiveFeatureEnv", side_effect=RuntimeError("capture unavailable")):
                with self.assertRaisesRegex(RuntimeError, "capture unavailable"):
                    cli.train(args)
            status = json.loads((Path(directory)/"live"/"status.json").read_text())
            self.assertFalse(status["training_started"])
            self.assertTrue(status["checkpoint_unchanged"])
            self.assertFalse((Path(directory)/"live"/"latest.json").exists())


if __name__ == "__main__":
    unittest.main()
