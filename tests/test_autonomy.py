import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from crosim_agent.actions import ACTION_INDEX
from crosim_agent.cli import parser,training_steps
from crosim_agent.features import Scene,feature_value
from crosim_agent.objectives import ObjectiveController
from crosim_agent.proxy import PracticeEnv
from crosim_agent.rewards import CycleReward,RewardState
from crosim_agent.session import TrainingLease,reset_training
from mosim_landmarks import HubTagDetector
from tests.test_agent import live_fixture


class AutonomousControlTests(unittest.TestCase):
    def test_unknown_hopper_cannot_hold_collection_forever(self):
        controller = ObjectiveController()
        for _ in range(170):
            decision = controller.update(Scene(),fill=0,fill_known=False)
        self.assertEqual(decision.goal,"return")
        self.assertIn("shoot",decision.snapshot()["allowed_actions"])
        self.assertGreater(sum(decision.allowed),4)

    def test_unknown_hopper_still_gets_sustained_space_attempts_at_own_hub(self):
        controller = ObjectiveController()
        scene = Scene(hub_visible=True,hub_x=.1,hub_scale=.2,tag_ids=(25,26))
        shots = 0
        for _ in range(90):
            decision = controller.update(scene,fill=0,fill_known=False)
            shots += decision.goal == "shoot" and sum(decision.allowed) > 1
        self.assertGreaterEqual(shots,18)

    def test_missing_hub_and_hopper_still_get_a_timed_exploratory_shot(self):
        controller = ObjectiveController()
        decisions = [controller.update(Scene(fuel_excluded=8),fill=0,fill_known=False) for _ in range(230)]
        shots = [d for d in decisions if d.goal == "shoot" and sum(d.allowed) > 1]
        self.assertGreaterEqual(len(shots),18)
        self.assertFalse(shots[0].target_visible)
        self.assertIn("unknown",shots[0].reason)

    def test_visible_target_guidance_survives_missing_inventory(self):
        engine = CycleReward()
        engine.reset(RewardState(fill_valid=False,goal="return",hub_visible=True,hub_scale=.1))
        parts,_,info = engine.update(RewardState(fill_valid=False,goal="return",hub_visible=True,hub_scale=.4))
        self.assertGreater(parts["approach"],0)
        self.assertEqual(info["phi_fuel_after"],0)
        self.assertTrue(info["task_progress"])

    def test_sensor_dropout_practice_exposes_known_and_unknown_inventory(self):
        env = PracticeEnv(sensor_dropout=True)
        obs,_ = env.reset(seed=6)
        seen = set()
        for _ in range(40):
            seen.add(bool(feature_value(obs,"fill_known")))
            obs,*_ = env.step(ACTION_INDEX["coast"])
        self.assertEqual(seen,{True,False})
        env.close()

    def test_stalled_live_match_never_restarts_before_timer_end(self):
        with live_fixture() as (env,pdi):
            env.reset()
            env._live_start = time.perf_counter()-50
            env.reward_engine.idle_seconds = 36
            env.reward_engine.phi_fuel = .3
            _,_,terminal,truncated,info = env.step(ACTION_INDEX["coast"])
            self.assertFalse(terminal)
            self.assertFalse(truncated)
            self.assertFalse(info.get("auto_restart",False))
            self.assertFalse(env._held_keys)
            env.reset()
            self.assertEqual(pdi.events.count(("down","r")),0)

    def test_live_default_is_continuous_and_offline_default_is_bounded(self):
        args = parser().parse_args(["train"])
        self.assertGreater(training_steps(args.steps,True),1_000_000)
        self.assertEqual(training_steps(args.steps,False),10000)
        self.assertEqual(training_steps(200,True),200)
        with self.assertRaises(ValueError):
            training_steps(-1,True)

    def test_real_cli_entry_accepts_default_and_explicit_continuous_steps(self):
        from crosim_agent import cli
        with patch.object(cli,"train",return_value=0) as train:
            self.assertEqual(cli.main(["train"]),0)
            self.assertEqual(cli.main(["train","--steps","0"]),0)
            self.assertEqual(train.call_count,2)

    def test_ppo_restarts_multiple_completed_live_matches_without_rearming(self):
        from crosim_agent.training import make_model
        from crosim_agent.proxy import PracticeEnv
        model = make_model(PracticeEnv())
        model.get_env().close()
        with live_fixture() as (env,pdi):
            from crosim_agent.timer import MatchClock
            env.match_clock = MatchClock(zero_hold=0,zero_reads=2)
            reads = [0]
            def reset_reader():
                reads[0] = 0
            def update(frame,now):
                reads[0] += 1
                env._async_score.snapshot.timer_seconds = 1 if reads[0] <= 2 else 0
                env._async_score.snapshot.timer_at = now
                return 0
            env._async_score.reset = reset_reader
            env._async_score.update = update
            model.set_env(env)
            model.learn(total_timesteps=128)
            self.assertGreaterEqual(pdi.events.count(("down","r")),2)


class TrainingResetTests(unittest.TestCase):
    def test_reset_removes_resume_paths_and_preserves_calibration_and_vision(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            (root/"crosim_agent_models/live").mkdir(parents=True)
            (root/"crosim_agent_models/live/latest.json").write_text('{"bad":"weights"}')
            (root/"mosim_demos").mkdir()
            (root/"mosim_demos/old.npz").write_bytes(b"old demo")
            (root/"mosim_ppo_model_final_v4.zip").write_bytes(b"old policy")
            (root/"mosim_config.json").write_text('{"calibration":true}')
            (root/"vision_weights").mkdir()
            (root/"vision_weights/best.pt").write_bytes(b"vision")
            archive,moved = reset_training(root)
            self.assertFalse((root/"crosim_agent_models").exists())
            self.assertFalse((root/"mosim_demos").exists())
            self.assertTrue((archive/"crosim_agent_models/live/latest.json").is_file())
            self.assertTrue((archive/"mosim_ppo_model_final_v4.zip").is_file())
            self.assertTrue((root/"mosim_config.json").is_file())
            self.assertTrue((root/"vision_weights/best.pt").is_file())
            self.assertEqual(set(json.loads((archive/"reset.json").read_text())["archived"]),set(moved))

    def test_second_trainer_and_reset_are_refused_while_first_is_running(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            with TrainingLease(folder):
                with self.assertRaisesRegex(RuntimeError,"Another CroSim"):
                    with TrainingLease(folder):
                        pass
                with self.assertRaises(RuntimeError):
                    reset_training(folder)


class LandmarkContinuityTests(unittest.TestCase):
    @staticmethod
    def frame():
        marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),25,100)
        gray = np.full((360,640),255,np.uint8)
        gray[100:200,250:350] = marker
        return cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)

    def test_short_decode_gap_tracks_current_pixels_and_expires(self):
        detector = HubTagDetector(interval=0)
        frame = self.frame()
        first = detector.detect(frame,10)
        self.assertEqual([a.tag_id for a in first],[25])
        moved = cv2.warpAffine(frame,np.float32([[1,0,5],[0,1,3]]),(640,360),borderValue=(255,255,255))
        with patch.object(detector,"_decode",return_value=[]):
            tracked = detector.detect(moved,10.1)
            self.assertEqual([a.tag_id for a in tracked],[25])
            self.assertTrue(tracked[0].tracked)
            np.testing.assert_allclose(tracked[0].center,np.array(first[0].center)+[5/640,3/360],atol=.003)
            self.assertEqual(detector.detect(moved,10.8),[])

    def test_occluded_marker_is_not_tracked_into_blank_space(self):
        detector = HubTagDetector(interval=0)
        detector.detect(self.frame(),10)
        with patch.object(detector,"_decode",return_value=[]):
            self.assertEqual(detector.detect(np.full((360,640,3),255,np.uint8),10.1),[])
