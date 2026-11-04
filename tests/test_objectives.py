import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import mosim_env
from crosim_agent.actions import ACTIONS, ACTION_INDEX, KEYS, action_bits, tick_bits
from crosim_agent.features import Scene, FeatureExtractor, FeatureHistory, pack_features
from crosim_agent.objectives import ObjectiveController
from crosim_agent.rewards import CycleReward, RewardState, LABELS
from mosim_debug import REWARD_LABELS


class ObjectiveTests(unittest.TestCase):
    def scene(self):
        return Scene(fuel_visible=True, fuel_x=-0.5, fuel_y=0.8, fuel_mass=0.02,
                     dense_x=0.6, dense_y=0.5, dense_mass=0.7,
                     hub_visible=True, hub_x=0.0, hub_y=0.5, hub_scale=0.2)

    def test_collect_prefers_a_large_cluster_over_a_nearby_single_ball(self):
        decision = ObjectiveController().update(self.scene(), fill=0.1)
        self.assertEqual(decision.goal, "collect")
        self.assertEqual(decision.target_kind, "dense fuel cluster")
        self.assertEqual(decision.target_x, 0.6)
        self.assertIn("search_right", decision.snapshot()["allowed_actions"])
        self.assertIn("shoot", decision.snapshot()["allowed_actions"])
        self.assertGreater(sum(decision.allowed),4)

    def test_full_hopper_overrides_cluster_and_latches_until_nearly_empty(self):
        controller = ObjectiveController()
        for fill in (0.95, 0.6, 0.2, 0.09):
            decision = controller.update(self.scene(), fill=fill)
            self.assertIn(decision.goal, ("return","shoot"))
            self.assertEqual(decision.target_kind, "blue hub marker")
            if fill >= .8:
                self.assertIn("collect_forward",decision.snapshot()["allowed_actions"])
        for _ in range(35):
            decision = controller.update(Scene(),fill=.07)
        self.assertEqual(decision.goal,"collect")

    def test_aligned_loaded_robot_shoots_or_waits_at_the_hub(self):
        scene = self.scene()
        scene.hub_scale = 0.7
        for active, goal, action in ((True, "shoot", "shoot"), (False, "wait", "coast")):
            decision = ObjectiveController().update(scene, fill=0.9, hub_active=active, no_progress=9)
            self.assertEqual(decision.goal, goal)
            self.assertEqual(decision.snapshot()["recommended_action"],action)
            if active:
                self.assertGreater(sum(decision.allowed),4)
        scene.hub_scale = 0.1
        self.assertEqual(ObjectiveController().update(scene, fill=0.9, hub_active=False).goal, "return")

    def test_endgame_can_return_with_a_partial_hopper(self):
        self.assertIn(ObjectiveController().update(self.scene(), fill=0.2, time_remaining=12).goal, ("return","shoot"))

    def test_loaded_robot_must_try_space_instead_of_approaching_forever(self):
        scene = self.scene()
        scene.hub_scale = 0.4
        decision = ObjectiveController().update(scene, fill=0.9)
        self.assertEqual(decision.goal, "shoot")
        self.assertIn("shoot",decision.snapshot()["allowed_actions"])
        self.assertGreater(sum(decision.allowed),4)

    def test_alternating_turns_and_coasts_cannot_reset_the_turn_budget(self):
        scene = Scene()
        controller = ObjectiveController()
        for tick in range(24):
            decision = controller.update(scene, fill=0.0, keys=tick_bits(12 if tick % 4 < 2 else 13, tick % 2))
        self.assertEqual(decision.turn_budget, 0)
        self.assertIn("collect_forward",decision.snapshot()["allowed_actions"])
        self.assertIn("aim_left",decision.snapshot()["allowed_actions"])
        for _ in range(7):
            decision = controller.update(scene, fill=0.0, keys=action_bits(1))
        self.assertGreater(decision.turn_budget, 0)

    def test_stalled_robot_gets_a_short_non_rotating_escape(self):
        scene = self.scene()
        scene.hub_visible = False
        decision = ObjectiveController().update(scene, fill=0.2, no_progress=9)
        self.assertEqual(decision.goal, "recover")
        self.assertGreater(sum(decision.allowed),4)
        self.assertIn(decision.snapshot()["recommended_action"],("retreat","return_left","return_right"))

    def test_disconnected_balls_are_grouped_without_labeling_each_ball(self):
        frame = np.zeros((300, 600, 3), np.uint8)
        for x in range(410, 471, 12):
            cv2.circle(frame, (x, 130), 4, (0, 255, 255), -1)
        cv2.circle(frame, (100, 230), 4, (0, 255, 255), -1)
        scene = FeatureExtractor(mosim_env.CFG, landmarks=False,zones=False).analyze(frame)
        self.assertLess(scene.fuel_x, 0)
        self.assertGreater(scene.dense_x, 0)
        self.assertGreater(scene.dense_mass, scene.fuel_mass*3)
        self.assertEqual(ObjectiveController().update(scene, fill=0.1).target_kind, "dense fuel cluster")


class ProgressRewardTests(unittest.TestCase):
    def test_spinning_with_large_view_motion_still_gets_no_progress_and_turn_costs(self):
        engine = CycleReward()
        engine.reset(RewardState(fill=0.2))
        total = 0.0
        for tick in range(70):
            parts, _, info = engine.update(RewardState(fill=0.2, motion=999, rotating=True,
                                                       movement_requested=True, elapsed=tick/10))
            total += sum(parts.values())
        self.assertGreater(engine.idle_seconds, 3)
        self.assertLess(parts["idle"], 0)
        self.assertLess(parts["spin"], 0)
        self.assertLess(total, 0)
        self.assertFalse(info["task_progress"])
        self.assertNotIn("smooth", LABELS)
        self.assertNotIn("switch", LABELS)
        self.assertNotIn("smooth", REWARD_LABELS)

    def test_shooting_without_points_does_not_clear_no_progress(self):
        engine = CycleReward()
        state = RewardState(fill=0.9, shooting=True, goal="shoot", hub_visible=True, hub_scale=0.8)
        engine.reset(state)
        for _ in range(40):
            parts, _, _ = engine.update(state)
        self.assertLess(parts["idle"], 0)

    def test_only_loaded_wait_at_a_close_inactive_hub_is_exempt(self):
        for scale, expected_idle in ((0.8, False), (0.1, True)):
            state = RewardState(fill=0.9, goal="wait", hub_visible=True, hub_scale=scale, hub_active=False)
            engine = CycleReward()
            engine.reset(state)
            for _ in range(40):
                parts, _, _ = engine.update(state)
            self.assertEqual(parts["idle"] < 0, expected_idle)

    def test_fill_noise_cannot_keep_refreshing_grace(self):
        engine = CycleReward()
        engine.reset(RewardState(fill=0.2))
        for tick in range(60):
            engine.update(RewardState(fill=0.20 if tick % 2 else 0.21, motion=100))
        self.assertGreater(engine.idle_seconds, 3)

    def test_supported_unloading_allows_the_next_collection_cycle_without_invented_points(self):
        engine = CycleReward()
        engine.reset(RewardState(fill=0.8, goal="shoot"))
        for fill in (0.6, 0.4, 0.2, 0.05):
            parts, _, _ = engine.update(RewardState(fill=fill, goal="shoot", shooting=True))
            self.assertEqual(parts["score"], 0)
        _, _, info = engine.update(RewardState(fill=0.1, goal="collect"))
        self.assertTrue(info["task_progress"])
        self.assertEqual(engine.credited_points, 0)
        self.assertEqual(engine.idle_seconds, 0)


class ObjectivePolicyTests(unittest.TestCase):
    def test_mask_matches_sampling_training_and_save_load_paths(self):
        import torch
        from crosim_agent.training import make_model
        from crosim_agent.proxy import PracticeEnv
        model = make_model(PracticeEnv())
        observations, masks = [], []
        for fill, scale, active in ((0.1, 0.1, True), (0.9, 0.1, True), (0.9, 0.8, True), (0.9, 0.8, False)):
            scene = Scene(fuel_visible=True, fuel_x=0.3, fuel_y=0.5,
                          hub_visible=True, hub_x=0, hub_scale=scale)
            decision = ObjectiveController().update(scene, fill=fill, hub_active=active)
            observations.append(FeatureHistory().reset(pack_features(scene, fill=fill, objective=decision)))
            masks.append(decision.allowed)
        batch = torch.as_tensor(np.asarray(observations))
        probs = model.policy.get_distribution(batch).distribution.probs
        self.assertTrue(torch.all(probs[~torch.as_tensor(masks)] == 0))
        actions, _, log_prob = model.policy(batch)
        _, evaluated, entropy = model.policy.evaluate_actions(batch, actions)
        torch.testing.assert_close(log_prob, evaluated)
        self.assertTrue(torch.isfinite(entropy).all())
        (-evaluated.mean()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.policy.parameters() if p.grad is not None))
        model.get_env().close()

    def test_camera_uses_h_in_the_shared_and_agent_bindings(self):
        self.assertEqual(KEYS[9], "h")
        self.assertEqual(mosim_env.ACTION_KEYS[mosim_env.CAMERA_IDX], "h")
        self.assertEqual(mosim_env.VK_CODES["h"], 0x48)
        self.assertNotIn("i", mosim_env.VK_CODES)

    def test_default_old_checkpoint_gets_a_fresh_starter_but_explicit_model_is_checked(self):
        from crosim_agent import cli
        from crosim_agent.checkpoints import save_checkpoint, write_json_atomic, fingerprint, CheckpointSchemaMismatch

        class Model:
            num_timesteps = 1
            def save(self, name):
                Path(name).write_bytes(b"old weights")

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            directory = Path(folder)/"live"
            old = save_checkpoint(Model(), directory, domain="old")
            metadata = json.loads(old.with_suffix(".json").read_text())
            metadata["spec"]["policy_version"] = "old-v1"
            metadata["fingerprint"] = fingerprint(metadata["spec"])
            write_json_atomic(old.with_suffix(".json"), metadata)
            args = cli.parser().parse_args(["bootstrap", "--models", folder])
            with patch("crosim_agent.training.bootstrap", return_value=(Model(), old, {})) as bootstrap:
                _, info = cli.load_or_bootstrap(args, directory, cli.reward_settings(args))
                bootstrap.assert_called_once()
                self.assertEqual(info["domain"], "bootstrap_proxy")
                self.assertTrue(old.exists())
            args.model = str(old)
            with self.assertRaises(CheckpointSchemaMismatch):
                cli.load_or_bootstrap(args, directory, cli.reward_settings(args))


if __name__ == "__main__":
    unittest.main()
