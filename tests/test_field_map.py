from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from mosim_field import FieldPoseTracker,FIELD
from mosim_overlay import paint
from mosim_debug import RewardLedger
from crosim_agent.rewards import CycleReward,RewardState,LABELS
from crosim_agent.features import Scene
from crosim_agent.checkpoints import save_checkpoint,resolve_checkpoint,write_json_atomic,fingerprint,CheckpointSchemaMismatch
from tests.test_zones import camera,project
from tests import test_debug
from tests.test_agent import live_fixture


def robot_region(h,point):
    u,v = project(h,[point])[0]/[h.width,h.height]
    return [float(u-.05),float(v-.15),float(u+.05),float(v)]


class FieldPoseTests(unittest.TestCase):
    def test_robot_uses_floor_projection_rather_than_camera_position(self):
        h = camera()
        original = h.rotation.copy()
        state = FieldPoseTracker().update([h],robot_region(h,[8,4,0]),captured_at=10,now=10)
        np.testing.assert_allclose([state["pose"]["x"],state["pose"]["y"]],[8.25,4],atol=1e-10)
        self.assertGreater(abs(state["pose"]["x"]-h.center[0]),2)
        self.assertFalse(state["opponent_zone"])
        self.assertTrue(state["pose_current"])
        np.testing.assert_array_equal(h.rotation,original)
        json.dumps(state,allow_nan=False)

    def test_red_zone_needs_all_pose_bounds_inside_the_boundary(self):
        h = camera((10,4,2.3),(14,4,0))
        gate = FieldPoseTracker()
        edge = gate.update([h],robot_region(h,[FIELD["red_start"],4,0]),captured_at=10,now=10)
        self.assertIsNone(edge["opponent_zone"])
        red = FieldPoseTracker().update([h],robot_region(h,[14,4,0]),captured_at=10,now=10)
        self.assertTrue(red["opponent_zone"])
        self.assertEqual(red["zone"],"red")

    def test_disagreeing_camera_hypotheses_cannot_create_zone_cost(self):
        h = camera()
        other = replace(h,center=h.center+np.array([6,0,0]))
        state = FieldPoseTracker().update([h,other],robot_region(h,[8,4,0]),captured_at=10,now=10)
        self.assertIsNone(state["opponent_zone"])
        self.assertGreater(state["pose"]["uncertainty_m"],2)

    def test_missing_or_stale_pose_is_last_known_only_and_resets_cleanly(self):
        h = camera()
        tracker = FieldPoseTracker()
        tracker.update([h],robot_region(h,[8,4,0]),captured_at=10,now=10)
        missing = tracker.update([],[],captured_at=10.1,now=10.1)
        self.assertFalse(missing["pose_current"])
        self.assertIsNone(missing["opponent_zone"])
        self.assertEqual(missing["pose"]["captured_at"],10)
        stale = tracker.update([h],robot_region(h,[8,4,0]),captured_at=10,now=11)
        self.assertFalse(stale["pose_current"])
        tracker.reset()
        self.assertEqual(tracker.update([],[],now=12)["pose"]["x"],8.25)

    def test_impossible_pose_jump_is_rejected(self):
        h = camera()
        tracker = FieldPoseTracker()
        region = robot_region(h,[8,4,0])
        tracker.update([h],region,captured_at=10,now=10)
        shifted = replace(h,center=h.center+np.array([6,0,0]))
        state = tracker.update([shifted],region,captured_at=10.1,now=10.1)
        self.assertFalse(state["pose_current"])
        self.assertIn("jump rejected",state["status"])

    def test_optional_robot_detections_project_to_map_without_inventing_heading(self):
        h = camera()
        u,v = project(h,[[10,4,0]])[0]/[h.width,h.height]
        detection = {"name":"red_robot","box":[u-.02,v-.1,u+.02,v],"captured_at":10}
        state = FieldPoseTracker().update([h],robot_region(h,[8,4,0]),captured_at=10,now=10,detections=[detection])
        self.assertEqual(len(state["robots"]),1)
        self.assertAlmostEqual(state["robots"][0]["x"],10)
        self.assertNotIn("heading",state["robots"][0])


class ZoneRewardTests(unittest.TestCase):
    def setUp(self):
        self.engine = CycleReward()
        self.tick = 0

    def samples(self,evidence,count):
        amounts,entries = [],0
        for _ in range(count):
            self.tick += 1
            state = RewardState(opponent_zone=evidence,zone_sample_at=self.tick/10)
            parts,_,info = self.engine.update(state)
            amounts.append(parts["red_zone"])
            entries += info["red_zone_entry"]
        return sum(amounts),entries

    def test_crossing_pays_once_then_only_a_small_occupancy_cost(self):
        self.samples(False,5)
        cost,entries = self.samples(True,10)
        self.assertEqual(entries,1)
        self.assertAlmostEqual(cost,-.75-7*.008)
        continued,entries = self.samples(True,10)
        self.assertEqual(entries,0)
        self.assertAlmostEqual(continued,-.08)

    def test_unknown_gaps_do_not_charge_or_reinvent_an_entry(self):
        self.samples(False,5)
        self.samples(True,5)
        self.assertEqual(self.samples(None,20),(0,0))
        cost,entries = self.samples(True,5)
        self.assertEqual(entries,0)
        self.assertAlmostEqual(cost,-.016)

    def test_boundary_flicker_and_repeated_stale_image_cannot_confirm_crossing(self):
        self.samples(False,5)
        for _ in range(20):
            self.assertEqual(self.samples(True,2),(0,0))
            self.assertEqual(self.samples(None,2),(0,0))
        state = RewardState(opponent_zone=True,zone_sample_at=99)
        for _ in range(20):
            parts,_,_ = self.engine.update(state)
            self.assertEqual(parts["red_zone"],0)

    def test_initial_red_localization_is_occupancy_without_an_invented_crossing(self):
        cost,entries = self.samples(True,5)
        self.assertEqual(entries,0)
        self.assertAlmostEqual(cost,-.016)

    def test_replaying_a_confirmed_red_frame_does_not_keep_charging(self):
        self.samples(False,5)
        self.samples(True,5)
        for _ in range(20):
            parts,_,_ = self.engine.update(RewardState(opponent_zone=True,zone_sample_at=self.tick/10))
            self.assertEqual(parts["red_zone"],0)

    def test_leaving_and_reentering_is_a_new_crossing(self):
        self.samples(False,5)
        self.samples(True,5)
        self.assertEqual(self.samples(False,5),(0,0))
        self.assertEqual(self.samples(True,5)[1],1)

    def test_live_pose_reaches_the_actual_reward_ledger(self):
        with live_fixture() as (env,pdi):
            env.reset()
            evidence = [False]
            def scene(frame):
                return Scene(field_map={"pose_current":True,"opponent_zone":evidence[0],
                                        "pose":{"captured_at":time.perf_counter()}})
            with patch.object(env,"_analyze_frame",side_effect=scene):
                for _ in range(3):
                    env.step(0)
                evidence[0] = True
                cost = sum(env.step(0)[4]["reward_components"]["red_zone"] for _ in range(3))
            self.assertLess(cost,-.75)
            self.assertEqual(env._ledger.labels["red_zone"],"Red alliance zone")


class MinimapAndCheckpointTests(unittest.TestCase):
    def test_minimap_and_ninth_reward_row_fit_every_corner_at_720p(self):
        h = camera()
        data = FieldPoseTracker().update([h],robot_region(h,[8,4,0]),captured_at=10,now=10)
        data["zone_cost"] = {"entry":.75,"per_second":.08}
        state = {"field_map":data,"ledger":RewardLedger(labels=LABELS).snapshot(),
                 "objective":{"goal":"return","target_kind":"blue hub","reason":"Test guidance"}}
        for corner in ("top-left","top-right","bottom-left","bottom-right"):
            painter = test_debug.RendererTests.Painter()
            paint(painter,1280,720,state,{},corner=corner,now=10.1)
            self.assertIn("FIELD MAP / ESTIMATED POSE",painter.texts)
            self.assertIn("Red alliance zone",painter.texts)
            for box,_,_ in painter.rects:
                self.assertGreaterEqual(min(box),0)
                self.assertLessEqual(box[2],1280)
                self.assertLessEqual(box[3],720)
        painter = test_debug.RendererTests.Painter()
        paint(painter,1280,720,state,{},now=14)
        self.assertNotIn("POSE UNKNOWN",painter.texts)
        self.assertTrue(any("last pose held" in text for text in painter.texts))

    def test_v3_reward_upgrade_preserves_checkpoint_bytes_and_rejects_other_changes(self):
        class Model:
            num_timesteps = 2199
            def save(self,path):
                Path(path).write_bytes(b"existing learned weights")
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = save_checkpoint(Model(),folder,domain="live_screen")
            metadata = json.loads(path.with_suffix('.json').read_text())
            metadata["spec"]["reward_version"] = "independent-sensor-reward-v3"
            metadata["spec"]["reward_config"].update(score_per_point=2.0,fuel_potential=1.5,fuel_target_potential=.35,spin_per_second=.15)
            for name in ("blue_bump_per_second","blue_bump_grace_seconds","red_structure_per_second"):
                metadata["spec"]["reward_config"].pop(name)
            for name in ("red_zone_entry_cost","red_zone_per_second","zone_confirm_seconds","shift_zone_per_second","shift_zone_grace_seconds"):
                metadata["spec"]["reward_config"].pop(name)
            metadata["fingerprint"] = fingerprint(metadata["spec"])
            write_json_atomic(path.with_suffix('.json'),metadata)
            selected,restored = resolve_checkpoint(folder)
            self.assertTrue(restored["compatible_reward_extension"])
            self.assertEqual(restored["timesteps"],2199)
            self.assertEqual(hashlib.sha256(selected.read_bytes()).hexdigest(),metadata["weights_sha256"])
            metadata["spec"]["observation_dim"] += 1
            metadata["fingerprint"] = fingerprint(metadata["spec"])
            write_json_atomic(path.with_suffix('.json'),metadata)
            with self.assertRaises(CheckpointSchemaMismatch):
                resolve_checkpoint(folder)
