"""Fuel access tests with known 3-D cameras and real OpenCV pose fitting."""
from dataclasses import replace
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from mosim_env import CFG
from mosim_landmarks import HubAnchor
from mosim_vision import ObjectDetector
from mosim_zones import (FuelZoneFilter, CameraHypothesis, tag_world_corners,
                         FIELD_LENGTH, ALLIANCE_ZONE_DEPTH, ZONE_MARGIN)
from crosim_agent.actions import action_bits, ACTION_INDEX
from crosim_agent.features import FeatureExtractor, Scene
from crosim_agent.objectives import ObjectiveController


def camera(center=(6,4,2.3), target=(10,4,0), fov=65):
    center = np.array(center,dtype=float)
    forward = np.array(target)-center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward,[0,0,1])
    right /= np.linalg.norm(right)
    down = np.cross(forward,right)
    return CameraHypothesis(np.array([right,down,forward]),center,
                            540/(2*math.tan(math.radians(fov)/2)),960,540,0.0)


def project(hypothesis, points):
    camera_points = (np.asarray(points)-hypothesis.center) @ hypothesis.rotation.T
    return camera_points[:,:2]/camera_points[:,2:]*hypothesis.focal+[hypothesis.width/2,hypothesis.height/2]


def anchors_for(hypothesis, ids=(3,4), size=0.15, captured_at=10):
    anchors = []
    for tag in ids:
        corners = project(hypothesis,tag_world_corners(tag,size))/[960,540]
        low, high = corners.min(axis=0),corners.max(axis=0)
        scale = np.linalg.norm((corners-np.roll(corners,1,axis=0))*[960,540],axis=1).mean()/960
        kind = "outpost" if tag in (13,14,29,30) else "hub"
        anchors.append(HubAnchor(tag,"red" if tag<=16 else "blue",tuple([*low,*high]),
                                 tuple(corners.mean(axis=0)),float(scale),captured_at,
                                 tuple(tuple(point) for point in corners),kind))
    return anchors


class ZoneGeometryTests(unittest.TestCase):
    def test_ground_projection_round_trip_and_boundary_margin(self):
        h = camera()
        points = [[8,4,0.075],[FIELD_LENGTH-ALLIANCE_ZONE_DEPTH-0.1,4,0.075],[15.8,4,0.075]]
        pixels = project(h,points)
        reconstructed, valid = h.ground_points(pixels)
        np.testing.assert_allclose(reconstructed,points,atol=1e-10)
        self.assertTrue(valid.all())
        np.testing.assert_array_equal(FuelZoneFilter().point_eligibility(pixels,[h]),[True,False,False])

    def test_actual_pose_fit_rejects_red_fuel_and_keeps_neutral_fuel(self):
        h = camera()
        mask = np.zeros((540,960),np.uint8)
        neutral, red = project(h,[[8.5,4,0.075],[15.5,4,0.075]])
        cv2.circle(mask,tuple(np.round(neutral).astype(int)),4,255,-1)
        cv2.circle(mask,tuple(np.round(red).astype(int)),13,255,-1)
        result = FuelZoneFilter().apply(mask,anchors_for(h),now=10)
        self.assertGreater(result.hypotheses,0)
        self.assertTrue(result.eligible[round(neutral[1]),round(neutral[0])])
        self.assertFalse(result.eligible[round(red[1]),round(red[0])])
        self.assertGreater(result.excluded_regions,0)

    def test_outpost_floor_and_raised_chute_fuel_are_both_rejected(self):
        h = camera((10.5,7,2.3),(16.5,7,0.5))
        gate = FuelZoneFilter()
        fits, ids = gate.estimate(anchors_for(h,(13,14)),960,540,now=10)
        self.assertTrue(fits)
        self.assertEqual(ids,(13,14))
        points = project(h,[[15.5,7,0.075],[16.2,7,1.3]])
        self.assertFalse(gate.point_eligibility(points,fits).any())

    def test_red_alliance_filter_mirrors_the_excluded_end(self):
        h = camera((10.5,4,2.3),(6.5,4,0))
        pixels = project(h,[[8,4,0.075],[0.5,4,0.075]])
        np.testing.assert_array_equal(FuelZoneFilter("red").point_eligibility(pixels,[h]),[True,False])

    def test_no_stale_single_or_malformed_tag_can_reuse_old_pose(self):
        h = camera()
        gate = FuelZoneFilter()
        mask = np.zeros((540,960),np.uint8)
        mask[340:350,400:410] = 255
        anchors = anchors_for(h)
        self.assertTrue(gate.apply(mask,anchors,now=10).hypotheses)
        for rejected in ([],anchors[:1],[replace(a,captured_at=9) for a in anchors],
                         [replace(a,corners=((float("nan"),0),)*4) for a in anchors]):
            result = gate.apply(mask,rejected,now=10)
            self.assertFalse(result.hypotheses)
            self.assertFalse(result.eligible.any())
            self.assertTrue(result.excluded.any())

    def test_disagreement_or_sky_pixels_are_not_assumed_accessible(self):
        h = camera()
        pixel = project(h,[[8,4,0.075]])
        self.assertTrue(FuelZoneFilter().point_eligibility(pixel,[h])[0])
        shifted = replace(h,center=h.center+np.array([8,0,0]))
        self.assertFalse(FuelZoneFilter().point_eligibility(pixel,[h,shifted])[0])
        self.assertFalse(FuelZoneFilter().point_eligibility(np.array([[480,-1000]]),[h])[0])

    def test_component_touching_exclusion_is_removed_whole(self):
        h = camera()
        boundary = FIELD_LENGTH-ALLIANCE_ZONE_DEPTH-ZONE_MARGIN
        pixel = project(h,[[boundary,4,0.075]])[0]
        mask = np.zeros((540,960),np.uint8)
        cv2.circle(mask,tuple(np.round(pixel).astype(int)),10,255,-1)
        gate = FuelZoneFilter()
        with patch.object(gate,"estimate",return_value=([h],(3,4))):
            result = gate.apply(mask)
        self.assertFalse(result.eligible.any())
        self.assertTrue(np.array_equal(result.excluded,mask))


class ZoneIntegrationTests(unittest.TestCase):
    def frame(self):
        h = camera()
        frame = np.zeros((540,960,3),np.uint8)
        good,bad = project(h,[[8.5,4,0.075],[15.5,4,0.075]])
        cv2.circle(frame,tuple(np.round(good).astype(int)),5,(0,255,255),-1)
        cv2.circle(frame,tuple(np.round(bad).astype(int)),15,(0,255,255),-1)
        config = dict(CFG)
        for name in ("score","hopper","hub"):
            config[name+"_region"] = [0,0,0.01,0.01]
        return frame,config,anchors_for(h),good,bad

    def test_filter_precedes_grid_nearest_and_largest_cluster_selection(self):
        frame,config,anchors,good,bad = self.frame()
        extractor = FeatureExtractor(config)
        with patch.object(extractor.tags,"detect",return_value=anchors),patch("mosim_zones.time.perf_counter",return_value=10):
            scene = extractor.analyze(frame,captured_at=10)
        self.assertTrue(scene.fuel_visible)
        self.assertGreater(scene.fuel_excluded,0)
        self.assertGreater(scene.fuel_y,0.6)
        self.assertGreater(scene.dense_y,0.6)
        self.assertEqual(scene.grid[0].sum(),0)

    def test_default_vision_marks_unverified_fuel_as_non_collectible(self):
        frame,config,_,_,_ = self.frame()
        detector = ObjectDetector(config,discover_weights=False)
        with patch.object(detector.hub_tags,"detect",return_value=[]):
            boxes = detector.detect(frame)
        self.assertTrue(boxes)
        self.assertTrue(all(not box.collectible for box in boxes))

    def test_optional_yolo_cannot_bypass_zone_exclusion(self):
        import torch
        frame,config,anchors,good,bad = self.frame()
        detector = ObjectDetector(config,discover_weights=False)
        detector.names = {0:"fuel"}
        rows = [[x-4,y-4,x+4,y+4,0.9,0] for x,y in (good,bad)]
        detector.model = SimpleNamespace(predict=lambda **kwargs:[SimpleNamespace(boxes=SimpleNamespace(data=torch.tensor(rows)))])
        with patch.object(detector.hub_tags,"detect",return_value=anchors),patch("mosim_zones.time.perf_counter",return_value=10):
            boxes = detector.detect(frame,captured_at=10)
        self.assertTrue(boxes[0].collectible)
        self.assertFalse(boxes[1].collectible)

    def test_only_excluded_fuel_never_produces_a_forward_intake_target(self):
        scene = Scene(fuel_excluded=12,zone_status="zone unknown: fuel skipped")
        for fill in (0.05,0.3,0.95):
            controller = ObjectiveController()
            for tick in range(20):
                decision = controller.update(scene,fill=fill,keys=action_bits(ACTION_INDEX["aim_left"]))
                self.assertFalse(decision.target_visible)
                for index,allowed in enumerate(decision.allowed):
                    if allowed:
                        self.assertFalse(action_bits(index)[0])
                        self.assertFalse(action_bits(index)[6])
            self.assertIn("retreat",decision.snapshot()["allowed_actions"])
            self.assertIn("retreat",decision.snapshot()["allowed_actions"])

    def test_excluded_fuel_alone_does_not_toggle_partial_load_between_goals(self):
        scene = Scene(fuel_excluded=10,hub_visible=True,hub_scale=0.1)
        controller = ObjectiveController()
        decision = controller.update(scene,fill=0.3)
        self.assertFalse(controller.returning)
        self.assertTrue(all(not action_bits(i)[6] for i,allowed in enumerate(decision.allowed) if allowed))
        if decision.target_visible:
            self.assertEqual(decision.target_kind,"blue hub marker")

    def test_proxy_red_fuel_cannot_dominate_the_target_or_be_picked_up(self):
        from crosim_agent.proxy import PracticeEnv, RED_ZONE_START
        env = PracticeEnv()
        env.reset(seed=14)
        env.position = np.array([RED_ZONE_START-0.1,0.0])
        env.yaw = 0.0
        env.hopper_open = True
        env.fuel_remaining[:] = 0
        env.fuel[0] = [RED_ZONE_START+0.2,0.0]
        env.fuel_remaining[0] = 1000
        scene = env._scene()
        self.assertFalse(scene.fuel_visible)
        self.assertEqual(scene.fuel_excluded,1)
        env.step(ACTION_INDEX["collect_forward"])
        self.assertEqual(env.hopper,0)
        env.close()


if __name__ == "__main__":
    unittest.main()
