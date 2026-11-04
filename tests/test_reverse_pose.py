import math
import unittest
from unittest.mock import patch
import numpy as np
from mosim_field import FieldPoseTracker
from mosim_zones import FuelZoneFilter
from crosim_agent.features import Scene
from tests.test_zones import camera,anchors_for
from tests.test_field_map import robot_region
from tests.test_agent import live_fixture


class ReversePoseTests(unittest.TestCase):
    def test_same_robot_pose_survives_reversed_camera_without_180_heading_jump(self):
        tracker=FieldPoseTracker()
        normal=camera((6,4,2.3),(10,4,0))
        reverse=camera((10,4,2.3),(6,4,0))
        first=tracker.update([normal],robot_region(normal,[7.75,4,0]),captured_at=10,now=10)
        held=tracker.update([],[],captured_at=10.1,now=10.1,reversed_view=True)
        self.assertEqual(held['pose']['captured_at'],10)
        second=tracker.update([reverse],robot_region(reverse,[8.25,4,0]),captured_at=10.3,now=10.3,reversed_view=True)
        self.assertTrue(second['pose_current'])
        np.testing.assert_allclose([second['pose']['x'],second['pose']['y']],[8,4],atol=1e-6)
        self.assertLess(abs(second['pose']['heading']-first['pose']['heading']),1e-6)
        self.assertEqual(second['pose']['camera_view'],'reversed')

    def test_live_reversed_view_is_enabled_after_settling_and_keeps_pose_cache(self):
        with live_fixture() as (env,_):
            env.reset();env._view_parity=1;env._view_change_steps=0
            with patch.object(env.extractor,'analyze',return_value=Scene()) as analyze:
                env._analyze_frame(env._latest_frame)
                self.assertTrue(analyze.call_args.kwargs['pose_enabled'])
                self.assertTrue(analyze.call_args.kwargs['reversed_view'])
                env._view_change_steps=2
                env._analyze_frame(env._latest_frame)
                self.assertFalse(analyze.call_args.kwargs['pose_enabled'])

    def test_joint_hub_and_field_tags_constrain_the_same_camera_pose(self):
        h=camera()
        anchors=anchors_for(h,(3,4,1))
        fits,ids=FuelZoneFilter().estimate(anchors,960,540,now=10)
        self.assertEqual(ids,(1,3,4))
        self.assertTrue(fits)
        pair,_=FuelZoneFilter().estimate(anchors_for(h,(3,4)),960,540,now=10)
        self.assertLess(min(np.linalg.norm(f.center-h.center) for f in fits),.15)
        self.assertLess(max(np.linalg.norm(f.center-h.center) for f in fits),
                        max(np.linalg.norm(f.center-h.center) for f in pair))

