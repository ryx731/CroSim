import unittest
import numpy as np
import cv2
from mosim_field import FieldPoseTracker,project_fuel_clusters
from tests.test_zones import camera,project
from tests.test_field_map import robot_region


class FuelMapTests(unittest.TestCase):
    def test_invalid_camera_does_not_freeze_valid_red_side_display_pose(self):
        h = camera((10,4,2.3),(14,4,0))
        class BadCamera:
            width,height,rotation = h.width,h.height,h.rotation
            def ground_points(self,pixels,plane_height=0):
                return np.zeros((len(pixels),3)),np.zeros(len(pixels),bool)
        tracker = FieldPoseTracker()
        data = tracker.update([h,BadCamera()],robot_region(h,[14,4,0]),captured_at=10,now=10)
        self.assertTrue(data['pose_current'])
        self.assertGreater(data['pose']['x'],13)
        self.assertIsNone(data['opponent_zone'])

    def test_pose_keeps_up_when_robot_moves_past_red_bump(self):
        h = camera((9,4,2.3),(14,4,0))
        tracker = FieldPoseTracker()
        for i,x in enumerate(np.linspace(11,14,31)):
            data=tracker.update([h],robot_region(h,[x,4,0]),captured_at=10+i/10,now=10+i/10)
        self.assertTrue(data['pose_current'])
        self.assertGreater(data['pose']['x'],13)

    def test_large_fuel_group_projects_and_expires_after_tag_loss(self):
        h = camera()
        mask = np.zeros((h.height,h.width),np.uint8)
        u,v = project(h,[[8,4,0]])[0]
        cv2.circle(mask,(round(u),round(v)),9,255,-1)
        blobs=project_fuel_clusters(mask,[h],10)
        self.assertEqual(len(blobs),1)
        self.assertGreater(blobs[0]['confidence'],.9)
        tracker=FieldPoseTracker()
        tracker.update([h],robot_region(h,[8,4,0]),fuel_mask=mask,captured_at=10,now=10)
        self.assertTrue(tracker.update([],[],now=11)['fuel_clusters'])
        self.assertFalse(tracker.update([],[],now=13)['fuel_clusters'])

    def test_alpha_tracks_confidence_and_stale_blobs_disappear(self):
        from mosim_minimap import paint_minimap
        from tests.test_debug import RendererTests
        class Painter(RendererTests.Painter):
            def __init__(self):
                super().__init__();self.alphas=[]
            def blob(self,box,alpha):self.alphas.append(alpha)
        painter=Painter()
        data={'fuel_clusters':[{'x':8,'y':4,'confidence':c,'captured_at':at} for c,at in ((.2,10),(.9,10),(.9,5))]}
        paint_minimap(painter,0,0,1,data,10)
        self.assertEqual(len(painter.alphas),2)
        self.assertGreater(painter.alphas[1],painter.alphas[0])
