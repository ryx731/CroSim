import unittest
from unittest.mock import patch
import cv2
import numpy as np
from mosim_overlay import TkPainter
from crosim_agent.hud import HudPanelExtractor,ShiftArrowTracker


class HudPolishTests(unittest.TestCase):
    def test_locked_small_arrow_box_ignores_round_fuel_in_old_search_strip(self):
        frame=np.zeros((1080,1920,3),np.uint8)
        cv2.rectangle(frame,(478,950),(854,1027),(220,80,10),-1)
        cv2.rectangle(frame,(1060,950),(1440,1027),(10,10,230),-1)
        cv2.rectangle(frame,(488,970),(531,1013),(0,255,255),-1)
        panel=HudPanelExtractor({'score_region':[.36,.84,.45,.9]});tracker=ShiftArrowTracker()
        for i in range(3):
            panel(frame);state=tracker.update(frame,panel.arrow_region,captured_at=i/10,now=i/10)
        self.assertIs(state,True)
        locked=list(panel.arrow_region)
        self.assertLess((locked[2]-locked[0])*1920,50)
        cv2.rectangle(frame,(488,970),(531,1013),(220,80,10),-1)
        cv2.circle(frame,(430,990),22,(0,255,255),-1)
        for i in range(3,6):
            panel(frame);state=tracker.update(frame,panel.arrow_region,captured_at=i/10,now=i/10)
        np.testing.assert_allclose(panel.arrow_region,locked)
        self.assertIs(state,False)

    def test_native_blobs_use_distinct_real_alpha_channels_and_cache_images(self):
        class Canvas:
            def create_image(self,*args,**kwargs):pass
        painter=TkPainter(Canvas())
        with patch('PIL.ImageTk.PhotoImage',side_effect=lambda bitmap,**kwargs:bitmap.copy()) as make:
            painter.blob((0,0,30,30),.2)
            painter.blob((0,0,30,30),.8)
            painter.blob((0,0,30,30),.8)
        self.assertEqual(make.call_count,2)
        self.assertLess(painter.images[0].getpixel((15,15))[3],painter.images[1].getpixel((15,15))[3])
