"""Missed-end watchdog, observed shift costs, and retained field-state regressions."""
import time
import unittest

import numpy as np

from crosim_agent.hud import HudReader,HudPanelExtractor,ShiftArrowTracker
from crosim_agent.rewards import CycleReward,RewardConfig,RewardState
from mosim_field import FieldPoseTracker,FIELD
from mosim_layout import panel_geometry
from tests.test_agent import live_fixture


class WatchdogTests(unittest.TestCase):
    def test_positive_ocr_cannot_prevent_165_second_reset_or_cause_repeated_resets(self):
        with live_fixture() as (env,pdi):
            env.reset()
            now = time.perf_counter()
            env.match_clock.observe(40,now,now=now)
            env._live_start = now-164
            self.assertFalse(env._episode_has_ended(env._elapsed()))
            self.assertFalse(env._tap_guarded('r'))
            env._live_start = now-166
            _,_,terminal,_,info = env.step(0)
            self.assertTrue(terminal)
            self.assertEqual(info['reason'],'2:45 wall-clock watchdog')
            env.reset()
            env.reset()
            self.assertEqual(pdi.events.count(('down','r')),1)

    def test_frozen_capture_wait_reaches_watchdog_without_ocr_or_extra_movement(self):
        with live_fixture() as (env,pdi):
            env.reset()
            pdi.events.clear()
            elapsed = [164.9]
            env._elapsed = lambda: elapsed[0]
            env._latest_capture_at = time.perf_counter()-10
            frame = env._latest_frame
            def frozen_grab():
                elapsed[0] = 165.1
                return frame
            env._grab_focused = frozen_grab
            _,_,terminal,_,_ = env.step(1)
            self.assertTrue(terminal)
            self.assertNotIn(('down','w'),pdi.events)
            env.reset()
            self.assertEqual(pdi.events.count(('down','r')),1)

    def test_recovered_capture_reobserves_before_grading_a_stale_action(self):
        with live_fixture() as (env,pdi):
            env.reset()
            pdi.events.clear()
            env._latest_capture_at = time.perf_counter()-10
            _,reward,terminal,truncated,_ = env.step(1)
            self.assertTrue(truncated)
            self.assertFalse(terminal)
            self.assertEqual(reward,0)
            self.assertFalse(pdi.events)


class ShiftRewardTests(unittest.TestCase):
    def test_scoring_multiplier_is_eighty_percent_larger(self):
        self.assertEqual(RewardConfig().score_per_point,2*1.8)

    def test_both_wrong_zone_cases_charge_but_both_correct_cases_do_not(self):
        for shift,blue in ((True,False),(False,True),(True,True),(False,False)):
            engine = CycleReward()
            costs = []
            for step in range(1,41):
                state = RewardState(shift_on=shift,blue_zone=blue,elapsed=step/10,zone_sample_at=step/10)
                parts,_,_ = engine.update(state)
                costs.append(parts['shift_zone'])
            self.assertEqual(sum(costs[:19]),0)
            if shift != blue:
                self.assertAlmostEqual(costs[-1],-.01)
                self.assertLess(sum(costs),-.15)
            else:
                self.assertEqual(sum(costs),0)

    def test_uncertain_shift_boundary_and_repeated_pose_pause_cost(self):
        engine = CycleReward(RewardConfig(shift_zone_grace_seconds=0,zone_confirm_seconds=0))
        for shift,blue,at in ((True,False,1),(None,False,2),(True,None,3),(True,False,1)):
            parts,_,_ = engine.update(RewardState(shift_on=shift,blue_zone=blue,zone_sample_at=at,elapsed=4))
            self.assertAlmostEqual(parts['shift_zone'],-.01 if at == 1 and shift is True and engine.steps == 1 else 0)

    def test_shift_change_starts_a_new_travel_grace(self):
        engine = CycleReward()
        for step in range(1,40):
            engine.update(RewardState(shift_on=True,blue_zone=True,zone_sample_at=step/10,elapsed=step/10))
        parts,_,_ = engine.update(RewardState(shift_on=False,blue_zone=True,zone_sample_at=4,elapsed=4))
        self.assertEqual(parts['shift_zone'],0)

    def test_live_shift_uses_arrow_vision_rather_than_schedule_or_stale_capture(self):
        with live_fixture() as (env,_):
            env.reset()
            env.hud_panel.status = 'auto blue scorecard: alliance score'
            env._hub.source = 'schedule'
            env._hub.vision.state = True
            self.assertIsNone(env._observed_shift())
            env._hub.source = 'vision'
            env.shift_arrow.state = True
            env.shift_arrow.valid = True
            env.shift_arrow.at = time.perf_counter()
            self.assertTrue(env._observed_shift())
            env.shift_arrow.state = False
            self.assertFalse(env._observed_shift())
            env._latest_capture_at -= 1
            self.assertIsNone(env._observed_shift())


class FieldStateAndRefreshTests(unittest.TestCase):
    def test_arrow_follows_scorecard_and_detects_on_off_in_both_layouts(self):
        import cv2
        for left in (435,488):
            frame = np.zeros((1080,1920,3),np.uint8)
            cv2.rectangle(frame,(478,950),(854,1027),(220,80,10),-1)
            cv2.rectangle(frame,(1060,950),(1440,1027),(10,10,230),-1)
            panel = HudPanelExtractor({'score_region':[.36,.84,.45,.90]})
            tracker = ShiftArrowTracker()
            for index in range(3):
                panel(frame)
                result = tracker.update(frame,panel.arrow_region,captured_at=index/10,now=index/10)
            self.assertFalse(result)
            cv2.rectangle(frame,(left,970),(left+43,1013),(0,255,255),-1)
            cv2.arrowedLine(frame,(left+34,991),(left+9,991),(0,0,0),5,tipLength=.5)
            for index in range(3,6):
                panel(frame)
                result = tracker.update(frame,panel.arrow_region,captured_at=index/10,now=index/10)
            self.assertTrue(result)
            self.assertIsNone(tracker.update(frame,None,captured_at=.6,now=.6))

    def test_region_state_debounces_crossings_and_holds_through_missing_tags(self):
        tracker = FieldPoseTracker()
        tracker.pose['x'] = 2
        self.assertEqual(tracker.track_region(True,10)['region'],'blue')
        tracker.pose['x'] = FIELD['blue_end']+.1
        self.assertEqual(tracker.track_region(True,10.1)['region'],'blue')
        tracker.pose['x'] = 8
        self.assertEqual(tracker.track_region(True,10.2)['region'],'blue')
        self.assertEqual(tracker.track_region(True,10.6)['region'],'neutral')
        held = tracker.track_region(False,20)
        self.assertEqual(held['region'],'neutral')
        self.assertIn('held',held['source'])

    def test_compact_hud_and_attached_state_tracker_fit_720p(self):
        from crosim_agent.rewards import LABELS
        boxes = panel_geometry(1280,720,{'ledger':{'labels':LABELS}})
        self.assertLessEqual(boxes['reward'][3],620)
        self.assertEqual(boxes['minimap'][3],418)
        for name in ('reward','minimap'):
            x,y,w,h = boxes[name]
            self.assertTrue(0 <= x <= 1280-w and 0 <= y <= 720-h)

    def test_unchanged_score_is_rechecked_after_half_a_second(self):
        reader = HudReader()
        calls = []
        reader.score.read_crop = lambda *args,**kwargs: calls.append(1) or 0
        crop = np.full((96,168,3),255,np.uint8)
        for at in (0,.1,.3,.7):
            reader.score.update_crop(crop,at)
        self.assertEqual(len(calls),3)

