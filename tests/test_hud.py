from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from mosim_env import CFG
from crosim_agent.hud import HudPanelExtractor, HudReader, InventoryTracker, InventorySnapshot
from crosim_agent.features import Scene
from crosim_agent.objectives import ObjectiveController
from crosim_agent.rewards import CycleReward, RewardState
from tests.test_agent import live_fixture


class HudTests(unittest.TestCase):
    def test_scorecard_locator_uses_large_alliance_score_and_tracks_layout(self):
        frame = np.zeros((1080,1920,3),np.uint8)
        cv2.rectangle(frame,(480,920),(860,1000),(180,80,0),-1)
        cv2.rectangle(frame,(1060,920),(1440,1000),(0,0,220),-1)
        cv2.rectangle(frame,(860,920),(1060,1000),(255,255,255),-1)
        cv2.putText(frame,"12",(565,976),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        cv2.putText(frame,"57",(745,980),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),3)
        extractor = HudPanelExtractor(CFG)
        panel = extractor(frame)
        self.assertIn("auto blue",extractor.status)
        self.assertEqual(panel.shape,(96,720,3))
        self.assertGreater(extractor.timer_region[0],extractor.score_region[0])
        self.assertGreater(extractor.score_region[0],680/1920)
        self.assertGreater(extractor.score_region[1],0.8)

    def test_score_confirms_zero_then_awards_increase(self):
        panel = np.zeros((96,400,3),np.uint8)
        with patch("crosim_agent.hud.read_hud_number",side_effect=[0,0,3,3]):
            reader = HudReader()
            self.assertEqual(reader.update_crop(panel,0),0)
            self.assertEqual(reader.update_crop(panel,0.3),0)
            self.assertEqual(reader.confirmed,0)
            self.assertEqual(reader.update_crop(panel,3),0)
            self.assertEqual(reader.update_crop(panel,3.3),3)

    @staticmethod
    def hopper_frame(loaded=False):
        frame = np.zeros((192,192,3),np.uint8)
        frame[10:140,22:170] = [65,65,65]
        if loaded:
            for y in (32,75,118):
                for x in (46,80,114,148):
                    cv2.circle(frame,(x,y),12,(0,255,255),-1)
        return frame

    @staticmethod
    def tracker():
        return InventoryTracker({**CFG,"hopper_region":[0,0,1,1]})

    def test_ball_shapes_spread_inside_basket_establish_a_load(self):
        tracker = self.tracker()
        for at in (10,10.1,10.2):
            result = tracker.update(self.hopper_frame(True),now=at)
        self.assertTrue(result.known)
        self.assertGreaterEqual(result.fill,.8)
        self.assertGreaterEqual(result.visual_balls,8)

    def test_yellow_scenery_at_roi_edges_does_not_fill_basket(self):
        tracker = self.tracker()
        frame = self.hopper_frame()
        frame[:8] = [0,255,255]
        frame[:140,:10] = [0,255,255]
        frame[:140,182:] = [0,255,255]
        for at in (10,10.1,10.2):
            result = tracker.update(frame,now=at)
        self.assertTrue(result.known)
        self.assertEqual(result.fill,0)

    def test_large_yellow_patch_without_multiple_ball_shapes_cannot_mean_full(self):
        tracker = self.tracker()
        frame = self.hopper_frame()
        cv2.rectangle(frame,(50,40),(140,105),(0,255,255),-1)
        for at in (10,10.1,10.2):
            result = tracker.update(frame,now=at)
        self.assertLess(result.fill,.8)

    def test_short_empty_glimpse_does_not_erase_a_sustained_load(self):
        tracker = self.tracker()
        for index in range(7):
            result = tracker.update(self.hopper_frame(True),now=10+index*.1)
        for at in (10.7,10.8):
            result = tracker.update(self.hopper_frame(),now=at)
        self.assertGreaterEqual(result.fill,.8)

    def test_missing_basket_or_stale_capture_holds_estimate_as_uncertain(self):
        tracker = self.tracker()
        for at in (10,10.1,10.2):
            first = tracker.update(self.hopper_frame(True),now=at)
        lost = tracker.update(np.full((192,192,3),[0,255,255],np.uint8),now=10.3)
        self.assertEqual(first.fill,lost.fill)
        self.assertFalse(lost.known)
        stale = tracker.update(self.hopper_frame(),captured_at=10,now=14)
        self.assertEqual(stale.fill,first.fill)
        self.assertFalse(stale.known)

    def test_repeated_old_image_cannot_supply_multiple_confirmations(self):
        tracker = self.tracker()
        for at in (10,10.1,10.2):
            result = tracker.update(self.hopper_frame(True),captured_at=10,now=at)
        self.assertFalse(result.known)

    def test_fully_obscured_black_view_is_unknown_rather_than_empty(self):
        tracker = self.tracker()
        for at in (10,10.1,10.2):
            result = tracker.update(np.zeros((192,192,3),np.uint8),now=at)
        self.assertFalse(result.known)

    def test_local_cyan_rim_tracks_a_shifted_basket_reading_box(self):
        tracker = InventoryTracker({**CFG,"hopper_region":[.3,.3,.7,.9]})
        frame = np.zeros((400,600,3),np.uint8)
        cv2.rectangle(frame,(210,140),(390,330),(190,255,0),2)
        first,found = tracker.basket_region(frame)
        self.assertTrue(found)
        moved = cv2.warpAffine(frame,np.float32([[1,0,10],[0,1,10]]),(600,400))
        second,found = tracker.basket_region(moved)
        self.assertTrue(found)
        np.testing.assert_allclose(np.array(second)-first,[10/600,10/400,10/600,10/400],atol=.003)

    def test_first_inventory_baseline_does_not_award_preexisting_fuel(self):
        engine = CycleReward()
        engine.reset(RewardState(fill=0,fill_valid=False))
        parts,_,_ = engine.update(RewardState(fill=0.9,fill_valid=True))
        self.assertLessEqual(parts["fuel"],0)


class StableGoalTests(unittest.TestCase):
    def test_low_partial_load_no_longer_switches_when_excluded_clusters_flicker(self):
        controller = ObjectiveController()
        for index in range(100):
            scene = Scene(fuel_excluded=10 if index%2 else 0)
            decision = controller.update(scene,fill=0.2,confirm_seconds=1)
            self.assertFalse(controller.returning)
            self.assertNotEqual(decision.goal,"shoot")

    def test_loaded_goal_latches_through_low_or_unknown_views_until_confirmed_empty(self):
        controller = ObjectiveController()
        scene = Scene()
        for _ in range(12):
            controller.update(scene,fill=1,confirm_seconds=1)
        self.assertTrue(controller.returning)
        for fill in (0.3,0.1,0.05):
            controller.update(scene,fill=fill,confirm_seconds=1)
            self.assertTrue(controller.returning)
        for _ in range(20):
            controller.update(scene,fill=0,fill_known=False,confirm_seconds=1)
        self.assertTrue(controller.returning)
        for _ in range(12):
            controller.update(scene,fill=0,confirm_seconds=1)
        self.assertFalse(controller.returning)

    def test_space_is_held_through_a_brief_hub_detection_loss(self):
        controller = ObjectiveController()
        scene = Scene(hub_visible=True,hub_x=0.4,hub_scale=0.15)
        decision = controller.update(scene,fill=0.9)
        self.assertIn("shoot",decision.snapshot()["allowed_actions"])
        self.assertGreater(sum(decision.allowed),4)
        scene.hub_visible = False
        for _ in range(10):
            self.assertEqual(controller.update(scene,fill=0.9).goal,"shoot")

    def test_unknown_inventory_does_not_clear_a_return_latch_or_force_idle(self):
        controller = ObjectiveController()
        controller.update(Scene(),fill=1)
        decision = controller.update(Scene(),fill=0.1,fill_known=False)
        self.assertTrue(controller.returning)
        self.assertNotEqual(decision.snapshot()["allowed_actions"],["coast"])

    def test_known_neutral_side_hub_face_does_not_start_a_shot_probe(self):
        controller = ObjectiveController()
        scene = Scene(hub_visible=True,hub_x=0,hub_scale=.2,tag_ids=[19,20])
        self.assertNotEqual(controller.update(scene,fill=.9).goal,"shoot")

    def test_unreliable_views_interrupt_full_confirmation(self):
        controller = ObjectiveController()
        for _ in range(5):
            controller.update(Scene(),fill=1,confirm_seconds=1)
        controller.update(Scene(),fill=1,fill_known=False,confirm_seconds=1)
        for _ in range(5):
            controller.update(Scene(),fill=1,confirm_seconds=1)
        self.assertFalse(controller.returning)


class ActualInputTests(unittest.TestCase):
    def test_manual_space_can_credit_confirmed_points_even_when_command_is_not_shoot(self):
        with live_fixture() as (env,pdi):
            env.reset()
            env._guard.user = SimpleNamespace(GetAsyncKeyState=lambda key: 0x8000 if key==0x20 else 0)
            values = iter([2,0])
            env._score_reader.update = lambda frame,now: next(values)
            _,_,_,_,info = env.step(0)
            self.assertEqual(info["credited_points"],2)
            self.assertEqual(info["reward_components"]["score"],2*env.reward_config.score_per_point)
            self.assertTrue(info["reward_diagnostics"]["observed_space"])

    def test_loaded_shoot_mask_sends_a_real_space_key_in_the_input_driver(self):
        from crosim_agent.training import make_model
        from crosim_agent.proxy import PracticeEnv
        model = make_model(PracticeEnv())
        import torch
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()
            model.policy.action_net.bias[11] = 20
        with live_fixture() as (env,pdi):
            env.reset()
            env.inventory.snapshot = InventorySnapshot(fill=.95,known=True)
            env.inventory.update = lambda *args,**kwargs: env.inventory.snapshot
            env._update_shaping(env._latest_frame,True,False)
            env._hub_active = True
            env.scene = Scene(hub_visible=True,hub_x=0.4,hub_scale=0.15)
            for _ in range(15):
                env._update_objective(dt=0.1)
            obs = env.feature_history.reset(env._feature_frame())
            action,_ = model.predict(obs,deterministic=True)
            env.step(action)
            self.assertIn(("down","space"),pdi.events)
            self.assertIn("space",env._held_keys)
            model.get_env().close()


if __name__ == "__main__":
    unittest.main()
