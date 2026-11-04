import unittest
from unittest.mock import patch
import numpy as np
from crosim_agent.rewards import CycleReward,RewardState
from crosim_agent.hud import read_hud_number
from crosim_agent.objectives import ObjectiveController
from crosim_agent.features import Scene
from crosim_agent.actions import action_bits,ACTION_INDEX
from mosim_contacts import structure_contact


class ScoreFirstTests(unittest.TestCase):
    def test_final_async_points_reach_reward_and_ledger_before_reset(self):
        import time
        from tests.test_agent import live_fixture
        with live_fixture() as (env,_):
            env.reset()
            env._async_score.thread = object()
            calls = [0]
            def update(*args):
                calls[0] += 1
                return 3 if calls[0] == 2 else 0
            env._async_score.update = update
            env._live_start = time.perf_counter()-166
            _,reward,terminal,_,info = env.step(0)
            self.assertTrue(terminal)
            self.assertEqual(info['credited_points'],3)
            self.assertEqual(env._ledger.totals['score'],10.8)
            self.assertGreater(info['reward_components']['score'],10)

    def test_late_score_without_any_shooting_pays_full_amount(self):
        parts,_,info = CycleReward().update(RewardState(elapsed=160),points=33)
        self.assertAlmostEqual(parts['score'],118.8)
        self.assertEqual(info['score_credited'],33)

    def test_ocr_disagreement_requires_a_second_matching_layout(self):
        crop = np.full((50,80,3),255,np.uint8)
        crop[10:40,20:50] = 0
        with patch('mosim_env.ocr_read',side_effect=[33,38,33]):
            self.assertEqual(read_hud_number(crop,timeout=2),33)
        with patch('mosim_env.ocr_read',side_effect=[33,38,39]):
            self.assertIsNone(read_hud_number(crop,timeout=2))

    def test_spin_lock_requires_translation_to_reopen_turns(self):
        controller = ObjectiveController()
        for _ in range(18):
            d = controller.update(Scene(),fill=0,keys=action_bits(ACTION_INDEX['aim_left']))
        self.assertFalse(d.allowed[ACTION_INDEX['aim_left']])
        for _ in range(7):
            d = controller.update(Scene(),fill=0,keys=action_bits(ACTION_INDEX['retreat']))
        self.assertTrue(d.allowed[ACTION_INDEX['aim_left']])

    def test_contact_costs_have_grace_and_do_not_use_stale_samples(self):
        engine = CycleReward()
        for i in range(1,21):
            parts,_,_ = engine.update(RewardState(blue_bump=True,zone_sample_at=i/10))
            if i <= 15:self.assertEqual(parts['contact'],0)
        self.assertAlmostEqual(parts['contact'],-.08)
        parts,_,_ = engine.update(RewardState(blue_bump=True,zone_sample_at=2))
        self.assertEqual(parts['contact'],0)
        for i in range(21,27):
            parts,_,_ = engine.update(RewardState(red_structure=True,zone_sample_at=i/10))
        self.assertAlmostEqual(parts['contact'],-.04)

    def test_structure_estimate_requires_fresh_tight_bounds(self):
        data={'pose_current':True,'pose':{'uncertainty_m':.2,'bounds':[4.5,2,4.7,2.2]}}
        self.assertEqual(structure_contact(data),(True,False))
        data['pose_current']=False
        self.assertEqual(structure_contact(data),(None,None))
