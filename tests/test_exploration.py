"""Regression coverage for the live policy's repeated-hopper/no-driving trap."""
import unittest

import numpy as np
import torch

from crosim_agent.actions import ACTIONS,ACTION_INDEX,action_bits
from crosim_agent.features import Scene,FeatureHistory,pack_features
from crosim_agent.objectives import ObjectiveController,GUIDANCE_SCALE
from crosim_agent.policy import EXPLORATION_FRACTION
from crosim_agent.proxy import PracticeEnv
from crosim_agent.training import make_model


class GuidedExplorationTests(unittest.TestCase):
    def setUp(self):
        self.env = PracticeEnv()
        self.addCleanup(self.env.close)
        self.model = make_model(self.env)
        # Reproduce the failed session: an effectively certain hopper policy.
        with torch.no_grad():
            self.model.policy.action_net.weight.zero_()
            self.model.policy.action_net.bias.zero_()
            self.model.policy.action_net.bias[ACTION_INDEX['toggle_hopper']] = 30

    def observation(self,scene=None,controller=None,**kwargs):
        scene = scene or Scene(fuel_visible=True,fuel_y=.8,fuel_mass=.2)
        decision = (controller or ObjectiveController()).update(scene,fill=0,**kwargs)
        return FeatureHistory().reset(pack_features(scene,objective=decision)),decision

    def test_collapsed_hopper_policy_still_explores_driving_with_matching_ppo_log_probs(self):
        obs,_ = self.observation()
        tensor = torch.as_tensor(np.repeat(obs[None,:],512,axis=0))
        torch.manual_seed(9)
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(tensor).distribution
            self.assertAlmostEqual(distribution.probs[0,ACTION_INDEX['collect_forward']].item(),
                                   GUIDANCE_SCALE+EXPLORATION_FRACTION/len(ACTIONS),places=5)
            actions,_,log_prob = self.model.policy(tensor)
            _,evaluated,_ = self.model.policy.evaluate_actions(tensor,actions)
        torch.testing.assert_close(log_prob,evaluated)
        self.assertGreater((actions == ACTION_INDEX['collect_forward']).sum().item(),150)
        self.assertTrue(torch.isfinite(evaluated).all())

    def test_guidance_and_random_exploration_respect_excluded_fuel_mask(self):
        obs,decision = self.observation(Scene(fuel_excluded=8))
        with torch.no_grad():
            probabilities = self.model.policy.get_distribution(torch.as_tensor(obs[None,:])).distribution.probs[0]
        for index,allowed in enumerate(decision.allowed):
            if not allowed:
                self.assertEqual(probabilities[index].item(),0)
        self.assertAlmostEqual(probabilities.sum().item(),1,places=6)

    def test_hopper_cooldown_blocks_repeat_and_then_restores_choice(self):
        controller = ObjectiveController()
        _,before = self.observation(controller=controller)
        self.assertTrue(before.allowed[ACTION_INDEX['toggle_hopper']])
        _,after = self.observation(controller=controller,keys=action_bits(ACTION_INDEX['toggle_hopper']))
        self.assertFalse(after.allowed[ACTION_INDEX['toggle_hopper']])
        for _ in range(31):
            _,after = self.observation(controller=controller,keys=action_bits(ACTION_INDEX['coast']))
        self.assertTrue(after.allowed[ACTION_INDEX['toggle_hopper']])

    def test_scoring_and_recovery_get_guided_probability_without_forcing_actions(self):
        scene = Scene(hub_visible=True,hub_x=0,hub_scale=.4)
        decision = ObjectiveController().update(scene,fill=.95)
        obs = FeatureHistory().reset(pack_features(scene,objective=decision))
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(torch.as_tensor(obs[None,:])).distribution
        self.assertGreaterEqual(distribution.probs[0,ACTION_INDEX['shoot']].item(),GUIDANCE_SCALE)
        self.assertGreater(distribution.probs[0,ACTION_INDEX['toggle_hopper']].item(),.5)

    def test_missing_target_guidance_translates_after_short_search_turn(self):
        controller = ObjectiveController()
        controller.update(Scene(),fill=0,keys=action_bits(ACTION_INDEX['aim_left']),dt=1)
        obs,_ = self.observation(Scene(),controller)
        with torch.no_grad():
            probabilities = self.model.policy.get_distribution(torch.as_tensor(obs[None,:])).distribution.probs[0]
        self.assertGreaterEqual(probabilities[ACTION_INDEX['retreat']].item(),GUIDANCE_SCALE)

