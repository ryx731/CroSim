from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from stable_baselines3 import PPO
from crosim_agent.actions import ACTIONS,ACTION_INDEX,action_bits,tick_bits
from crosim_agent.features import Scene,pack_features,FeatureHistory
from crosim_agent.objectives import ObjectiveController
from crosim_agent.demonstrations import HumanMatch,exact_action,load_matches,fit_human
from crosim_agent.rewards import RewardConfig
from crosim_agent.training import make_model
from crosim_agent.proxy import PracticeEnv
from crosim_agent.checkpoints import save_checkpoint


def observation(x=0):
    scene=Scene(fuel_visible=True,fuel_x=x,fuel_y=.8)
    decision=ObjectiveController().update(scene,fill=0)
    return FeatureHistory().reset(pack_features(scene,objective=decision))


class HumanLearningTests(unittest.TestCase):
    def test_teach_records_human_input_without_injecting_keys_then_fits_on_stop(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from crosim_agent import cli,demonstrations
        from tests.test_agent import live_fixture
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory,live_fixture(inject_inputs=False) as (env,pdi):
            args=cli.parser().parse_args(['teach','--demos',directory,'--models',directory])
            env._guard.user=SimpleNamespace(GetAsyncKeyState=lambda key:0)
            original=env.watch_step;calls=[0]
            def step(keys):
                calls[0]+=1
                if calls[0]>80:raise KeyboardInterrupt
                return original(keys)
            with patch.object(cli,'HERE',Path(directory)),patch.object(cli,'require_ready'),patch('crosim_agent.live.LiveFeatureEnv',return_value=env),patch('mosim_env.read_keys_now',return_value=action_bits(1)),patch.object(env,'watch_step',side_effect=step),patch.object(demonstrations,'fit_command') as fit:
                demonstrations.teach_command(args)
                fit.assert_called_once_with(args)
            self.assertTrue(list(Path(directory).glob('human_*.npz')))
            self.assertFalse(pdi.events)

    def test_exact_labels_preserve_complete_combos_and_reject_rounding(self):
        for i in range(len(ACTIONS)):self.assertEqual(exact_action(action_bits(i)),i)
        keys=action_bits(ACTION_INDEX['collect_forward']);keys[7]=1
        self.assertEqual(exact_action(keys),-1)
        self.assertEqual(exact_action(action_bits(ACTION_INDEX['camera_swap'])),ACTION_INDEX['camera_swap'])

    def test_reaction_alignment_never_crosses_a_pause_and_saves_raw_unsupported_keys(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            match=HumanMatch(directory,RewardConfig(),reaction_ticks=2)
            obs=observation()
            match.add(obs,action_bits(1),1)
            match.add(obs,action_bits(1),1.1)
            self.assertFalse(match.actions)
            match.add(obs,action_bits(1),3)
            match.add(obs,action_bits(1),3.1)
            self.assertFalse(match.actions)
            bad=action_bits(1);bad[7]=1
            match.add(obs,bad,3.2);match.add(obs,bad,3.3)
            self.assertEqual(match.actions,[-1])
            path=match.save()
            with np.load(path,allow_pickle=False) as data:
                np.testing.assert_array_equal(data['keys'][0],bad)
            self.assertEqual(load_matches(directory,RewardConfig()),[])

    def test_human_fit_preserves_value_net_and_survives_save_load_then_ppo(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            for _ in range(3):
                match=HumanMatch(directory,RewardConfig(),reaction_ticks=0)
                for tick in range(120):match.add(observation(),action_bits(1),tick/10)
                match.save()
            env=PracticeEnv(duration=2);model=make_model(env)
            before={n:t.clone() for n,t in model.policy.mlp_extractor.value_net.state_dict().items()}
            metrics=fit_human(model,directory,RewardConfig(),epochs=3)
            self.assertLess(metrics['best_loss'],metrics['before_loss'])
            self.assertEqual(len(metrics['heldout_matches']),1)
            for n,t in before.items():torch.testing.assert_close(t,model.policy.mlp_extractor.value_net.state_dict()[n])
            path=save_checkpoint(model,directory,domain='human_baseline')
            loaded=PPO.load(path,device='cpu',env=env)
            self.assertEqual(loaded.policy.guidance_fraction,.1)
            loaded.learn(128)
            self.assertGreater(loaded._n_updates,0)
            env.close()

    def test_h_action_is_a_tap_with_cooldown_and_uses_real_driver(self):
        from tests.test_agent import live_fixture
        index=ACTION_INDEX['camera_swap']
        self.assertTrue(tick_bits(index,0)[9]);self.assertFalse(tick_bits(index,1).any())
        controller=ObjectiveController()
        d=controller.update(Scene(),fill=0,keys=action_bits(index))
        self.assertFalse(d.allowed[index])
        with live_fixture() as (env,pdi):
            env.reset();pdi.events.clear();env.step(index)
            self.assertEqual(pdi.events.count(('down','h')),1)
            self.assertNotIn(('down','i'),pdi.events)
            env._view_parity=1;env._steps_since_swap=40
            env._update_objective()
            self.assertEqual(sum(env.decision.allowed),1)
            self.assertTrue(env.decision.allowed[index])

