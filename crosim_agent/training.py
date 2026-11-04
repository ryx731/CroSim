"""CPU MLP training and automatic proxy-teacher bootstrap."""
from pathlib import Path
import time

import numpy as np

from .checkpoints import save_checkpoint, write_json_atomic
from .proxy import PracticeEnv, ACTION_REPEAT
from .rewards import RewardConfig
from .teacher import collect_teacher_data, clone_policy, evaluate, aggregate_learner_data


def load_model(path, env=None):
    import torch
    from stable_baselines3 import PPO
    # Tiny matrix multiplies are slower with a large thread pool, especially
    # between live rollouts. Apply this to resumed models as well as new ones.
    torch.set_num_threads(1)
    from stable_baselines3.common.save_util import load_from_zip_file
    from .actions import ACTIONS
    data,params,_=load_from_zip_file(path,device='cpu')
    if data['action_space'].n == len(ACTIONS)-1:
        from .migration import expand_camera_model
        return expand_camera_model(path,data,params,env)
    return PPO.load(str(path), device="cpu", env=env)


def make_model(env, reward_config=None, seed=123):
    import torch
    from stable_baselines3 import PPO
    from .policy import CurrentWithDeltas, ObjectivePolicy
    torch.set_num_threads(1)
    config = reward_config or RewardConfig()
    return PPO(ObjectivePolicy, env, device="cpu", seed=seed,
               learning_rate=2.5e-4, n_steps=128, batch_size=64, n_epochs=4,
               gamma=config.gamma**ACTION_REPEAT, gae_lambda=0.95,
               ent_coef=0.02, clip_range=0.15, target_kl=0.03,
               policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]},
                              "features_extractor_class": CurrentWithDeltas},
               verbose=0)


def bootstrap(directory, reward_config=None, *, samples=6000, epochs=12, seed=123):
    config = reward_config or RewardConfig()
    env = PracticeEnv(reward_config=config,sensor_dropout=True)
    model = make_model(env, config, seed)
    print(f"[bootstrap] Generating {samples} practice examples; no game capture or input.", flush=True)
    try:
        observations, labels = collect_teacher_data(env, samples, seed)
        metrics = clone_policy(model, observations, labels, epochs=epochs, seed=seed)
        # Gather corrections on the model's own state distribution. The teacher
        # is the local heuristic, so this requires no human demonstrations.
        metrics["aggregation"] = []
        for round_index, mixture in enumerate((0.6, 0.25, 0.0)):
            extra_obs, extra_labels = aggregate_learner_data(model, env, steps=max(800, samples//3),
                                                             seed=seed+round_index*1000+30000,
                                                             teacher_fraction=mixture)
            observations = np.concatenate((observations, extra_obs))
            labels = np.concatenate((labels, extra_labels))
            result = clone_policy(model, observations, labels, epochs=max(4, epochs//2), seed=seed+round_index+1)
            metrics["aggregation"].append(result)
            print(f"[bootstrap] Automatic correction round {round_index+1}/3: {len(labels)} examples", flush=True)
        # Retain exploration after cloning rather than starting PPO with an
        # almost deterministic steering habit. Argmax behavior is unchanged.
        import torch
        with torch.no_grad():
            model.policy.action_net.weight.mul_(.65)
            model.policy.action_net.bias.mul_(.65)
        metrics["exploration_logit_scale"] = .65
        validation_env = PracticeEnv(reward_config=config)
        try:
            validation_obs, validation_actions = collect_teacher_data(validation_env, 600, seed+10000)
            predictions, _ = model.predict(validation_obs, deterministic=True)
            metrics["heldout_action_accuracy"] = float(np.mean(predictions == validation_actions))
            metrics["evaluation"] = evaluate(model, validation_env)
        finally:
            validation_env.close()
        metrics["description"] = "Synthetic practice initialization; not evidence of real MoSimulator skill"
        path = save_checkpoint(model, directory, domain="bootstrap_proxy", metrics=metrics,
                               reward_config=config)
        print(f"[bootstrap] Saved {path.name}; held-out action agreement "
              f"{metrics['heldout_action_accuracy']:.1%}. Practice score "
              f"{metrics['evaluation']['mean_score']:.1f}; live skill unmeasured.", flush=True)
        return model, path, metrics
    except BaseException:
        model.get_env().close()
        raise


def training_callback(directory, *, live=False, freeze_actor_steps=0, reward_config=None):
    from stable_baselines3.common.callbacks import BaseCallback

    class TrainingObserver(BaseCallback):
        def __init__(self):
            super().__init__()
            self.started_steps = 0
            self.last_status = 0.0
            self.last_checkpoint = 0
            self.episodes = 0
            self.latest_episode = {}
            self.actor_frozen = None

        def _on_training_start(self):
            self.started_steps = self.model.num_timesteps
            self.last_checkpoint = self.model.num_timesteps

        def _on_rollout_start(self):
            freeze = self.model.num_timesteps-self.started_steps < freeze_actor_steps
            if freeze != self.actor_frozen:
                for module in (self.model.policy.mlp_extractor.policy_net, self.model.policy.action_net):
                    module.requires_grad_(not freeze)
                self.actor_frozen = freeze
            if live and self.model.num_timesteps-self.last_checkpoint >= 512:
                self.training_env.env_method("release_all_keys")
                save_checkpoint(self.model,directory,domain="live_screen",
                                metrics=self.latest_episode,reward_config=reward_config)
                self.last_checkpoint = self.model.num_timesteps

        def _on_step(self):
            for done, info in zip(self.locals.get("dones", []), self.locals.get("infos", [])):
                if done:
                    self.episodes += 1
                    self.latest_episode = {"episode": self.episodes,
                                           "credited_points": info.get("credited_points"),
                                           "uncredited_points": info.get("uncredited_points"),
                                           "points": info.get("alliance_points_total", info.get("points")),
                                           "interrupted": info.get("interrupted", False),
                                           "auto_restart":info.get("auto_restart",False),
                                           "episode_stats": info.get("episode", {})}
                    if live or self.episodes % 10 == 0:
                        print(f"[episode {self.episodes}] {self.latest_episode}", flush=True)
            now = time.perf_counter()
            if now-self.last_status >= 2:
                self.last_status = now
                write_json_atomic(Path(directory)/"status.json", {
                    "phase": "training", "domain": "live_screen" if live else "practice_proxy",
                    "steps": self.model.num_timesteps, "episodes": self.episodes,
                    "latest_episode": self.latest_episode,
                    "actor_value_warmup": bool(self.actor_frozen)})
            return True

        def _on_rollout_end(self):
            if live:
                self.training_env.env_method("release_all_keys")

        def _on_training_end(self):
            # Do not persist a frozen policy into subsequent sessions.
            for module in (self.model.policy.mlp_extractor.policy_net, self.model.policy.action_net):
                module.requires_grad_(True)

    return TrainingObserver()
