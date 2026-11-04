"""Emphasize the current state; expose history as explicit change signals."""
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

from .features import FRAME_DIM, HISTORY, FEATURE_INDEX
from .actions import ACTIONS, ACTION_INDEX
from .objectives import GUIDANCE_SCALE


EXPLORATION_FRACTION = .05


def guidance_actions(current):
    """Weak action prior from public sensor/goal inputs; never forced controls.

    Keep hopper deployment out of this prior: intake-dry is not evidence that
    another toggle will help. The learned policy can still try it on cooldown.
    """
    def get(name):
        return current[:, FEATURE_INDEX[name]]
    def choose(condition, yes, no):
        return torch.where(condition, torch.full_like(no,ACTION_INDEX[yes]),no)
    x = get("target_x") / GUIDANCE_SCALE
    y = get("target_y") / GUIDANCE_SCALE
    collect = torch.full_like(x,ACTION_INDEX["collect_forward"],dtype=torch.long)
    diagonal = choose(x < 0,"collect_forward_left",torch.full_like(collect,ACTION_INDEX["collect_forward_right"]))
    sideways = choose(x < 0,"collect_left",torch.full_like(collect,ACTION_INDEX["collect_right"]))
    collect = torch.where(x.abs() > .16,torch.where(y > .65,sideways,diagonal),collect)
    search = choose(x < 0,"search_left",torch.full_like(collect,ACTION_INDEX["search_right"]))
    collect = torch.where(x.abs() > .45,search,collect)
    aim = choose(x < 0,"aim_left",torch.full_like(collect,ACTION_INDEX["aim_right"]))
    returning = choose(x.abs() <= .14,"approach_hub",aim)
    result = torch.where(get("goal_return") > 0,returning,collect)
    missing_target = choose(get("turn_budget") <= 0,"retreat",torch.full_like(collect,ACTION_INDEX["aim_left"]))
    result = torch.where(get("target_visible") <= 0,missing_target,result)
    result = choose(get("goal_recover") > 0,"retreat",result)
    result = choose(get("goal_shoot") > 0,"shoot",result)
    return choose(get("goal_wait") > 0,"coast",result)


DELTA_NAMES = ("fuel_x", "fuel_y", "hub_x", "hub_scale", "fill", "motion")


class CurrentWithDeltas(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=FRAME_DIM+len(DELTA_NAMES)+len(ACTIONS))
        self.register_buffer("delta_indices", torch.tensor([FEATURE_INDEX[n] for n in DELTA_NAMES], dtype=torch.long))
        self.register_buffer("mask_indices", torch.tensor([FEATURE_INDEX["allow_"+a.name] for a in ACTIONS], dtype=torch.long))

    def forward(self, observations):
        frames = observations.reshape(-1, HISTORY, FRAME_DIM)
        current = frames[:, -1, :]
        delta = (current-frames[:, 0, :]).index_select(1, self.delta_indices)
        masks = current.index_select(1, self.mask_indices)
        return torch.cat((current, delta, masks), dim=1)


class ObjectiveMlpExtractor(torch.nn.Module):
    """Carry the observation's legal-action bits through the actor unchanged."""
    latent_dim_pi = latent_dim_vf = 64

    def __init__(self, features_dim):
        super().__init__()
        core = features_dim-len(ACTIONS)
        def network():
            return torch.nn.Sequential(torch.nn.Linear(core, 64), torch.nn.Tanh(),
                                       torch.nn.Linear(64, 64), torch.nn.Tanh())
        self.policy_net, self.value_net = network(), network()

    def forward_actor(self, features):
        advice = torch.nn.functional.one_hot(guidance_actions(features[:,:FRAME_DIM]),len(ACTIONS))
        return torch.cat((self.policy_net(features[:, :-len(ACTIONS)]),
                          features[:, -len(ACTIONS):],advice.to(features.dtype)), dim=1)

    def forward_critic(self, features):
        return self.value_net(features[:, :-len(ACTIONS)])

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)


class ObjectivePolicy(ActorCriticPolicy):
    """Use one guided distribution for sampling and PPO log-prob evaluation.

    No after-the-fact substitution of a sampled action: the action actually
    executed has the same distribution/log probability PPO learns from.
    """
    def __init__(self,*args,guidance_fraction=GUIDANCE_SCALE,**kwargs):
        if not 0 <= guidance_fraction <= 1-EXPLORATION_FRACTION:
            raise ValueError('Invalid guidance fraction')
        self.guidance_fraction=float(guidance_fraction)
        super().__init__(*args,**kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = ObjectiveMlpExtractor(self.features_dim).to(self.device)

    def _get_action_dist_from_latent(self, latent_pi):
        logits = self.action_net(latent_pi[:, :64])
        mask = latent_pi[:, 64:64+len(ACTIONS)] > 0.5
        # A malformed/uninitialized observation can only coast, never select
        # arbitrary game keys or create NaNs from an all-masked categorical.
        mask[:, 0] |= ~mask.any(dim=1)
        learned = torch.softmax(logits.masked_fill(~mask,-1e9),dim=1)
        uniform = mask.to(logits.dtype)/mask.sum(dim=1,keepdim=True)
        advice = latent_pi[:,64+len(ACTIONS):] * mask
        # A forbidden recommendation falls back to legal exploration. No fuel
        # access restriction can be bypassed by the guidance or exploration.
        advice = torch.where(advice.sum(dim=1,keepdim=True) > 0,advice,uniform)
        probabilities = ((1-self.guidance_fraction-EXPLORATION_FRACTION)*learned
                         + self.guidance_fraction*advice + EXPLORATION_FRACTION*uniform)
        return self.action_dist.proba_distribution(
            action_logits=probabilities.clamp_min(1e-30).log().masked_fill(~mask,-1e9))
