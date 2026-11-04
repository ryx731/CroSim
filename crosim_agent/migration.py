"""Expand the former 17-action policy by named input columns, preserving weights."""
import json
from pathlib import Path
import torch
from .actions import ACTIONS
from .features import FEATURE_NAMES
from .policy import DELTA_NAMES


def expand_camera_model(path,data,params,env=None):
    from .training import make_model
    from .proxy import PracticeEnv
    from .rewards import RewardConfig
    meta=json.loads(Path(path).with_suffix('.json').read_text())
    old_names=meta['spec']['feature_names']
    if meta['spec']['actions'] != [{'name':a.name,'keys':list(a.keys)} for a in ACTIONS[:-1]]:
        raise ValueError('Unsupported action migration')
    model=make_model(env or PracticeEnv(),RewardConfig())
    old_core=old_names+['delta_'+n for n in DELTA_NAMES]
    new_core=list(FEATURE_NAMES)+['delta_'+n for n in DELTA_NAMES]
    mapping=[new_core.index(n) for n in old_core]
    def expand(name,old,new):
        if old.shape == new.shape:return old.clone()
        out=torch.zeros_like(new)
        if name in ('mlp_extractor.policy_net.0.weight','mlp_extractor.value_net.0.weight'):
            out[:,mapping]=old
        elif name in ('action_net.weight','action_net.bias'):
            out[:len(ACTIONS)-1]=old
        else:raise ValueError('Unsupported migrated tensor: '+name)
        return out
    state=model.policy.state_dict()
    for name,old in params['policy'].items():
        if name.endswith(('delta_indices','mask_indices')):continue
        state[name]=expand(name,old,state[name])
    state['action_net.bias'][-1]=-1.5
    model.policy.load_state_dict(state)
    # Optimizer IDs have the same parameter order; pad moments exactly as the
    # corresponding learned tensors, keeping the new camera moments at zero.
    old_optimizer=params.get('policy.optimizer')
    if old_optimizer:
        new_optimizer=model.policy.optimizer.state_dict()
        names=[n for n,_ in model.policy.named_parameters()]
        ids=new_optimizer['param_groups'][0]['params']
        old_ids=old_optimizer['param_groups'][0]['params']
        for name,new_id,old_id in zip(names,ids,old_ids):
            if old_id not in old_optimizer['state']:continue
            target=dict(model.policy.named_parameters())[name]
            new_optimizer['state'][new_id]={k:(v if not torch.is_tensor(v) else v.clone() if v.ndim==0 else expand(name,v,target)) for k,v in old_optimizer['state'][old_id].items()}
        model.policy.optimizer.load_state_dict(new_optimizer)
    for name in ('num_timesteps','_n_updates','_episode_num'):
        if name in data:setattr(model,name,data[name])
    return model
