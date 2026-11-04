"""Human feature/action trajectories -> categorical actor initialization -> PPO.

No offline PPO on human actions: their behavior probabilities are unknown.
Unsupported joint controls are retained as raw keys, never silently rounded.
"""
from collections import deque
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import uuid

import numpy as np

from .actions import ACTIONS
from .features import OBS_DIM,HISTORY,FRAME_DIM,FEATURE_INDEX
from .checkpoints import schema,fingerprint,write_json_atomic


DEMO_VERSION = 1


def exact_action(keys):
    keys = np.asarray(keys)
    if keys.shape != (10,) or not np.isin(keys,(0,1)).all():
        raise ValueError('Expected ten binary key states')
    return next((i for i,a in enumerate(ACTIONS) if np.array_equal(keys,a.bits())),-1)


class HumanMatch:
    def __init__(self,directory,config,reaction_ticks=2):
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.path=self.directory/('human_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')+'.npz')
        self.spec=schema(config)
        self.delay=int(reaction_ticks)
        if not 0 <= self.delay <= 5:raise ValueError('Reaction delay must be 0..5 ticks')
        self.history=deque(maxlen=self.delay+1)
        self.observations=[];self.actions=[];self.keys=[];self.times=[]
        self.tick=0;self.last_at=None;self.points=0;self.complete=False

    def add(self,observation,keys,at):
        observation=np.asarray(observation,np.float32)
        if observation.shape != (OBS_DIM,) or not np.isfinite(observation).all():
            raise ValueError('Invalid demonstration observation')
        if self.last_at is not None and (at-self.last_at > .5 or at <= self.last_at):
            self.history.clear()
        self.last_at=at
        self.history.append(observation.copy())
        self.tick+=1
        if len(self.history) <= self.delay or self.tick%2:return
        self.observations.append(self.history[0].copy())
        self.actions.append(exact_action(keys))
        self.keys.append(np.asarray(keys,np.int8).copy())
        self.times.append(float(at))

    def save(self):
        if not self.observations:return None
        metadata={'version':DEMO_VERSION,'spec':self.spec,'reaction_ticks':self.delay,
                  'points':self.points,'complete':self.complete,'samples':len(self.actions),
                  'unsupported':sum(a < 0 for a in self.actions)}
        temporary=self.path.with_name(self.path.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            with temporary.open('wb') as f:
                np.savez_compressed(f,observations=np.asarray(self.observations,np.float32),
                                    actions=np.asarray(self.actions,np.int64),keys=np.asarray(self.keys,np.int8),
                                    times=np.asarray(self.times),metadata=json.dumps(metadata))
            temporary.replace(self.path)
        finally:temporary.unlink(missing_ok=True)
        return self.path


def load_matches(directory,config):
    matches=[]
    expected=schema(config)
    # Reward changes do not invalidate recorded images/features or human labels.
    structural=('feature_version','feature_names','history','observation_dim','action_version','actions')
    for path in sorted(Path(directory).glob('human_*.npz')):
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data['metadata']))
            if meta.get('version') != DEMO_VERSION or any(meta['spec'].get(k) != expected[k] for k in structural):
                raise ValueError(f'Incompatible human recording: {path.name}')
            obs=np.asarray(data['observations'],np.float32)
            actions=np.asarray(data['actions'],np.int64)
        if obs.shape != (len(actions),OBS_DIM) or not np.isfinite(obs).all() or np.any((actions < -1)|(actions >= len(ACTIONS))):
            raise ValueError(f'Invalid human recording: {path.name}')
        valid=actions >= 0
        safe=np.clip(actions,0,len(ACTIONS)-1)
        current=obs.reshape(-1,HISTORY,FRAME_DIM)[:,-1,:]
        masks=current[:,[FEATURE_INDEX['allow_'+a.name] for a in ACTIONS]] > .5
        valid &= masks[np.arange(len(actions)),safe]
        if valid.any():matches.append((path,obs[valid],actions[valid],len(actions)-int(valid.sum())))
    return matches


def fit_human(model,directory,config,*,epochs=10,seed=123):
    """Fit complete categorical controls, retaining the best held-out actor.

    Split by match to avoid neighboring frames leaking into validation. Train
    the raw learned actor, not the fixed guidance mixture. Value weights stay
    intact; actor optimizer moments are cleared after this separate update.
    """
    import copy
    import torch
    matches=load_matches(directory,config)
    if not matches:raise ValueError('No usable human matches. Run teach_agent.cmd first.')
    if sum(len(m[2]) for m in matches) < 32:raise ValueError('Need at least 32 usable decisions (roughly 7 seconds).')
    if epochs < 1:raise ValueError('Epochs must be positive')
    rng=np.random.default_rng(seed)
    # Whole-match validation once two or more matches exist.
    order=rng.permutation(len(matches));held=set(order[:max(1,len(matches)//5)]) if len(matches)>1 else set()
    train=[m for i,m in enumerate(matches) if i not in held]
    validation=[m for i,m in enumerate(matches) if i in held]
    x=np.concatenate([m[1] for m in train]);y=np.concatenate([m[2] for m in train])
    vx=np.concatenate([m[1] for m in validation]) if validation else x
    vy=np.concatenate([m[2] for m in validation]) if validation else y
    actor=list(model.policy.mlp_extractor.policy_net.parameters())+list(model.policy.action_net.parameters())
    for parameter in actor:parameter.requires_grad_(True)
    optimizer=torch.optim.Adam(actor,lr=2e-4)
    counts=np.bincount(y,minlength=len(ACTIONS))
    weights=torch.as_tensor(np.minimum(3,np.sqrt(max(1,counts.max())/np.maximum(1,counts))),dtype=torch.float32,device=model.device)
    def logits(batch):
        latent=model.policy.mlp_extractor.forward_actor(model.policy.extract_features(batch))
        return model.policy.action_net(latent[:,:64])
    def measure():
        losses=[];correct=0
        with torch.no_grad():
            for offset in range(0,len(vy),256):
                output=logits(torch.as_tensor(vx[offset:offset+256],device=model.device))
                labels=torch.as_tensor(vy[offset:offset+256],device=model.device)
                losses.append(float(torch.nn.functional.cross_entropy(output,labels,reduction='sum')))
                correct+=int((output.argmax(1)==labels).sum())
        return sum(losses)/len(vy),correct/len(vy)
    before,_=measure();best=before;best_state=copy.deepcopy(model.policy.state_dict());best_epoch=0
    model.policy.set_training_mode(True)
    try:
        for epoch in range(epochs):
            indices=rng.permutation(len(y))
            for start in range(0,len(y),128):
                ix=indices[start:start+128]
                batch=torch.as_tensor(x[ix],device=model.device)
                labels=torch.as_tensor(y[ix],device=model.device)
                loss=torch.nn.functional.cross_entropy(logits(batch),labels,weight=weights)
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(actor,.5);optimizer.step()
            score,accuracy=measure()
            print(f'[human fit] {epoch+1}/{epochs}: {"held-out match" if validation else "training"} loss {score:.3f}; action agreement {accuracy:.0%}',flush=True)
            if score < best:
                best=score;best_epoch=epoch+1;best_state=copy.deepcopy(model.policy.state_dict())
        model.policy.load_state_dict(best_state)
    except BaseException:
        model.policy.load_state_dict(best_state)
        raise
    finally:model.policy.set_training_mode(False)
    if best_epoch == 0:raise ValueError('Human fit did not improve action prediction; current checkpoint kept.')
    for parameter in actor:model.policy.optimizer.state.pop(parameter,None)
    model.policy.guidance_fraction=.1
    model.policy_kwargs['guidance_fraction']=.1
    return {'source':'human_demonstrations','matches':len(matches),'training_samples':len(y),
            'validation_samples':len(vy) if validation else 0,'heldout_matches':[m[0].name for m in validation],
            'before_loss':before,'best_loss':best,'best_epoch':best_epoch,
            'unsupported_or_masked':sum(m[3] for m in matches),'action_counts':counts.tolist(),
            'guidance_fraction':.1,'real_game_skill_validated':False}


def fit_command(args):
    from .cli import HERE,reward_settings,load_or_bootstrap,require_ready
    from .session import TrainingLease
    from .checkpoints import save_checkpoint
    require_ready(False)
    with TrainingLease(HERE):
        config=reward_settings(args)
        # Validate recordings before bootstrapping or altering any model.
        matches=load_matches(args.demos,config)
        if sum(len(m[2]) for m in matches) < 32:raise ValueError('Need at least 32 usable human decisions; record a longer match first.')
        directory=Path(args.models)/'live'
        model,_=load_or_bootstrap(args,directory,config)
        try:
            metrics=fit_human(model,args.demos,config,epochs=args.epochs,seed=args.seed)
            path=save_checkpoint(model,directory,domain='human_baseline',metrics=metrics,reward_config=config)
            write_json_atomic(directory/'human_fit.json',metrics)
            print(f"[human fit] {metrics['training_samples']} training samples; {metrics['unsupported_or_masked']} unsupported/restricted samples excluded. Holdout: {metrics['validation_samples']}.",flush=True)
            print(f'[human fit] Saved {path.name}. Run train_agent.cmd when ready for autonomous PPO.',flush=True)
        finally:
            if model.get_env() is not None:model.get_env().close()


def teach_command(args):
    from .cli import require_ready,debug_settings,reward_settings,HERE
    from .live import LiveFeatureEnv
    from .timer import MatchClock
    from .session import TrainingLease
    from mosim_env import read_keys_now
    config=reward_settings(args);require_ready(True)
    saved=[];match=None
    with TrainingLease(HERE):
        env=LiveFeatureEnv(debug_options=debug_settings(args),reward_config=config,inject_inputs=False)
        print('[teach] YOU control MoSim. Start a match with R, then press ]. [ saves and fits. No game keys are injected.',flush=True)
        try:
            observation,_=env.reset()
            match=HumanMatch(args.demos,config,args.reaction_ticks)
            previous_reset=False
            while True:
                env._wait_ready()
                reset=bool(env._guard.user.GetAsyncKeyState(0x52)&0x8000)
                if reset and not previous_reset:
                    match.points=env.episode_points
                    path=match.save()
                    if path:saved.append(path)
                    env._live_start=None;env.match_clock=MatchClock()
                    observation,_=env.reset();match=HumanMatch(args.demos,config,args.reaction_ticks)
                previous_reset=reset
                keys=read_keys_now()
                match.add(observation,keys,time.perf_counter())
                env._human_demo_samples=len(match.actions)
                observation,_,over,_,_=env.watch_step(keys)
                match.points=env.episode_points
                if len(match.actions) and len(match.actions)%128 == 0:match.save()
                if over:
                    match.complete=True
                    path=match.save()
                    if path:saved.append(path)
                    match=None
                    print('[teach] Match saved. Press R for a new match, then ] to continue.',flush=True)
                    with env._guard.input_lock:env._guard._pause('HUMAN MATCH COMPLETE: R then ]')
                    env._wait_ready()
                    env._live_start=None;env.match_clock=MatchClock()
                    observation,_=env.reset();match=HumanMatch(args.demos,config,args.reaction_ticks)
        except KeyboardInterrupt:pass
        finally:
            if match is not None:
                path=match.save()
                if path:saved.append(path)
            env.close()
    print(f'[teach] Saved {len(saved)} match/partial recording(s).',flush=True)
    if saved:
        fit_command(args)
