"""An observable-feature heuristic for offline bootstrap, not an expert promise."""
import numpy as np

from .actions import ACTION_INDEX
from .features import FEATURE_INDEX, FRAME_DIM, HISTORY, feature_value
from .objectives import GUIDANCE_SCALE


def teacher_action(observation):
    def get(name):
        value = feature_value(observation,name)
        return value/GUIDANCE_SCALE if name in {"target_visible","target_x","target_y","target_scale"} else value
    allowed = [name for name in ACTION_INDEX if get("allow_"+name) > 0.5]
    def choose(name):
        if name not in allowed:
            name = next((fallback for fallback in ("collect_forward","approach_hub","retreat","return_left","coast") if fallback in allowed),"coast")
        return ACTION_INDEX[name] if allowed else 0
    if get("goal_wait"):
        return choose("coast")
    if get("goal_shoot"):
        return choose("shoot")
    if get("goal_recover"):
        return choose("retreat")
    if get("intake_dry") >= 0.5:
        return choose("toggle_hopper")
    if not get("target_visible"):
        return choose("aim_left")
    x, y = get("target_x"), get("target_y")
    if get("goal_return"):
        return choose(("aim_left" if x < 0 else "aim_right") if abs(x) > 0.14 else "approach_hub")
    if abs(x) > 0.45:
        return choose("search_left" if x < 0 else "search_right")
    if abs(x) > 0.16 and y > 0.65:
        return choose("collect_left" if x < 0 else "collect_right")
    if abs(x) > 0.16:
        return choose("collect_forward_left" if x < 0 else "collect_forward_right")
    return choose("collect_forward")


def collect_teacher_data(env, steps=6000, seed=123):
    observations, labels = [], []
    observation, _ = env.reset(seed=seed)
    for index in range(int(steps)):
        action = teacher_action(observation)
        observations.append(observation.copy())
        labels.append(action)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            observation, _ = env.reset(seed=seed+index+1)
    return np.asarray(observations, np.float32), np.asarray(labels, np.int64)


def aggregate_learner_data(model, env, steps=1500, seed=123, teacher_fraction=0.0):
    rng = np.random.default_rng(seed)
    observations, labels = [], []
    observation, _ = env.reset(seed=seed)
    for index in range(int(steps)):
        expert = teacher_action(observation)
        observations.append(observation.copy())
        labels.append(expert)
        proposed, _ = model.predict(observation, deterministic=True)
        executed = expert if rng.random() < teacher_fraction else int(proposed)
        observation, _, terminal, truncated, _ = env.step(executed)
        if terminal or truncated:
            observation, _ = env.reset(seed=seed+index+1)
    return np.asarray(observations, np.float32), np.asarray(labels, np.int64)


def clone_policy(model, observations, actions, *, epochs=12, batch_size=128, seed=123):
    import torch
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    counts = np.bincount(actions, minlength=model.action_space.n)
    # Capped balancing: no infinite weights for actions the teacher never chose.
    weights = np.sqrt(max(int(counts.max()), 1)/np.maximum(counts, 1))
    weights = torch.as_tensor(np.minimum(weights, 4), dtype=torch.float32, device=model.device)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    model.policy.set_training_mode(True)
    losses = []
    previous_start = len(FEATURE_INDEX)-model.action_space.n
    for _ in range(int(epochs)):
        order = rng.permutation(len(actions))
        for offset in range(0, len(actions), batch_size):
            indices = order[offset:offset+batch_size]
            batch = observations[indices].copy().reshape(-1, HISTORY, FRAME_DIM)
            drop = rng.random(len(indices)) < 0.4
            batch[drop, :, previous_start:] = 0
            tensor = torch.as_tensor(batch.reshape(len(indices), -1), device=model.device)
            target = torch.as_tensor(actions[indices], device=model.device)
            distribution = model.policy.get_distribution(tensor).distribution
            loss = torch.nn.functional.cross_entropy(distribution.logits, target, weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach()))
    model.policy.set_training_mode(False)
    return {"samples": int(len(actions)), "epochs": int(epochs),
            "mean_loss": float(np.mean(losses[-max(1, len(actions)//batch_size):])),
            "action_counts": counts.tolist()}


VALIDATION_SEEDS = tuple(range(9001, 9013))


def evaluate(model, env, seeds=VALIDATION_SEEDS):
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ValueError("Evaluation requires at least one episode seed")
    scores, returns, collected, decisions = [], [], [], []
    for seed in seeds:
        observation, _ = env.reset(seed=seed)
        total = 0.0
        count = 0
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminal, truncated, info = env.step(action)
            total += reward
            count += 1
            if terminal or truncated:
                scores.append(info["points"])
                returns.append(total)
                collected.append(info["collected"])
                decisions.append(count)
                break
    return {"domain": "practice_proxy", "episodes": len(scores), "seeds": list(seeds),
            "mean_score": float(np.mean(scores)), "mean_collected": float(np.mean(collected)),
            "mean_return": float(np.mean(returns)), "median_score": float(np.median(scores)),
            "scoring_episodes": sum(score > 0 for score in scores),
            "scores": scores, "collected": collected, "decisions": decisions,
            "real_game_skill_validated": False}
