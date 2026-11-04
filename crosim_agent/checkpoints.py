"""Versioned model artifacts and atomic publication of the latest pointer."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import uuid

from .actions import ACTIONS, ACTION_VERSION
from .features import FEATURE_NAMES, FEATURE_VERSION, HISTORY, OBS_DIM
from .proxy import ACTION_REPEAT, CONTROL_DT
from .rewards import REWARD_VERSION, RewardConfig
from .objectives import OBJECTIVE_VERSION
from mosim_zones import FUEL_ACCESS_VERSION
from .hud import HUD_SENSORS_VERSION
from mosim_field import FIELD_MAP_VERSION


class CheckpointSchemaMismatch(ValueError):
    pass


def schema(reward_config=None):
    config = reward_config or RewardConfig()
    return {"schema": 1, "feature_version": FEATURE_VERSION, "feature_names": list(FEATURE_NAMES),
            "policy_version": "guided-mixture-v3", "policy_hidden_layers": [64, 64],
            "objective_version": OBJECTIVE_VERSION,
            "history": HISTORY, "observation_dim": OBS_DIM,
            "action_version": ACTION_VERSION,
            "actions": [{"name": a.name, "keys": list(a.keys)} for a in ACTIONS],
            "action_repeat": ACTION_REPEAT, "control_dt": CONTROL_DT,
            "reward_version": REWARD_VERSION, "reward_config": asdict(config)}


def fingerprint(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_checkpoint(model, directory, *, domain, metrics=None, reward_config=None, promote=True):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "policy"):
        import torch
        if any(not torch.isfinite(parameter).all().item() for parameter in model.policy.parameters()):
            raise ValueError("Refusing to publish a checkpoint with non-finite model weights")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    name = f"{domain}_{stamp}"
    temporary = directory / ("pending_"+uuid.uuid4().hex+".zip")
    destination = directory / (name+".zip")
    try:
        model.save(str(temporary))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    spec = schema(reward_config)
    metadata = {"model": destination.name, "domain": domain,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "timesteps": int(model.num_timesteps), "spec": spec,
                "fuel_access_version": FUEL_ACCESS_VERSION,
                "hud_sensors_version": HUD_SENSORS_VERSION,
                "field_map_version":FIELD_MAP_VERSION,
                "weights_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "fingerprint": fingerprint(spec), "metrics": metrics or {}}
    write_json_atomic(destination.with_suffix(".json"), metadata)
    # Point to the model only after BOTH weights and metadata are durable.
    if promote:
        write_json_atomic(directory / "latest.json", metadata)
    return destination


def resolve_checkpoint(directory, explicit=None, reward_config=None):
    directory = Path(directory).resolve()
    if explicit:
        model_path = Path(explicit).expanduser().resolve()
        metadata_path = model_path.with_suffix(".json")
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
    else:
        pointer = directory / "latest.json"
        if not pointer.is_file():
            return None
        metadata = json.loads(pointer.read_text(encoding="utf-8"))
        model_path = (directory / metadata["model"]).resolve()
        if not model_path.is_relative_to(directory):
            raise ValueError("Checkpoint pointer escapes the model directory")
        metadata_path = model_path.with_suffix(".json")
    if not metadata_path.is_file() or not model_path.is_file():
        raise ValueError("Checkpoint weights/metadata pair is incomplete; use a known complete checkpoint")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_spec = schema(reward_config)
    expected = fingerprint(expected_spec)
    recorded = metadata.get("spec",{})
    if metadata.get("fingerprint") != fingerprint(recorded):
        raise ValueError("Checkpoint schema metadata failed its integrity check")
    # Known reward/controller upgrades preserve tensor shapes and optimizer.
    # Validate the entire old config, including its original scoring weight.
    shift_spec = json.loads(json.dumps(expected_spec))
    shift_spec["reward_version"] = "shift-position-reward-v5"
    for name in ("blue_bump_per_second","blue_bump_grace_seconds","red_structure_per_second"):
        shift_spec["reward_config"].pop(name)
    shift_spec["reward_config"].update(fuel_potential=1.5,fuel_target_potential=.35,spin_per_second=.15)
    boundary_spec = json.loads(json.dumps(shift_spec))
    boundary_spec["reward_version"] = "field-boundary-reward-v4"
    for name in ("shift_zone_per_second","shift_zone_grace_seconds"):
        boundary_spec["reward_config"].pop(name)
    boundary_spec["reward_config"]["score_per_point"] = expected_spec["reward_config"]["score_per_point"]/1.8
    previous_spec = json.loads(json.dumps(boundary_spec))
    previous_spec["reward_version"] = "independent-sensor-reward-v3"
    for name in ("red_zone_entry_cost","red_zone_per_second","zone_confirm_seconds"):
        previous_spec["reward_config"].pop(name)
    compatible_specs = []
    for objective in (expected_spec["objective_version"],"advisory-policy-v4","guided-exploration-v3"):
        for policy in (expected_spec["policy_version"],"objective-masked-v2"):
            for candidate in (expected_spec,shift_spec,boundary_spec,previous_spec):
                candidate = json.loads(json.dumps(candidate))
                candidate["objective_version"] = objective
                candidate["policy_version"] = policy
                compatible_specs.append(candidate)
    # A single new H action adds one mask and one previous-action feature.
    # Old network columns/rows are expanded by name during load_model().
    for candidate in list(compatible_specs):
        legacy = json.loads(json.dumps(candidate))
        legacy['actions'] = legacy['actions'][:-1]
        legacy['action_version'] = 'objective-actions-v2-camera-H'
        legacy['feature_names'] = [n for n in legacy['feature_names'] if n not in ('allow_camera_swap','previous_camera_swap')]
        legacy['observation_dim'] = len(legacy['feature_names'])*legacy['history']
        compatible_specs.append(legacy)
    compatible_extension = recorded in compatible_specs and fingerprint(recorded) != expected
    if fingerprint(recorded) != expected and not compatible_extension:
        raise CheckpointSchemaMismatch("Checkpoint uses a different feature/action/reward schema. "
                         "Use --fresh or choose a compatible feature-agent checkpoint; v4 pixel models are separate.")
    if metadata.get("model") != model_path.name:
        raise ValueError("Checkpoint metadata names a different model file")
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata.get("weights_sha256"):
        raise ValueError("Checkpoint contents failed their integrity check; choose another complete checkpoint")
    if compatible_extension:
        metadata = {**metadata,"compatible_reward_extension":True}
    return model_path, metadata
