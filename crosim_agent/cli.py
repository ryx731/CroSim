"""One-command feature-agent startup; offline commands never touch the game."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = HERE / "crosim_agent_models"


def dependency_errors(live=False):
    required = {"numpy": "numpy", "cv2": "opencv-python", "gymnasium": "gymnasium",
                "torch": "torch", "stable_baselines3": "stable-baselines3"}
    if live:
        required.update(bettercam="bettercam", pydirectinput="pydirectinput", pytesseract="pytesseract", tkinter="tkinter")
    errors = [f"Missing {module} ({package})" for module, package in required.items()
              if importlib.util.find_spec(module) is None]
    if live and sys.platform != "win32":
        errors.append("Live capture/input requires the interactive Windows desktop")
    return errors


def preflight(live=False, quiet=False):
    errors = dependency_errors(live)
    warnings = []
    if not errors:
        import cv2
        if not hasattr(cv2, "aruco"):
            warnings.append("AprilTag hub anchors unavailable in this OpenCV build")
        if live:
            from mosim_env import CONFIG_PATH, CFG, find_tesseract
            if not Path(CONFIG_PATH).is_file():
                errors.append("No local calibration. Run: python mosim_env.py setup")
            else:
                try:
                    data = json.loads(Path(CONFIG_PATH).read_text(encoding="utf-8-sig"))
                    if not isinstance(data, dict):
                        raise ValueError("configuration must be a JSON object")
                    for name in ("score", "hopper", "hub"):
                        box = CFG[f"{name}_region"]
                        if len(box) != 4 or not all(isinstance(v, (int, float)) for v in box) \
                                or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
                            errors.append(f"Invalid {name}_region; re-run setup")
                    if not str(CFG.game_window_title).strip():
                        errors.append("game_window_title is empty")
                except (ValueError, OSError, TypeError) as exc:
                    errors.append(f"Could not validate calibration: {exc}")
            if not find_tesseract():
                errors.append("Tesseract executable not found; configure its path with setup")
            if not CFG.hub_calibrated:
                warnings.append("Hub indicator uncalibrated; schedule is used and may need validation")
    if not quiet:
        print("[check] " + ("Ready" if not errors else "Not ready"))
        for message in errors:
            print("  ERROR: " + message)
        for message in warnings:
            print("  NOTE: " + message)
        if errors:
            print(f'  Python in use: {sys.executable}')
            print('  Install missing packages into this interpreter with -m pip install -r requirements.txt')
    return {"ready": not errors, "errors": errors, "warnings": warnings}


def reward_settings(args):
    from .rewards import RewardConfig
    values = {}
    if args.rewards:
        raw = json.loads(Path(args.rewards).read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict):
            raise ValueError("Reward settings must be a JSON object")
        values.update(raw)
    return RewardConfig(**values)


def debug_settings(args):
    return {"overlay": not args.no_overlay, "log_rewards": True,
            "log_dir": str(HERE/"mosim_debug_logs"), "vision_fps": 30,
            "corner": args.corner}


def require_ready(live):
    if not preflight(live)["ready"]:
        raise ValueError("Preflight failed before any game input was sent")


def load_or_bootstrap(args, directory, config, env=None):
    from .checkpoints import resolve_checkpoint, CheckpointSchemaMismatch
    from .training import bootstrap, load_model
    try:
        resolved = None if args.fresh else resolve_checkpoint(directory, args.model, config)
    except CheckpointSchemaMismatch:
        if args.model:
            raise
        print("[agent] Old policy schema: building the objective-based starter; previous files are preserved.", flush=True)
        resolved = None
    if resolved:
        path, metadata = resolved
        print(f"[agent] Resuming {path.name} ({metadata['domain']})", flush=True)
        if metadata.get("compatible_reward_extension"):
            print("[agent] Continuing learned weights with compatible controller/reward updates.",flush=True)
        model = load_model(path, env)
        return model, metadata
    model, path, metrics = bootstrap(directory, config, samples=args.bootstrap_samples,
                                      epochs=args.bootstrap_epochs, seed=args.seed)
    if env is not None:
        model.get_env().close()
        model.set_env(env)
    return model, {"model": path.name, "domain": "bootstrap_proxy", "metrics": metrics}


def train(args):
    from .session import TrainingLease
    with TrainingLease(HERE):
        return _train(args)


def training_steps(requested,live):
    if requested is None:
        requested = 0 if live else 10000
    if requested < 0:
        raise ValueError("--steps must be zero (continuous) or positive")
    return requested or 1_000_000_000


def _train(args):
    live = not args.offline
    steps = training_steps(args.steps,live)
    require_ready(live)
    config = reward_settings(args)
    from .checkpoints import save_checkpoint, write_json_atomic
    from .proxy import PracticeEnv
    from .training import training_callback
    from .teacher import evaluate
    directory = Path(args.models).resolve() / ("practice" if args.offline else "live")
    directory.mkdir(parents=True, exist_ok=True)
    env = model = None
    training_started = False
    starting_steps = starting_updates = 0
    status = "complete"
    before_evaluation = None
    try:
        # Bootstrap before native capture so launch can finish basic learning
        # without a game window, global hotkeys or GPU configuration.
        model, metadata = load_or_bootstrap(args, directory, config)
        if live:
            from .live import LiveFeatureEnv
            env = LiveFeatureEnv(debug_options=debug_settings(args), reward_config=config)
            warmup = 128 if metadata.get("domain") != "live_screen" else 0
            print("[agent] Ready. Focus MoSim, then press ]. [ stops and saves; ] pauses.", flush=True)
            if args.steps in (None,0):
                print("[agent] Continuous training: matches restart automatically; [ stops and saves.",flush=True)
        else:
            env = PracticeEnv(reward_config=config)
            warmup = 256 if metadata.get("domain") == "bootstrap_proxy" else 0
            before_evaluation = evaluate(model, env)
            print("[agent] Offline practice only: this does not measure real-game skill.", flush=True)
        old = model.get_env()
        if old:
            old.close()
        model.set_env(env)
        starting_steps = model.num_timesteps
        starting_updates = model._n_updates
        training_started = True
        model.learn(total_timesteps=steps, reset_num_timesteps=False,
                    callback=training_callback(directory, live=live, freeze_actor_steps=warmup,
                                               reward_config=config))
    except KeyboardInterrupt:
        status = "stopped"
        print("\n[agent] Stopped; saving current progress.", flush=True)
    except BaseException:
        status = "failed"
        raise
    finally:
        try:
            if env is not None and hasattr(env, "release_all_keys"):
                env.release_all_keys()
            if model is not None and training_started and model.num_timesteps > starting_steps:
                metrics = {"status": status, "session_decisions": model.num_timesteps-starting_steps,
                           "updates_performed": model._n_updates-starting_updates}
                if not live and status != "failed":
                    evaluation_env = PracticeEnv(reward_config=config)
                    try:
                        metrics["evaluation"] = evaluate(model, evaluation_env)
                        metrics["before_evaluation"] = before_evaluation
                    finally:
                        evaluation_env.close()
                promote = status != "failed"
                if not live and before_evaluation and metrics.get("evaluation"):
                    after = metrics["evaluation"]
                    promote = promote and ((after["mean_score"], after["mean_collected"]) >=
                                           (before_evaluation["mean_score"], before_evaluation["mean_collected"]))
                metrics["promoted"] = promote
                domain = ("live_screen" if model._n_updates > starting_updates else metadata.get("domain", "bootstrap_proxy")) \
                    if live else "practice_proxy"
                path = save_checkpoint(model, directory, domain=domain,
                                       metrics=metrics, reward_config=config, promote=promote)
                write_json_atomic(directory/"status.json", {"phase": status, "model": path.name,
                                                            "steps": model.num_timesteps, "metrics": metrics})
                print(f"[agent] Saved {path}", flush=True)
                if not promote:
                    reason = "Session failed" if status == "failed" else "Practice candidate regressed on validation"
                    print(f"[agent] {reason}; kept the previous latest pointer.", flush=True)
            elif model is not None:
                write_json_atomic(directory/"status.json", {"phase": status,
                                                            "training_started": False,
                                                            "checkpoint_unchanged": True})
        finally:
            if env is not None:
                env.close()
            elif model is not None and model.get_env() is not None:
                model.get_env().close()


def play(args):
    require_ready(True)
    config = reward_settings(args)
    from .training import load_model
    from .checkpoints import resolve_checkpoint
    from .live import LiveFeatureEnv
    resolved = resolve_checkpoint(Path(args.models)/"live", args.model, config)
    if resolved is None:
        raise ValueError("No feature-agent checkpoint. Run train_agent.cmd first; old pixel models use the legacy play command.")
    path, metadata = resolved
    model = load_model(path)
    env = LiveFeatureEnv(debug_options=debug_settings(args), reward_config=config, dry_run=args.dry_run)
    try:
        for _ in range(args.episodes):
            observation, _ = env.reset()
            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminal, truncated, info = env.step(action)
                if terminal or truncated:
                    print(f"[agent] Match ended: {info.get('credited_points', 0)} shoot-correlated points credited")
                    break
    except KeyboardInterrupt:
        print("\n[agent] Play stopped.")
    finally:
        env.close()


def bootstrap_command(args):
    from .session import TrainingLease
    with TrainingLease(HERE):
        return _bootstrap_command(args)


def _bootstrap_command(args):
    require_ready(False)
    config = reward_settings(args)
    directory = Path(args.models)/"live"
    model, metadata = load_or_bootstrap(args, directory, config)
    if model.get_env() is not None:
        model.get_env().close()
    from .checkpoints import write_json_atomic
    write_json_atomic(directory/"status.json", {"phase": "ready_for_live_training",
                                               "model": metadata.get("model"), "domain": metadata["domain"],
                                               "steps": model.num_timesteps,
                                               "metrics": metadata.get("metrics", {})})


def watch(args):
    require_ready(True)
    from mosim_env import read_keys_now
    from .live import LiveFeatureEnv
    env = LiveFeatureEnv(debug_options=debug_settings(args), reward_config=reward_settings(args), inject_inputs=False)
    print("[agent watch] Revised rewards, human controls. Restart with R, then press ]. [ stops.")
    try:
        env.reset()
        user = env._guard.user
        previous_reset = bool(user.GetAsyncKeyState(0x52) & 0x8000)
        while True:
            env._wait_ready()
            reset = bool(user.GetAsyncKeyState(0x52) & 0x8000)
            if reset and not previous_reset:
                env.reset()
            previous_reset = reset
            _, _, over, _, _ = env.watch_step(read_keys_now())
            if over:
                with env._guard.input_lock:
                    env._guard._pause("MATCH COMPLETE: restart with R, then press ]")
                env._wait_ready()
                env.reset()
    except KeyboardInterrupt:
        print("\n[agent watch] Stopped. Reward log saved.")
    finally:
        env.close()


def smoke(args):
    require_ready(False)
    from stable_baselines3.common.env_checker import check_env
    from .proxy import PracticeEnv
    from .training import make_model
    env = PracticeEnv(duration=5, reward_config=reward_settings(args))
    try:
        check_env(env, warn=True)
        model = make_model(env, reward_settings(args), args.seed)
        model.learn(total_timesteps=128)
        observation, _ = env.reset(seed=args.seed)
        action, _ = model.predict(observation, deterministic=True)
        env.step(action)
        print("[smoke] PASS: feature extraction contract, action/reward API and one real PPO update; no capture/input.")
    finally:
        env.close()


def evaluate_command(args):
    require_ready(False)
    from .training import load_model
    from .checkpoints import resolve_checkpoint, write_json_atomic
    from .proxy import PracticeEnv
    from .teacher import evaluate
    config = reward_settings(args)
    directory = Path(args.models)/("practice" if args.practice else "live")
    resolved = resolve_checkpoint(directory, args.model, config)
    if resolved is None:
        raise ValueError("No compatible feature checkpoint. Run agent bootstrap first.")
    path, metadata = resolved
    model = load_model(path)
    env = PracticeEnv(reward_config=config)
    try:
        metrics = evaluate(model, env, range(args.seed, args.seed+args.episodes))
    finally:
        env.close()
    report = {"model": path.name, "checkpoint_domain": metadata["domain"], "evaluation": metrics}
    destination = Path(args.output) if args.output else directory/"evaluation.json"
    write_json_atomic(destination, report)
    print(f"[evaluate] PRACTICE ONLY: mean {metrics['mean_score']:.1f}, "
          f"median {metrics['median_score']:.1f}, "
          f"scored in {metrics['scoring_episodes']}/{metrics['episodes']} episodes. "
          "Real-game skill remains unmeasured.")
    print(f"[evaluate] Report: {destination.resolve()}")


def parser():
    root = argparse.ArgumentParser(prog="crosim-agent", description="Small feature-based MoSim learner; offline bootstrap + live PPO")
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser("check", help="check packages/calibration before taking over the game")
    p.add_argument("--offline", action="store_true")
    p.set_defaults(func=lambda args: 0 if preflight(not args.offline)["ready"] else 1)
    p = commands.add_parser("reset",help="archive all project training data and clear automatic resume paths")
    def reset_command(args):
        from .session import reset_training
        archive,moved = reset_training(HERE)
        print(f"[reset] Cleared active training: {', '.join(moved) or 'already empty'}")
        print(f"[reset] Previous files archived at {archive}; calibration and vision weights preserved.")
    p.set_defaults(func=reset_command)
    from .demonstrations import teach_command,fit_command
    for name, function in (("teach",teach_command),("imitate",fit_command),("train", train), ("play", play), ("watch", watch), ("bootstrap", bootstrap_command),
                           ("smoke", smoke), ("evaluate", evaluate_command)):
        p = commands.add_parser(name)
        p.add_argument("--models", default=str(DEFAULT_MODELS))
        p.add_argument("--model", help="explicit compatible feature-agent checkpoint")
        p.add_argument("--rewards", help="JSON overrides for versioned reward settings")
        p.add_argument("--seed", type=int, default=123)
        p.add_argument("--no-overlay", action="store_true")
        p.add_argument("--corner", choices=("top-right", "top-left", "bottom-right", "bottom-left"), default="top-right")
        p.add_argument("--fresh", action="store_true", help="create a separate new checkpoint, preserving previous models")
        p.add_argument("--bootstrap-samples", type=int, default=6000)
        p.add_argument("--bootstrap-epochs", type=int, default=12)
        if name in ('teach','imitate'):
            p.add_argument('--demos',default=str(HERE/'mosim_demos'/'human_features'))
            p.add_argument('--epochs',type=int,default=10)
            p.add_argument('--reaction-ticks',type=int,choices=range(6),default=2)
        if name == "train":
            p.add_argument("--offline", action="store_true", help="practice world only; does not capture or control the game")
            p.add_argument("--steps", type=int, default=None, help="additional decisions; 0 runs continuously (live default); offline default 10000")
        if name == "play":
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--episodes", type=int, default=1)
        if name == "evaluate":
            p.add_argument("--practice", action="store_true", help="load the practice rather than live checkpoint folder")
            p.add_argument("--episodes", type=int, default=12)
            p.add_argument("--output", help="write the practice evaluation JSON to this path")
            p.set_defaults(seed=9001)
        p.set_defaults(func=function)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ("steps", "bootstrap_samples", "bootstrap_epochs", "episodes", "epochs"):
        value = getattr(args,name,None)
        if value is not None and (value < 0 if name == "steps" else value <= 0):
            raise SystemExit(f"--{name.replace('_', '-')} must be {'nonnegative' if name == 'steps' else 'positive'}")
    try:
        return args.func(args) or 0
    except Exception as exc:
        print(f"[agent] {type(exc).__name__}: {exc}", file=sys.stderr)
        if os.environ.get("CROSIM_DEBUG") == "1":
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
