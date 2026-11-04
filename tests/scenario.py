"""Deterministic gameplay inputs used for regression against the supplied v4."""
import contextlib
import tempfile
import threading
from unittest.mock import patch

import numpy as np


class FakeGuard:
    def __init__(self, release, title, input_lock=None):
        self.status = "LIVE"
        self.release = release
        self.input_lock = input_lock or threading.RLock()

    def wait(self):
        return 0.0

    def check_input(self):
        return True

    def close(self):
        self.release()


class FakePDI:
    def __init__(self):
        self.events = []

    def keyDown(self, key):
        self.events.append(("down", key))

    def keyUp(self, key):
        self.events.append(("up", key))


def scenario_frames(config):
    frames = []
    for index in range(16):
        # Alternate view motion and stillness, then fill/drain the hopper.
        frame = np.full((360, 640, 3), 35 + (index % 4)*25, dtype=np.uint8)
        if 3 <= index <= 8:
            l, t, r, b = config["hopper_region"]
            frame[int(t*360):int(b*360), int(l*640):int(r*640)] = [0, 255, 255]
        frames.append(frame)
    return frames


ACTIONS = [[], ["w"], ["w", "shift"], ["w", "shift"], ["space"],
           ["space"], ["space"], ["a", "d", "j", "l"], ["h"], ["h"],
           [], ["h"], [], ["w"], ["w"]]


@contextlib.contextmanager
def fake_env(module, log_dir=None, inject=True, dry_run=False):
    class Camera:
        def __init__(self):
            self.frames = iter(scenario_frames(module.CFG))

        def grab(self):
            return next(self.frames)

        def release(self):
            pass

    pdi = FakePDI()
    is_new = hasattr(module, "RewardLedger")
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(module, "Grabber", Camera))
        stack.enter_context(patch.object(module, "need_pydirectinput", return_value=pdi))
        stack.enter_context(patch.object(module.time, "sleep", return_value=None))
        if is_new:
            stack.enter_context(patch("mosim_controls.InputGuard", FakeGuard))
            options = {"overlay": False, "log_rewards": log_dir is not None,
                       "log_dir": log_dir}
            env = module.MoSimEnv(verbose=False, debug_options=options, inject_inputs=inject, dry_run=dry_run)
        else:
            env = module.MoSimEnv(verbose=False)
        env._wait_for_tick = lambda: None
        points = iter([0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        env._score_reader.update = lambda frame, now: next(points)
        updates = iter([True]*5 + [False]*4 + [True]*7)
        env._hub.update = lambda frame, elapsed: (next(updates), 10.0)
        try:
            yield env, pdi
        finally:
            env.close()


def run_scenario(module):
    results = []
    with fake_env(module) as (env, pdi):
        env.reset()
        for index, keys in enumerate(ACTIONS):
            env._elapsed = lambda i=index: (i+1)*0.1
            action = np.array([int(k in keys) for k in module.ACTION_KEYS], dtype=np.int8)
            obs, reward, terminated, truncated, info = env.step(action)
            results.append({"reward": reward, "breakdown": dict(env.breakdown),
                            "fill": info["fill"], "motion": info["motion"],
                            "hub_active": info["hub_active"], "aux": obs["aux"].tolist()})
    return results
