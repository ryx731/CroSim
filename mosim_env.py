"""
MoSimEnv - Custom Gymnasium environment for MoSimulator (FRC REBUILT)
Screen-capture based RL environment using bettercam (DXGI) + pydirectinput.

Install:
    pip install bettercam pydirectinput gymnasium stable-baselines3 opencv-python torch numpy pytesseract

You ALSO need the separate Tesseract-OCR binary installed on Windows
(pytesseract just calls out to it): https://github.com/UB-Mannheim/tesseract/wiki

QUICK START (run with the game window visible on monitor 0):
    python mosim_env.py                 # interactive menu - start here if unsure
    python mosim_env.py doctor          # check installs, paths, config, GPU
    python mosim_env.py setup           # hands-free calibration (hotkeys + voice; do this FIRST)
    python mosim_env.py record          # record your own gameplay as demonstrations
    python mosim_env.py demos           # list / weight / disable recorded demos
    python mosim_env.py pretrain        # behavior-clone the demos into the PPO network
    python mosim_env.py train           # PPO. Auto-resumes; starts from the pretrained net
    python mosim_env.py play            # watch the trained agent, with live reward readout
    python mosim_env.py watch           # debug rewards while YOU play; sends no keys
    python mosim_env.py debug-preview --image setup_reference.png  # offline UI example
    python mosim_env.py reset           # delete checkpoints (never touches your demos)
Every command accepts --help.

CALIBRATION LIVES IN mosim_config.json, NOT IN THIS FILE.
`setup` writes the screen regions, colour thresholds and motion threshold it
measures into mosim_config.json next to this script. You do not need to edit
Python to re-tune anything; re-run `setup` (or edit the JSON) and every command
picks the new values up. The constants below are only the starting defaults.

WHAT CHANGED IN v4 (and why)
  * Hub-activity awareness. REBUILT only lets you score while YOUR hub is
    active, so shooting at a dead hub is wasted time. The env now tracks the
    blue hub's state two independent ways - a calibrated vision check on the
    hub indicator, and the published REBUILT phase schedule (20 s auto, 3 s
    pause, 10 s transition, 4 x 25 s shifts, 30 s endgame) - and feeds it to
    the network, rewards scoring while active, and penalises shooting while
    inactive. If neither source is trustworthy it fails open (assumes active)
    so it can never block real scoring.
  * Longer memory. The frame stack is now strided, so the network sees roughly
    1.5 s of history instead of 0.4 s. Short stacks are the main reason an
    agent cannot tell "I am part-way through a cycle" from "I just spawned".
  * Richer state. Alongside the frames the network gets: previous action, time
    left, hub active, time until the hub flips, hopper fill, which camera view
    is up, and how long since the last swap.
  * Balanced behaviour cloning. Plain cloning on a MultiBinary action space
    collapses onto whichever keys you press most and ignores the rest. Rare
    keys are now up-weighted, and `pretrain` reports per-key precision/recall
    against an "always off" baseline so key-mashing is visible immediately.
  * Human reaction delay. Your keypress at tick t answers what you saw a few
    ticks earlier. Labels are shifted by REACTION_DELAY_STEPS so the network
    learns cause, not coincidence.
  * Conflicting-key penalty. Holding W+S or J+L is a null command; it now costs
    something, which discourages the mashing that independent per-key sampling
    tends to produce.
  * Honest match clock. Time lost to PPO's gradient updates no longer counts
    against the episode timer or accrues penalties, so the reward function
    only ever scores real gameplay.

ON MIXING DIFFERENT RUNS TOGETHER (read this - it is a real limitation)
    The policy outputs one independent probability per key. If half your demos
    go left around the field at a given moment and half go right, cloning
    learns "50% left, 50% right" and sampling produces a blend of two good
    routes, which is usually a bad route. Three things here fight that, and one
    does not:
      1. More context (the strided stack + state vector above) means fewer
         moments actually look identical, so there is less to average over.
      2. `play` picks the most likely action instead of sampling, and cancels
         contradictory keys, so it commits to one route.
      3. PPO fine-tuning collapses the tie toward whichever route actually
         scores.
      4. What does NOT fix it: more demos of more routes. The averaging is
         structural. If you want the agent to commit to one approach, record
         demos that agree with each other, and use `demos weight` to lean on
         your best runs.

NOTE ON OLD FILES: the observation now has extra inputs, so networks and
checkpoints saved before v4 cannot be loaded; new ones use *_v4 names. Your
recorded demos ARE still used - every older demo format is upgraded on load,
with camera swaps, hub state and hopper fill recovered from the stored video.
"""

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from mosim_debug import REWARD_LABELS, RewardLedger, RewardLog

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_IMPORT_ERROR = None
except Exception as exc:  # doctor/demos still work without it installed
    gym, spaces, GYM_IMPORT_ERROR = None, None, exc


HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Lazy imports
#
# torch, stable-baselines3, bettercam, pydirectinput and pytesseract are heavy
# and/or Windows-only. Importing them only when a command actually needs them
# means `doctor`, `demos` and `--help` start instantly and still work on a
# machine where the game isn't installed.
# --------------------------------------------------------------------------- #

def need_torch():
    import torch
    return torch


def need_sb3():
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
    return PPO, BaseCallback, DummyVecEnv, VecMonitor, VecNormalize


def need_bettercam():
    import bettercam
    return bettercam


_PDI = None


def need_pydirectinput():
    global _PDI
    if _PDI is None:
        import pydirectinput
        pydirectinput.PAUSE = 0.0  # no built-in delay between calls; we time our own
        _PDI = pydirectinput
    return _PDI


_TESSERACT_CMD = None


def find_tesseract(refresh: bool = False):
    """
    Locate tesseract.exe: the configured path first, then PATH, then the usual
    Windows install folders. Returns None if it can't be found. The answer is
    cached, so the OCR loop doesn't hit the disk every call.
    """
    global _TESSERACT_CMD
    if _TESSERACT_CMD and not refresh:
        return _TESSERACT_CMD
    candidates = []
    if CFG.tesseract_cmd:
        candidates.append(CFG.tesseract_cmd)
    on_path = shutil.which("tesseract")
    if on_path:
        candidates.append(on_path)
    roots = [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"),
             os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs")]
    for root in roots:
        if root:
            candidates.append(os.path.join(root, "Tesseract-OCR", "tesseract.exe"))
    for path in candidates:
        if path and os.path.isfile(path):
            _TESSERACT_CMD = path
            return path
    return None


def need_pytesseract():
    os.environ.setdefault("OMP_THREAD_LIMIT","1")
    import pytesseract
    cmd = find_tesseract()
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    return pytesseract


def torch_device():
    torch = need_torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Config file
#
# Anything `setup` can measure lives here, so you never have to edit Python to
# re-calibrate. Missing keys fall back to the defaults below, so upgrading this
# script won't invalidate a config you already wrote.
# --------------------------------------------------------------------------- #

CONFIG_PATH = os.path.join(HERE, "mosim_config.json")

CONFIG_DEFAULTS = {
    # --- Tesseract ---
    # Leave "" if tesseract.exe is on your PATH.
    "tesseract_cmd": r"C:\Program Files\Tesseract-OCR\tesseract.exe",

    # --- Score OCR (all written by `setup`; the defaults reproduce the old behaviour) ---
    "ocr_mode": "fixed",          # "fixed" = one brightness cut-off, "otsu" = automatic
    "ocr_threshold": 150,         # the cut-off used by "fixed" (0-255)
    "ocr_invert": False,          # True = flip to dark-on-light before OCR
    "ocr_scale": 3,               # how much the score box is enlarged first

    # --- Setup wizard: hotkeys that work from any window, so you never need
    # the terminal while the game is fullscreen. Pick keys the game doesn't use.
    # (Avoid F-keys on a laptop: they usually need Fn held.)
    "setup_go_key": "]",
    "setup_skip_key": "[",
    "setup_voice": True,          # spoken prompts through Windows' built-in voice

    # --- Capture ---
    "monitor_index": 0,
    "game_window_title": "MoSim",  # MoSim / MoSimulator aliases; pauses on focus loss

    # --- Screen regions, as (left, top, right, bottom) fractions of the full
    # captured frame. Fractions scale automatically to any resolution. ---
    # Blue alliance score on the HUD.
    "score_region": [0.365, 0.905, 0.455, 0.965],
    # The robot's own hopper cavity under the chase camera.
    "hopper_region": [0.45, 0.48, 0.57, 0.60],
    # The blue hub indicator / arrow in the bottom corner. This default is a
    # guess - `setup` measures the real one.
    "hub_region": [0.03, 0.86, 0.20, 0.99],

    # --- Colour thresholds (OpenCV HSV, hue 0-179) ---
    "fuel_hsv_lower": [20, 100, 100],
    "fuel_hsv_upper": [35, 255, 255],
    "hub_hsv_lower": [20, 90, 120],
    "hub_hsv_upper": [40, 255, 255],

    # --- Measured thresholds (all written by `setup`) ---
    "motion_diff_threshold": 1.0,
    "hub_active_coverage": 0.06,     # full-res indicator coverage that means "active"
    "hub_calibrated": False,         # until True, the vision check is not trusted
    "hub_lowres_coverage": 0.06,     # same test on the 160x90 mask plane, for old demos
    "lowres_fill_gain": 1.0,         # maps low-res hopper coverage onto the full-res scale

    # --- Input timing ---
    "camera_tap_hold_seconds": 0.05,  # raise if the game misses H taps
    "tap_hold_seconds": 0.05,         # for K

    # --- Hub schedule (seconds). FRC REBUILT phase structure; adjust if
    # MoSimulator's timing differs from the real game. ---
    "match_auto_seconds": 20.0,
    "match_pause_seconds": 3.0,
    "match_transition_seconds": 10.0,
    "match_shift_seconds": 25.0,
    "match_shift_count": 4,
    "match_endgame_seconds": 30.0,
    # Whether the BLUE hub is active during shifts 1 and 3 (True) or 2 and 4
    # (False). Depends on who won auto; the env re-checks this against vision
    # during a match and corrects itself when it can.
    "blue_active_in_odd_shifts": True,
    "trust_schedule": True,           # use the schedule when vision is unavailable

    # --- Per-demo weights, keyed by filename. 0 disables a demo. ---
    "demo_weights": {},
}


class Config(dict):
    """Dict with attribute access, backed by mosim_config.json."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value

    @classmethod
    def load(cls):
        data = dict(CONFIG_DEFAULTS)
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                    data.update(json.load(fh))
            except Exception as exc:
                print(f"[config] {CONFIG_PATH} could not be read ({exc}); using defaults.")
        return cls(data)

    def save(self):
        with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
            json.dump(dict(self), fh, indent=2, sort_keys=True)
        print(f"[config] wrote {CONFIG_PATH}")

    # Convenience accessors that return the right numpy types.
    def hsv(self, which):
        lo = np.array(self[f"{which}_hsv_lower"], dtype=np.uint8)
        hi = np.array(self[f"{which}_hsv_upper"], dtype=np.uint8)
        return lo, hi

    def region(self, which):
        return tuple(self[f"{which}_region"])


CFG = Config.load()


# --------------------------------------------------------------------------- #
# Constants that are NOT calibration (change these in code if you want to)
# --------------------------------------------------------------------------- #

# --- Vision preprocessing ---
FRAME_W, FRAME_H = 160, 90       # what the CNN sees
MID_W, MID_H = 320, 180          # intermediate size; keeps small fuel visible in the mask
PLANES_PER_FRAME = 2             # plane 0 = grayscale, plane 1 = fuel (yellow) mask
MASK_BOOST = 4                   # brightens the downsampled mask so small balls survive

# --- Temporal context ---
# The stack holds FRAME_STACK frames spaced FRAME_STRIDE ticks apart, so it
# spans (FRAME_STACK-1)*FRAME_STRIDE ticks of history. At 10 Hz that's 1.5 s
# with the defaults, versus 0.4 s for a plain 4-frame stack. The extra span is
# what lets the network tell "mid-cycle" from "just started" - the single
# biggest lever on whether it learns your routines or just reacts.
FRAME_STACK = 6
FRAME_STRIDE = 3
HISTORY_TICKS = (FRAME_STACK - 1) * FRAME_STRIDE + 1
IMAGE_CHANNELS = FRAME_STACK * PLANES_PER_FRAME

# --- Control rate ---
# The env steps at a steady CONTROL_HZ (sleeping if it's ahead), so per-second
# rewards, frame-stack spacing and your recorded demos all line up.
CONTROL_HZ = 10
STEP_PERIOD = 1.0 / CONTROL_HZ
STEP_DT = STEP_PERIOD

# Match length is NOT a constant any more: it's the sum of the phase durations
# in mosim_config.json (HubSchedule.total), so the clock and the hub schedule
# can never drift apart. The REBUILT defaults add up to 163 s. If MoSimulator's
# match is a different length, change match_*_seconds in the config, not here.

# Discount factor, shared by PPO, reward normalization and the shaping term.
# 0.995 at 10 Hz is a ~20 s horizon: long enough for intake -> drive -> shoot.
GAMMA = 0.995

# Key used to instantly restart the match.
RESET_KEY = "r"

# --- Action mapping (UNCHANGED - every keybinding is the same as before) ---
# All of these can be held simultaneously (MultiBinary, not Discrete) EXCEPT
# "k" and "i", which are edge-triggered: one tap toggles hopper extend/retract
# (K) or swaps the camera view (H) rather than being held down.
ACTION_KEYS = [
    "w",      # 0: Forward
    "a",      # 1: Left
    "s",      # 2: Back
    "d",      # 3: Right
    "j",      # 4: Rotate CCW
    "l",      # 5: Rotate CW
    "shift",  # 6: Intake
    "space",  # 7: Score into hub
    "k",      # 8: Toggle extend/retract hopper (TAP ONLY)
    "h",      # 9: Swap camera view (TAP ONLY)
]
N_ACTIONS = len(ACTION_KEYS)
TAP_ONLY_INDICES = {8, 9}
CAMERA_IDX = ACTION_KEYS.index("h")
HOPPER_IDX = ACTION_KEYS.index("k")
INTAKE_IDX = ACTION_KEYS.index("shift")
SCORE_ACTION_INDEX = ACTION_KEYS.index("space")
FORWARD_IDX = ACTION_KEYS.index("w")
LEFT_IDX = ACTION_KEYS.index("a")
BACK_IDX = ACTION_KEYS.index("s")
RIGHT_IDX = ACTION_KEYS.index("d")
ROT_CCW_IDX = ACTION_KEYS.index("j")
ROT_CW_IDX = ACTION_KEYS.index("l")
MOVEMENT_ACTION_INDICES = {FORWARD_IDX, LEFT_IDX, BACK_IDX, RIGHT_IDX, ROT_CCW_IDX, ROT_CW_IDX}
# One character per key, for the live readout in `play` ("shift" and "s" would
# otherwise both print as S).
KEY_GLYPHS = ["W", "A", "S", "D", "J", "L", "^", "*", "K", "H"]
# Pairs that cancel each other out: holding both is a null command.
OPPOSING_PAIRS = ((FORWARD_IDX, BACK_IDX), (LEFT_IDX, RIGHT_IDX), (ROT_CCW_IDX, ROT_CW_IDX))

# --- Auxiliary (non-image) observation layout ---
# Index 0..N_ACTIONS-1 : previous action bits
AUX_TIME_FRAC = N_ACTIONS + 0      # how far through the match we are
AUX_HUB_ACTIVE = N_ACTIONS + 1     # 1 while the BLUE hub is scoring
AUX_HUB_CHANGE_SOON = N_ACTIONS + 2  # ramps 0 -> 1 over the last HUB_LOOKAHEAD seconds
AUX_FILL = N_ACTIONS + 3           # hopper fill, 0-1
AUX_VIEW_PARITY = N_ACTIONS + 4    # which camera view is up (swap count parity)
AUX_SWAP_RECENCY = N_ACTIONS + 5   # 0 right after an H tap, 1 once it's long past
AUX_DIM = N_ACTIONS + 6

HUB_LOOKAHEAD_SECONDS = 10.0       # horizon for the "hub is about to flip" input

# --- Real-motion detection ---
# Mean absolute grayscale difference between consecutive frames. Above the
# calibrated threshold = the view is really changing = the robot is really
# moving. Driving into a wall or holding W+S produces ~no change, so neither
# dodges the stillness penalty.

# --- Camera swap (H key) ---
VIEW_CHANGE_SETTLE_STEPS = 2       # steps after an H tap where measured motion is meaningless
CAMERA_SWAP_COOLDOWN_STEPS = 10    # swaps closer together than this count as spam
# Finding swaps in demos recorded before the I key was tracked: a one-frame
# change that is both large and much larger than the change around it, so fast
# driving (large but sustained) isn't mistaken for a cut.
CAMERA_SWAP_MIN_DIFF = 12.0
CAMERA_SWAP_ISOLATION_RATIO = 3.0
CAMERA_SWAP_CONTEXT_STEPS = 5

# --------------------------------------------------------------------------- #
# Reward weights
#
# Everything below is per real second unless it says otherwise, so changing
# CONTROL_HZ doesn't silently change the size of any reward. The env applies
# them using the fixed STEP_DT rather than measured wall time, so a slow frame
# or a gradient-update pause can't produce a freak penalty.
# --------------------------------------------------------------------------- #

# The only true objective. Everything else is shaping that nets to ~zero over a
# full cycle, so it can steer behaviour without inventing points.
SCORE_REWARD_PER_POINT = 10.0

# Standing still wastes match time.
STILLNESS_PENALTY_PER_SECOND = -0.30
# Much smaller while actually shooting at a live hub - stopping to aim is fine.
STILLNESS_PENALTY_WHILE_SCORING_PER_SECOND = -0.005

# Continuous time cost, sized so a whole match costs this much in total.
STEP_PENALTY_TOTAL_PER_EPISODE = -2.0

# Hub discipline. Shooting at a dead hub throws fuel away and takes time; this
# is the direct answer to "only score while the blue hub is active".
WASTED_SHOT_PENALTY_PER_SECOND = -0.40    # holding SPACE while the hub is inactive
EMPTY_SHOT_PENALTY_PER_SECOND = -0.05     # holding SPACE with an empty hopper
# Small credit for being loaded the moment the hub goes live. Potential-based
# (see below) so waiting around for it can't be farmed.
HUB_READY_POTENTIAL_SCALE = 1.5

# Anti-jitter. One charge per key state change between consecutive steps.
# Mashing flips half the keys every step and gets punished hard; holding a
# steady set costs nothing. An isolated camera swap is exempt.
KEY_CHANGE_PENALTY = -0.01
# Holding two keys that cancel (W+S, A+D, J+L) is a null command.
CONFLICT_PENALTY_PER_SECOND = -0.15

# --- Fuel shaping (potential-based, so it CANNOT be farmed) ---
# Reward each step is GAMMA * phi(s') - phi(s). Intaking pays immediately and
# shooting pays it back, so going around the loop nets ~0 and only real score
# reward survives. That gives a dense "intake is good, be loaded when the hub
# opens" signal without creating a hoarding exploit.
FUEL_HOLD_THRESHOLD_FRACTION = 0.35   # coverage that counts as a full hopper
FUEL_POTENTIAL_SCALE = 3.0            # shaping value of a completely full hopper
FILL_EMA_ALPHA = 0.3                  # smoothing on the noisy fill estimate

# --- Score OCR ---
# OCR is the slowest thing in the loop, so it only runs when the score box
# visibly changed (or every SCORE_OCR_MAX_INTERVAL_SECONDS as a safety net).
SCORE_CHANGE_FRACTION = 0.01
SCORE_OCR_MAX_INTERVAL_SECONDS = 2.0
MAX_SCORE_JUMP_PER_READ = 40          # bigger jumps are treated as misreads

# --- Hub detection smoothing ---
HUB_CONFIRM_STEPS = 3                 # agreeing reads needed before the state flips

# --- Lost-time compensation ---
# If a step's wall gap is much longer than STEP_PERIOD, the game kept running
# while we weren't playing (almost always a PPO gradient update). That time is
# excluded from the match clock so the reward function only ever scores real
# gameplay, and so episodes aren't silently cut short.
# A whole second is far longer than any slow frame or OCR call, and far
# shorter than a gradient update, so this only ever catches the real thing.
MAX_EXPECTED_STEP_GAP = max(STEP_PERIOD * 5.0, 1.0)

# --- Paths (v4: the observation changed, so older saves can't be loaded) ---
DEMO_DIR = os.path.join(HERE, "mosim_demos")
DISABLED_DEMO_DIR = os.path.join(DEMO_DIR, "disabled")
BC_INIT_PATH = os.path.join(HERE, "mosim_bc_init_v4")       # .zip appended by SB3
CHECKPOINT_DIR = os.path.join(HERE, "mosim_checkpoints_v4")
FINAL_MODEL_PATH = os.path.join(HERE, "mosim_ppo_model_final_v4.zip")
VECNORM_PATH = os.path.join(HERE, "mosim_vecnormalize_v4.pkl")
TENSORBOARD_DIR = os.path.join(HERE, "mosim_tensorboard")

# --- PPO hyperparameters ---
N_STEPS = 1024                 # ~100 s of play per update at 10 Hz
BATCH_SIZE = 128
N_EPOCHS = 4
SCRATCH_LR = 3e-4
SCRATCH_ENT_COEF = 0.01        # keeps a from-scratch policy exploring
FINETUNE_LR = 1e-4             # gentler when starting from a behavior-cloned net
FINETUNE_ENT_COEF = 0.002      # don't let the entropy bonus wash out what BC learned
VALUE_WARMUP_STEPS = 3000      # train only the value head first, so a random critic
                               # can't wreck the cloned policy in the first updates
KEEP_CHECKPOINTS = 10          # newest per-episode saves to keep on disk

# --- Behavior cloning ---
BC_ITERATIONS = 6000
BC_BATCH_SIZE = 64
BC_LR = 3e-4
BC_MIN_LR_FRACTION = 0.1       # cosine decay floor
# Your keypress at tick t is a response to what you saw a few ticks earlier.
# Labels are shifted by this many ticks so the network learns cause, not
# coincidence. 2 ticks = 200 ms at 10 Hz, about human reaction time.
REACTION_DELAY_STEPS = 2
# Fraction of samples where the "previous action" input is zeroed. Without this
# the network learns the shortcut "repeat the last action" (human input is very
# repetitive) instead of reading the screen. This is the single most important
# guard against an agent that just mashes your most-used keys.
BC_PREV_ACTION_DROPOUT = 0.6
# Rare keys (H, K, and often SPACE) are drowned out by W/A/S/D unless their
# positive examples are up-weighted. Weight per key is
# ((1-p)/p) ** BC_BALANCE_POWER, clipped to BC_MAX_POS_WEIGHT, applied only to
# the frames where the key WAS pressed.
BC_BALANCE_POWER = 0.5
BC_MAX_POS_WEIGHT = 12.0
# Mild brightness jitter on the grayscale plane, so the network keys off shapes
# and layout instead of memorising exact pixel values.
BC_BRIGHTNESS_JITTER = 12.0
BC_EVAL_EVERY = 250


# --------------------------------------------------------------------------- #
# Vision helpers (shared by the env, the recorder, the calibrator and the
# demo loader, so training and recording can never disagree about what a
# frame means)
# --------------------------------------------------------------------------- #

def crop_frac(image: np.ndarray, frac) -> np.ndarray:
    """Crop a (left, top, right, bottom) fractional box out of an image."""
    h, w = image.shape[:2]
    left, top, right, bottom = frac
    return image[int(top * h):int(bottom * h), int(left * w):int(right * w)]


def preprocess_planes(frame: np.ndarray) -> np.ndarray:
    """
    Full-res BGR frame -> (2, 90, 160) uint8: [grayscale, fuel mask].
    The mask is computed at 320x180 and then area-downsampled, so small yellow
    balls survive as faint blobs instead of vanishing.
    """
    mid = cv2.resize(frame, (MID_W, MID_H), interpolation=cv2.INTER_AREA)

    gray_mid = cv2.cvtColor(mid, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray_mid, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)

    lo, hi = CFG.hsv("fuel")
    hsv = cv2.cvtColor(mid, cv2.COLOR_BGR2HSV)
    mask_mid = cv2.inRange(hsv, lo, hi)
    mask = cv2.resize(mask_mid, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
    mask = np.clip(mask.astype(np.uint16) * MASK_BOOST, 0, 255).astype(np.uint8)

    return np.stack([gray, mask], axis=0)


def hopper_fill_fraction(frame: np.ndarray) -> float:
    """Fraction of the hopper region covered by fuel-coloured pixels (0-1)."""
    crop = crop_frac(frame, CFG.region("hopper"))
    if crop.size == 0:
        return 0.0
    lo, hi = CFG.hsv("fuel")
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lo, hi)
    return float(np.count_nonzero(mask)) / mask.size


def hopper_fill_from_planes(planes: np.ndarray) -> float:
    """
    Same measurement from the stored 160x90 mask plane, for demos recorded
    before fill was saved. Coarser, so `lowres_fill_gain` (set by `setup`)
    rescales it onto the full-res scale.
    """
    crop = crop_frac(planes[1], CFG.region("hopper"))
    if crop.size == 0:
        return 0.0
    return float(crop.mean()) / 255.0 * float(CFG.lowres_fill_gain)


def normalized_fill(fill: float) -> float:
    """Hopper fill as 0-1, where 1 means 'full enough to matter'."""
    return float(min(max(fill / max(FUEL_HOLD_THRESHOLD_FRACTION, 1e-6), 0.0), 1.0))


def hub_indicator_coverage(frame: np.ndarray) -> float:
    """Fraction of the hub-indicator box lit in the indicator's colour (0-1)."""
    crop = crop_frac(frame, CFG.region("hub"))
    if crop.size == 0:
        return 0.0
    lo, hi = CFG.hsv("hub")
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lo, hi)
    return float(np.count_nonzero(mask)) / mask.size


def hub_coverage_from_planes(planes: np.ndarray) -> float:
    """Hub-indicator coverage from the stored mask plane, for older demos."""
    crop = crop_frac(planes[1], CFG.region("hub"))
    if crop.size == 0:
        return 0.0
    return float(crop.mean()) / 255.0


OCR_TESS_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789"


def binarize_with(crop: np.ndarray, mode: str = "fixed", threshold: int = 150,
                  invert: bool = False, scale: float = 3) -> np.ndarray:
    """
    Score box -> clean black-and-white image for Tesseract.
      mode "fixed": pixels brighter than `threshold` become white
      mode "otsu" : the cut-off is picked automatically per image
      invert      : flip to dark-on-light (Tesseract often prefers this)
    A plain border is added at the end; Tesseract is much less reliable on
    digits that touch the edge of the image.
    """
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if mode == "otsu":
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, thresh = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)
    if invert:
        thresh = cv2.bitwise_not(thresh)
    background = 255 if float(thresh.mean()) > 127 else 0
    return cv2.copyMakeBorder(thresh, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=int(background))


def binarize_score_crop(crop: np.ndarray) -> np.ndarray:
    """The score box preprocessed with whatever `setup` found works on your screen."""
    return binarize_with(crop, CFG.ocr_mode, CFG.ocr_threshold, CFG.ocr_invert, CFG.ocr_scale)


def ocr_read(binary: np.ndarray, timeout=None, psm=7):
    """OCR an already-binarized image. Returns None if it isn't a clean integer."""
    pytesseract = need_pytesseract()
    options = {} if timeout is None else {"timeout": timeout}
    config = OCR_TESS_CONFIG if psm == 7 else f"--psm {int(psm)} -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(binary, config=config, **options).strip()
    # Scoreboards do not pad integers with leading zeroes. Those extra glyphs
    # are a common isolated-digit OCR artifact (for example, "07" for "0").
    clean = text.isascii() and text.isdigit() and (len(text) == 1 or not text.startswith("0"))
    return int(text) if clean else None


def ocr_score_crop(crop: np.ndarray, timeout=None):
    """OCR the score box. Returns None if it isn't a clean integer."""
    if crop.size == 0:
        return None
    binary = binarize_score_crop(crop)
    # Tesseract's line layout can discard an isolated scoreboard digit. Try
    # word layout after removing blank margins. Keep ALL foreground pixels so
    # a multi-digit score cannot be reduced to a single digit.
    # Never turn a blank read into zero.
    # Share one timeout across both attempts to keep async reads bounded.
    started = time.perf_counter()
    reading = ocr_read(binary, timeout=timeout)
    if reading is not None:
        return reading
    background_white = float(binary[:10].mean()) >= 127
    normalized = binary if background_white else cv2.bitwise_not(binary)
    ink = cv2.findNonZero(cv2.bitwise_not(normalized))
    if ink is None:
        return None
    x, y, width, height = cv2.boundingRect(ink)
    pad = max(10, height//8)
    tight = cv2.copyMakeBorder(normalized[y:y+height, x:x+width], pad, pad, pad, pad,
                              cv2.BORDER_CONSTANT, value=255)
    remaining = None if timeout is None else timeout-(time.perf_counter()-started)
    if remaining is not None and remaining <= 0.01:
        return None
    return ocr_read(tight, timeout=remaining, psm=8)


# --------------------------------------------------------------------------- #
# Hub activity
#
# REBUILT only scores into an ACTIVE hub, and the two alliances' hubs take
# turns. Knowing when yours is live is the difference between a cycle and a
# wasted 25 seconds, so the env tracks it two independent ways:
#
#   1. HubSchedule - the published phase structure (auto / pause / transition /
#      four shifts / endgame). Needs no vision at all; only ambiguity is which
#      alliance sits out shift 1, which depends on who won auto.
#   2. HubVision - a calibrated colour check on the indicator in the corner,
#      debounced so one bad frame can't flip the state.
#
# HubTracker combines them: vision leads when it's calibrated, the schedule
# fills in otherwise, and the two vote to settle the shift-parity ambiguity.
# If neither is trustworthy it reports ACTIVE, so a broken detector can never
# stop the agent from scoring - it just stops the extra hub shaping.
# --------------------------------------------------------------------------- #

class HubSchedule:
    """REBUILT match phases -> is the blue hub active, and when does it flip."""

    def __init__(self):
        self.auto = float(CFG.match_auto_seconds)
        self.pause = float(CFG.match_pause_seconds)
        self.transition = float(CFG.match_transition_seconds)
        self.shift = float(CFG.match_shift_seconds)
        self.shifts = int(CFG.match_shift_count)
        self.endgame = float(CFG.match_endgame_seconds)
        self.blue_odd = bool(CFG.blue_active_in_odd_shifts)

        self.shift_start = self.auto + self.pause + self.transition
        self.endgame_start = self.shift_start + self.shifts * self.shift
        self.total = self.endgame_start + self.endgame

    def phase(self, t: float) -> str:
        if t < self.auto:
            return "auto"
        if t < self.auto + self.pause:
            return "pause"
        if t < self.shift_start:
            return "transition"
        if t < self.endgame_start:
            return f"shift{int((t - self.shift_start) // self.shift) + 1}"
        if t < self.total:
            return "endgame"
        return "over"

    def blue_active(self, t: float, blue_odd: bool = None) -> bool:
        """Both hubs are live in auto, transition and endgame; shifts alternate."""
        if blue_odd is None:
            blue_odd = self.blue_odd
        name = self.phase(t)
        if name in ("auto", "pause", "transition", "endgame", "over"):
            return True
        index = int(name[len("shift"):])       # 1-based
        return (index % 2 == 1) == blue_odd

    def seconds_to_change(self, t: float, blue_odd: bool = None) -> float:
        """How long the current blue-hub state lasts. Capped at the lookahead."""
        now = self.blue_active(t, blue_odd)
        horizon = HUB_LOOKAHEAD_SECONDS
        step = 0.25
        ahead = step
        while ahead <= horizon:
            if self.blue_active(t + ahead, blue_odd) != now:
                return ahead
            ahead += step
        return horizon


class HubVision:
    """Debounced colour check on the hub indicator. None = not calibrated."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = None
        self._pending = None
        self._pending_count = 0
        self.last_coverage = 0.0

    def update(self, frame: np.ndarray):
        if not CFG.hub_calibrated:
            return None
        coverage = hub_indicator_coverage(frame)
        self.last_coverage = coverage
        reading = coverage >= float(CFG.hub_active_coverage)
        if reading == self.state:
            self._pending, self._pending_count = None, 0
            return self.state
        if reading == self._pending:
            self._pending_count += 1
        else:
            self._pending, self._pending_count = reading, 1
        if self._pending_count >= HUB_CONFIRM_STEPS or self.state is None:
            self.state = reading
            self._pending, self._pending_count = None, 0
        return self.state


class HubTracker:
    """Combines vision and schedule into one answer plus a countdown."""

    def __init__(self):
        self.schedule = HubSchedule()
        self.vision = HubVision()
        self.reset()

    def reset(self):
        self.vision.reset()
        self.blue_odd = bool(CFG.blue_active_in_odd_shifts)
        self._odd_votes = 0      # +1 each time vision agrees with blue_odd, -1 otherwise
        self.source = "assumed"

    def update(self, frame: np.ndarray, elapsed: float):
        """Returns (blue_hub_active, seconds_until_it_changes)."""
        seen = self.vision.update(frame)

        # During a shift, vision tells us which parity is right. Only shifts are
        # informative - every other phase has both hubs on.
        if seen is not None and self.schedule.phase(elapsed).startswith("shift"):
            agrees = (seen == self.schedule.blue_active(elapsed, self.blue_odd))
            self._odd_votes += 1 if agrees else -1
            if self._odd_votes <= -4:            # vision has clearly disagreed
                self.blue_odd = not self.blue_odd
                self._odd_votes = 0

        if seen is not None:
            self.source = "vision"
            return seen, self.schedule.seconds_to_change(elapsed, self.blue_odd)
        if CFG.trust_schedule:
            self.source = "schedule"
            return (self.schedule.blue_active(elapsed, self.blue_odd),
                    self.schedule.seconds_to_change(elapsed, self.blue_odd))
        # Fail open: never block the agent from scoring.
        self.source = "assumed"
        return True, HUB_LOOKAHEAD_SECONDS


def hub_active_from_planes(planes: np.ndarray, elapsed: float, schedule: HubSchedule) -> bool:
    """Hub state for an old demo: mask plane if calibrated, else the schedule."""
    if CFG.hub_calibrated:
        return hub_coverage_from_planes(planes) >= float(CFG.hub_lowres_coverage)
    return schedule.blue_active(elapsed)


# --------------------------------------------------------------------------- #
# Score reading
# --------------------------------------------------------------------------- #

class ScoreReader:
    """
    Turns noisy per-frame OCR into "how many NEW points did we just score".

    - Only runs OCR when the score box visibly changed (big speed-up).
    - A new value must be read twice in a row before it's believed.
    - The first confirmed read after reset() is a baseline only, so a stale
      score from the previous match can't pay out.
    - Drops (match reset) and implausible jumps re-baseline without reward.
    """

    def __init__(self, ocr_timeout=None, read_crop=None, max_interval=None):
        self.ocr_timeout = ocr_timeout
        self.read_crop = read_crop
        self.max_interval = SCORE_OCR_MAX_INTERVAL_SECONDS if max_interval is None else max_interval
        self.reset()

    def reset(self):
        self.confirmed = None
        self._pending = None
        self._last_bin = None
        self._last_ocr_time = -1e9
        self.ocr_failures = 0

    @staticmethod
    def _small_binary(crop: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        return binary

    def update(self, frame: np.ndarray, now: float) -> int:
        crop = crop_frac(frame, CFG.region("score"))
        return self.update_crop(crop, now)

    def update_crop(self, crop: np.ndarray, now: float) -> int:
        """Same confirmation logic when an asynchronous caller crops first."""
        if crop.size == 0:
            return 0

        small_bin = self._small_binary(crop)
        changed = (
            self._last_bin is None
            or self._last_bin.shape != small_bin.shape
            or np.count_nonzero(small_bin != self._last_bin) / small_bin.size
            > SCORE_CHANGE_FRACTION
        )
        due = (now - self._last_ocr_time) > self.max_interval
        if not (changed or due or self._pending is not None):
            return 0

        self._last_bin = small_bin
        self._last_ocr_time = now
        read = (self.read_crop or ocr_score_crop)(crop, timeout=self.ocr_timeout)
        if read is None:
            self.ocr_failures += 1
            return 0

        if self.confirmed is not None and read == self.confirmed:
            self._pending = None
            return 0

        if read == self._pending:  # second matching read -> believe it
            self._pending = None
            previous = self.confirmed
            self.confirmed = read
            if previous is None:
                return 0  # baseline only
            delta = read - previous
            if 0 < delta <= MAX_SCORE_JUMP_PER_READ:
                return delta
            return 0  # score dropped (match reset) or jump too big -> re-baseline

        self._pending = read
        return 0


# --------------------------------------------------------------------------- #
# Screen capture and input
# --------------------------------------------------------------------------- #

class Grabber:
    """bettercam wrapper: never spins forever if the screen stops updating."""

    def __init__(self):
        bettercam = need_bettercam()
        self.camera = bettercam.create(output_idx=int(CFG.monitor_index), output_color="BGR")
        self._last = None
        self._last_capture_time = 0.0
        self._capture_lock = threading.Lock()

    def grab(self) -> np.ndarray:
        return self.grab_timed()[0]

    def grab_timed(self):
        # DXGI capture is shared by the reward loop and fast overlay worker;
        # only one thread may acquire/map the native surface at a time.
        with self._capture_lock:
            return self._grab_timed_unlocked()

    def _grab_timed_unlocked(self):
        deadline = time.perf_counter() + 0.5
        while True:
            frame = self.camera.grab()
            if frame is not None:  # bettercam returns None until a new frame is ready
                self._last = frame
                self._last_capture_time = time.perf_counter()
                return frame, self._last_capture_time
            if self._last is not None:
                # A static desktop can have no new frame. Reuse immediately,
                # preserving its REAL timestamp rather than waiting half a second.
                return self._last, self._last_capture_time
            if time.perf_counter() > deadline:
                raise RuntimeError("No screen frame received. Check the capture monitor and game display mode.")
            time.sleep(0.001)

    @property
    def bounds(self):
        # Use the SAME DXGI output as capture, including negative desktop offsets.
        rect = self.camera._output.desc.DesktopCoordinates
        return (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))

    def release(self):
        try:
            self.camera.release()
        except Exception:
            pass


VK_CODES = {
    "w": 0x57, "a": 0x41, "s": 0x53, "d": 0x44,
    "j": 0x4A, "l": 0x4C, "shift": 0x10, "space": 0x20,
    "k": 0x4B, "h": 0x48,
}


def read_keys_now() -> np.ndarray:
    """
    Current physical key state as a length-N_ACTIONS 0/1 vector (Windows only).
    Uses GetAsyncKeyState, so it works whichever window has focus and doesn't
    interfere with the game receiving your keys.

    Tap-only keys (K, H) also count as pressed if they were pressed AND
    released since the previous poll (the API's "pressed since last call" bit),
    so a quick tap between two 10 Hz samples isn't missed.
    """
    import ctypes

    user32 = ctypes.windll.user32
    out = np.zeros(N_ACTIONS, dtype=np.int8)
    for idx, key in enumerate(ACTION_KEYS):
        state = user32.GetAsyncKeyState(VK_CODES[key])
        down = bool(state & 0x8000)
        tapped = idx in TAP_ONLY_INDICES and bool(state & 0x0001)
        out[idx] = 1 if (down or tapped) else 0
    return out


def escape_pressed() -> bool:
    import ctypes
    return bool(ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000)


def tap_key(key: str, hold: float = None):
    pdi = need_pydirectinput()
    if hold is None:
        hold = float(CFG.tap_hold_seconds)
    pdi.keyDown(key)
    time.sleep(hold)
    pdi.keyUp(key)


class FrameStack:
    """
    Holds the last HISTORY_TICKS plane-pairs and returns FRAME_STACK of them
    spaced FRAME_STRIDE apart, so the network sees ~1.5 s of history rather
    than the last 0.4 s.
    """

    def __init__(self):
        self._frames = deque(maxlen=HISTORY_TICKS)

    def reset(self, planes: np.ndarray):
        self._frames.clear()
        for _ in range(HISTORY_TICKS):
            self._frames.append(planes)

    def push(self, planes: np.ndarray):
        self._frames.append(planes)

    def latest(self) -> np.ndarray:
        return self._frames[-1]

    def image(self) -> np.ndarray:
        """(FRAME_STACK * 2, 90, 160) uint8, oldest first."""
        frames = list(self._frames)
        picked = [frames[-1 - i * FRAME_STRIDE] for i in range(FRAME_STACK)][::-1]
        return np.concatenate(picked, axis=0)


def stack_indices(t: int) -> np.ndarray:
    """
    Which recorded ticks make up the stack for tick t, matching FrameStack
    exactly (including the repeated first frame at the start of an episode).
    """
    idx = np.array([t - i * FRAME_STRIDE for i in range(FRAME_STACK)][::-1])
    return np.clip(idx, 0, None)


# --------------------------------------------------------------------------- #
# Observation / action spaces
# --------------------------------------------------------------------------- #

def build_observation_space():
    if spaces is None:
        raise SystemExit(f"gymnasium isn't installed: {GYM_IMPORT_ERROR}\n"
                         "Run `pip install gymnasium`, or `python mosim_env.py doctor`.")
    return spaces.Dict(
        {
            # FRAME_STACK strided frames x (grayscale plane + fuel-mask plane).
            "image": spaces.Box(
                low=0, high=255, shape=(IMAGE_CHANNELS, FRAME_H, FRAME_W), dtype=np.uint8
            ),
            # Everything the pixels don't say clearly: what we just did, how
            # much match is left, whether the hub is live and for how long,
            # how loaded we are, and which camera view is up.
            "aux": spaces.Box(low=0.0, high=1.0, shape=(AUX_DIM,), dtype=np.float32),
        }
    )


def build_aux(prev_action, time_frac, hub_active, seconds_to_hub_change,
              fill, view_parity, steps_since_swap) -> np.ndarray:
    aux = np.zeros(AUX_DIM, dtype=np.float32)
    aux[:N_ACTIONS] = np.asarray(prev_action, dtype=np.float32)
    aux[AUX_TIME_FRAC] = float(min(max(time_frac, 0.0), 1.0))
    aux[AUX_HUB_ACTIVE] = 1.0 if hub_active else 0.0
    aux[AUX_HUB_CHANGE_SOON] = float(
        min(max(1.0 - seconds_to_hub_change / HUB_LOOKAHEAD_SECONDS, 0.0), 1.0)
    )
    aux[AUX_FILL] = normalized_fill(fill)
    aux[AUX_VIEW_PARITY] = 1.0 if view_parity else 0.0
    aux[AUX_SWAP_RECENCY] = float(
        min(max(steps_since_swap / float(CAMERA_SWAP_COOLDOWN_STEPS), 0.0), 1.0)
    )
    return aux


def net_movement(action) -> tuple:
    """
    Collapse the six movement keys into a net (forward, strafe, rotate)
    command, each in {-1, 0, +1}. Opposing keys cancel, so W+S or J+L is no
    motion on that axis.
    """
    return (
        int(action[FORWARD_IDX]) - int(action[BACK_IDX]),
        int(action[RIGHT_IDX]) - int(action[LEFT_IDX]),
        int(action[ROT_CW_IDX]) - int(action[ROT_CCW_IDX]),
    )


def count_conflicts(action) -> int:
    """How many opposing key pairs are held at once."""
    return sum(1 for a, b in OPPOSING_PAIRS if action[a] and action[b])


def resolve_conflicts(action: np.ndarray, probs=None) -> np.ndarray:
    """
    Drop contradictory keys. Used when the agent PLAYS: sampling each key
    independently can produce W+S (a blend of "go forward" and "go back"), the
    classic symptom of averaging two different routes together. With
    probabilities available we keep the key the network actually preferred;
    without them we drop both.
    """
    out = np.array(action, dtype=np.int8)
    for a, b in OPPOSING_PAIRS:
        if out[a] and out[b]:
            if probs is None:
                out[a] = out[b] = 0
            elif probs[a] >= probs[b]:
                out[b] = 0
            else:
                out[a] = 0
    return out


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #

class MoSimEnv(gym.Env if gym is not None else object):
    """
    Gymnasium environment that plays MoSimulator via screen capture.

    Reward, in one place:
        + SCORE_REWARD_PER_POINT per point the blue alliance gains  <- the real goal
        + potential-based shaping for holding fuel, extra while the hub is live
        - stillness, unless shooting at a live hub
        - a steady time cost
        - shooting at a dead hub, or with an empty hopper
        - key changes (anti-mash) and contradictory keys

    The shaping terms are all potential-based or bounded, so the only way to
    make a large positive return is to actually score.
    """

    metadata = {"render_modes": []}
    reward_labels = REWARD_LABELS

    def __init__(self, verbose: bool = True, debug_options=None, inject_inputs=True,
                 dry_run=False):
        if gym is None:
            raise SystemExit(f"gymnasium isn't installed: {GYM_IMPORT_ERROR}")
        super().__init__()

        self.verbose = verbose
        self.grabber = None
        self._guard = self._overlay = self._vision = self._reward_log = None
        self._input_lock = threading.RLock()
        self._inject_inputs = bool(inject_inputs and not dry_run)
        self._mode = "agent" if self._inject_inputs else "dry_run" if dry_run else "watch"
        self._debug_options = debug_options or {}
        self._ledger = RewardLedger(labels=self.reward_labels)

        # MultiBinary: any subset of the keys can be active at once, since
        # REBUILT often needs 3-4 held simultaneously.
        self.action_space = spaces.MultiBinary(N_ACTIONS)
        self.observation_space = build_observation_space()

        self._stack = FrameStack()
        self._score_reader = ScoreReader()
        self._hub = HubTracker()
        self.episode_seconds = self._hub.schedule.total

        # ---- Held-key state ----
        self._held_keys = set()
        self._prev_tap_state = np.zeros(N_ACTIONS, dtype=np.int8)

        # ---- Input transition costs ----
        self._prev_action = np.zeros(N_ACTIONS, dtype=np.int8)

        # ---- Fuel shaping ----
        self._fill_ema = 0.0
        self._prev_phi = 0.0

        # ---- Camera swap ----
        self._view_change_steps = 0
        self._settle_credit = False
        self._rapid_swap = False
        self._steps_since_swap = 10 ** 6
        self._view_parity = 0
        self._last_real_moved = False

        # ---- Episode bookkeeping ----
        self._episode_start_time = None
        self._lost_seconds = 0.0
        self._last_step_wall = None
        self._next_tick = 0.0
        self._hub_active = True
        self._hub_change_in = HUB_LOOKAHEAD_SECONDS
        self.breakdown = {}
        self.episode_points = 0

        # Hardware and UI start only after all state needed for cleanup exists.
        try:
            from mosim_controls import InputGuard
            from mosim_overlay import DebugOverlay, enable_dpi_awareness
            from mosim_vision import DetectionWorker, ObjectDetector
            enable_dpi_awareness()
            options = self._debug_options
            # Validate/load custom weights before opening capture or injecting keys.
            detector = ObjectDetector(CFG, options.get("weights"),
                                      conf=options.get("conf", 0.4),
                                      imgsz=options.get("imgsz", 960),
                                      device=options.get("vision_device")) if options.get("overlay", True) else None
            self._guard = InputGuard(self._release_all_keys, CFG.game_window_title,
                                     input_lock=self._input_lock)
            self.grabber = Grabber()
            if options.get("overlay", True):
                self._vision = DetectionWorker(detector, fps=options.get("vision_fps", 30),
                                               frame_source=self._grab_vision_frame,
                                               enabled=self._vision_enabled)
            if options.get("log_rewards", True):
                self._reward_log = RewardLog(options.get("log_dir", os.path.join(HERE, "mosim_debug_logs")),
                                             {"mode": self._mode, "config": dict(CFG),
                                              "weights": str(detector.weights) if detector and detector.weights else None,
                                              "reward_weights": {
                                                  name: value for name, value in globals().items()
                                                  if name.isupper() and isinstance(value, (float, int))}})
                print(f"[debug] every reward tick -> {self._reward_log.path}")
            if options.get("overlay", True):
                self._overlay = DebugOverlay(self.grabber.bounds, self._vision, self._guard,
                                              options.get("corner", "top-right"))
            print("[controls] Focus MoSim and press ] to start. ] pauses/resumes; [ stops.")
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------------ #
    # Timing
    # ------------------------------------------------------------------ #

    def _wait_for_tick(self):
        """Sleep until the next control tick, so every step lasts ~STEP_PERIOD."""
        now = time.perf_counter()
        if now < self._next_tick:
            time.sleep(self._next_tick - now)
            self._next_tick += STEP_PERIOD
        else:
            self._next_tick = now + STEP_PERIOD  # behind schedule: resync

    def _account_for_lost_time(self):
        """
        The game keeps running during PPO's gradient updates, but the agent
        isn't playing then. That gap is subtracted from the match clock, so
        the reward function only ever grades real gameplay and episodes aren't
        silently truncated by a slow update.
        """
        now = time.perf_counter()
        if self._last_step_wall is not None:
            gap = now - self._last_step_wall
            if gap > MAX_EXPECTED_STEP_GAP:
                self._lost_seconds += gap - STEP_PERIOD
        self._last_step_wall = now

    def _elapsed(self) -> float:
        return max(time.time() - self._episode_start_time - self._lost_seconds, 0.0)

    def _wait_ready(self):
        waited = self._guard.wait()
        if self._episode_start_time is not None:
            self._lost_seconds += waited
        self._last_step_wall = time.perf_counter()

    def _grab_focused(self):
        # Focus can change while capture is waiting. Discard that image instead
        # of treating another app's pixels as game state or rewarding them.
        while True:
            self._wait_ready()
            timed_grab = getattr(self.grabber, "grab_timed", None)
            if timed_grab is not None:
                frame, captured = timed_grab()
            else:
                frame, captured = self.grabber.grab(), time.perf_counter()
            with self._input_lock:
                if self._guard.check_input():
                    self._latest_capture_at = captured
                    return frame

    def _vision_enabled(self):
        return (not self._guard.paused and not self._guard.stopped.is_set()
                and not self._guard.closed.is_set() and self._guard.focused())

    def _grab_vision_frame(self):
        if not self._vision_enabled():
            return None
        frame, captured = self.grabber.grab_timed()
        return (frame, captured) if self._vision_enabled() else None

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #

    def _get_obs(self) -> dict:
        elapsed = self._elapsed()
        return {
            "image": self._stack.image(),
            "aux": build_aux(
                prev_action=self._prev_action,
                time_frac=elapsed / self.episode_seconds,
                hub_active=self._hub_active,
                seconds_to_hub_change=self._hub_change_in,
                fill=self._fill_ema,
                view_parity=self._view_parity,
                steps_since_swap=self._steps_since_swap,
            ),
        }

    # ------------------------------------------------------------------ #
    # Reward pieces
    # ------------------------------------------------------------------ #

    def _potential(self, fill_ema: float, hub_active: bool) -> float:
        """
        Phi(s) for the shaping term. Holding fuel is worth something; holding
        fuel while the hub is live is worth more, which is what pulls the agent
        toward "be loaded and near the hub when it opens". Because the reward
        is GAMMA*Phi(s') - Phi(s), every bit of this is paid back when the
        state reverses, so none of it can be farmed.
        """
        f = normalized_fill(fill_ema)
        return FUEL_POTENTIAL_SCALE * f + (HUB_READY_POTENTIAL_SCALE * f if hub_active else 0.0)

    def _update_shaping(self, frame: np.ndarray, hub_active: bool, rebaseline: bool) -> float:
        fill = hopper_fill_fraction(frame)
        if rebaseline:
            # Right after a camera swap the hopper box looks at a different
            # picture, so re-baseline instead of paying out on the jump.
            # Still charge (GAMMA-1)*phi, the normal cost of one step of
            # holding potential: returning a flat 0 here would make every swap
            # a free step, and spamming H would farm it.
            self._fill_ema = fill
            phi = self._potential(fill, hub_active)
            self._prev_phi = phi
            return (GAMMA - 1.0) * phi
        self._fill_ema = FILL_EMA_ALPHA * fill + (1.0 - FILL_EMA_ALPHA) * self._fill_ema
        phi = self._potential(self._fill_ema, hub_active)
        shaping = GAMMA * phi - self._prev_phi
        self._prev_phi = phi
        return shaping

    def _control_costs(self, action: np.ndarray):
        """
        Returns (key_change_penalty, conflict_penalty). No movement bonus.
        Call exactly once per step: it also rolls the previous-action state.
        """
        changes = 0
        for i in range(N_ACTIONS):
            cur, prev = int(action[i]), int(self._prev_action[i])
            if i in TAP_ONLY_INDICES:
                # Only a fresh tap counts, and an isolated camera swap is free -
                # repeated camera changes are tracked separately.
                if cur and not prev and (i != CAMERA_IDX or self._rapid_swap):
                    changes += 1
            elif cur != prev:
                changes += 1
        key_change_penalty = KEY_CHANGE_PENALTY * changes
        self._last_key_changes = changes
        conflict_penalty = CONFLICT_PENALTY_PER_SECOND * STEP_DT * count_conflicts(action)

        self._prev_action = np.array(action, dtype=np.int8)
        return key_change_penalty, conflict_penalty

    # ------------------------------------------------------------------ #
    # Input injection
    # ------------------------------------------------------------------ #

    def _release_all_keys(self):
        with self._input_lock:
            if not self._held_keys:
                return
            pdi = need_pydirectinput()
            for key in list(self._held_keys):
                pdi.keyUp(key)
                self._held_keys.discard(key)

    def release_all_keys(self):
        """Called from a callback while PPO updates, so the robot stops driving."""
        self._release_all_keys()
        self._prev_action[:] = 0

    def _apply_action(self, action: np.ndarray):
        """
        Hold every key whose bit is 1 for as long as it stays 1; release it the
        moment it goes back to 0. Tap-only keys fire one quick press+release on
        the rising edge.
        """
        pdi = need_pydirectinput() if self._inject_inputs else None
        for i, key in enumerate(ACTION_KEYS):
            want_active = bool(action[i])

            if i in TAP_ONLY_INDICES:
                if want_active and not self._prev_tap_state[i]:
                    hold = (float(CFG.camera_tap_hold_seconds) if i == CAMERA_IDX
                            else float(CFG.tap_hold_seconds))
                    if pdi is not None and not self._tap_guarded(key, hold):
                        return False
                    if i == CAMERA_IDX:
                        self._rapid_swap = self._steps_since_swap < CAMERA_SWAP_COOLDOWN_STEPS
                        self._settle_credit = not self._rapid_swap
                        self._steps_since_swap = 0
                        self._view_change_steps = VIEW_CHANGE_SETTLE_STEPS
                        self._view_parity ^= 1
                self._prev_tap_state[i] = int(want_active)
                continue

            if pdi is not None:
                with self._input_lock:
                    if not self._guard.check_input():
                        return False
                    is_held = key in self._held_keys
                    if want_active and not is_held:
                        self._held_keys.add(key)
                        pdi.keyDown(key)
                    elif not want_active and is_held:
                        pdi.keyUp(key)
                        self._held_keys.discard(key)
        return True

    def _tap_guarded(self, key, hold=None):
        with self._input_lock:
            if not self._guard.check_input():
                return False
            pdi = need_pydirectinput()
            self._held_keys.add(key)
            try:
                pdi.keyDown(key)
                time.sleep(float(CFG.tap_hold_seconds) if hold is None else hold)
            finally:
                pdi.keyUp(key)
                self._held_keys.discard(key)
        return True

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #

    def _reset_game(self):
        if self._inject_inputs:
            while not self._tap_guarded(RESET_KEY):
                self._wait_ready()
            time.sleep(0.5)

    def _episode_has_ended(self,elapsed):
        return elapsed >= self.episode_seconds

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._wait_ready()

        self._release_all_keys()
        self._prev_tap_state[:] = 0
        self._view_change_steps = 0
        self._settle_credit = False
        self._rapid_swap = False
        self._steps_since_swap = 10 ** 6
        self._view_parity = 0
        self._last_real_moved = False

        self._prev_action[:] = 0

        self._reset_game()

        first_raw = self._grab_focused()
        self._latest_frame = first_raw
        self._stack.reset(preprocess_planes(first_raw))

        self._score_reader.reset()
        self._hub.reset()
        self.episode_seconds = self._hub.schedule.total

        # Start shaping from the current fill, so there's no fake first-step jump.
        self._fill_ema = hopper_fill_fraction(first_raw)
        self._hub_active, self._hub_change_in = self._hub.update(first_raw, 0.0)
        self._prev_phi = self._potential(self._fill_ema, self._hub_active)

        self._episode_start_time = time.time()
        self._lost_seconds = 0.0
        self._last_step_wall = time.perf_counter()
        self._next_tick = time.perf_counter() + STEP_PERIOD
        self.breakdown = {name: 0.0 for name in self.reward_labels}
        self.episode_points = 0

        self._ledger.reset()
        if self._vision:
            self._vision.reset()
        if self._reward_log:
            self._reward_log.write({"type": "reset", "episode": self._ledger.episode})
        self._publish_debug(first_raw, 0.0, 0.0, False)

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action).reshape(-1)
        if len(action) != N_ACTIONS or not np.isin(action, [0, 1]).all():
            raise ValueError(f"Expected {N_ACTIONS} binary action bits")
        action = action.astype(np.int8)

        self._account_for_lost_time()
        self._wait_ready()
        self._rapid_swap = False
        while not self._apply_action(action):
            self._wait_ready()
        self._wait_for_tick()  # let the action show up on screen

        raw_frame = self._grab_focused()
        self._latest_frame = raw_frame
        planes = preprocess_planes(raw_frame)
        elapsed = self._elapsed()
        self._on_frame(raw_frame, planes, elapsed)

        # --- Is the robot really moving? ---
        prev_gray = self._stack.latest()[0].astype(np.int16)
        motion = float(np.mean(np.abs(planes[0].astype(np.int16) - prev_gray)))
        view_changing = self._view_change_steps > 0
        if view_changing:
            self._view_change_steps -= 1
            # A single swap keeps the movement status from just before it, so
            # swapping while driving isn't punished. Rapid repeats stay "not
            # moving", so spamming H can't dodge the stillness penalty.
            moved = self._settle_credit and self._last_real_moved
        else:
            moved = motion > float(CFG.motion_diff_threshold)
            self._last_real_moved = moved

        # --- Hub state ---
        self._hub_active, self._hub_change_in = self._hub.update(raw_frame, elapsed)

        # --- Reward ---
        points = self._score_reader.update(raw_frame, time.perf_counter())
        self.episode_points += points
        score_reward = points * SCORE_REWARD_PER_POINT
        previous_phi = self._prev_phi
        shaping = self._update_shaping(raw_frame, self._hub_active, rebaseline=view_changing)

        shooting = bool(action[SCORE_ACTION_INDEX])
        if moved:
            stillness = 0.0
        elif shooting and self._hub_active:
            stillness = STILLNESS_PENALTY_WHILE_SCORING_PER_SECOND * STEP_DT
        else:
            stillness = STILLNESS_PENALTY_PER_SECOND * STEP_DT

        # The direct "only shoot at a live hub" signal.
        wasted_shot = 0.0
        if shooting and not self._hub_active:
            wasted_shot += WASTED_SHOT_PENALTY_PER_SECOND * STEP_DT
        if shooting and normalized_fill(self._fill_ema) < 0.05:
            wasted_shot += EMPTY_SHOT_PENALTY_PER_SECOND * STEP_DT

        time_penalty = (STEP_PENALTY_TOTAL_PER_EPISODE / self.episode_seconds) * STEP_DT
        key_change, conflict = self._control_costs(action)

        reward = (score_reward + shaping + stillness + time_penalty
                  + wasted_shot + key_change + conflict)

        components = {"score": score_reward, "shaping": shaping,
                      "stillness": stillness, "time": time_penalty,
                      "wasted_shot": wasted_shot, "key_change": key_change,
                      "conflict": conflict}
        shot_reasons = []
        if shooting and not self._hub_active:
            shot_reasons.append("shooting at inactive blue hub")
        if shooting and normalized_fill(self._fill_ema) < 0.05:
            shot_reasons.append("shooting with empty hopper estimate")
        reasons = {
            "score": f"Blue alliance gained {points} point(s); not individual robot credit",
            "shaping": "Camera swap rebaseline" if view_changing else "Discounted change in fuel / hub potential",
            "stillness": "Still while shooting at live hub" if shooting and self._hub_active else "View motion below threshold",
            "time": "Fixed control tick time cost",
            "wasted_shot": "; ".join(shot_reasons) or "No wasted shot",
            "key_change": f"{self._last_key_changes} charged key transition(s)",
            "conflict": "Opposing movement keys held together",
        }
        components, reasons, reward_extra = self._reward_terms(
            components, reasons, frame=raw_frame, action=action, elapsed=elapsed,
            points=points, motion=motion, moved=moved, view_changing=view_changing)
        reward = float(math.fsum(components.values()))
        for name, value in components.items():
            self.breakdown[name] += value
        self._ledger.record(components, elapsed, reasons)

        self._stack.push(planes)
        self._steps_since_swap += 1
        obs = self._get_obs()

        truncated = self._episode_has_ended(elapsed)
        terminated = False  # no visual match-end detection; the clock handles it

        info = {
            "points": points,
            "motion": motion,
            "hub_active": self._hub_active,
            "hub_source": self._hub.source,
            "fill": self._fill_ema,
            "reward_components": dict(components),
            "reward_reasons": dict(reasons),
            "reward_total": float(reward),
            "episode_return": self._ledger.total,
            "reward_diagnostics": {"phi_before": previous_phi, "phi_after": self._prev_phi,
                                   "gamma": GAMMA, "camera_rebaseline": view_changing,
                                   "moved": bool(moved), "shooting": shooting,
                                   "key_changes": self._last_key_changes,
                                   "conflicts": count_conflicts(action), **reward_extra},
        }
        self._publish_debug(raw_frame, elapsed, motion, moved)
        if self._reward_log:
            self._reward_log.write({"type": "step", "episode": self._ledger.episode,
                                    "step": self._ledger.step, "elapsed": elapsed,
                                    "action": action.tolist(), "mode": self._mode,
                                    "reward": float(reward), "components": components,
                                    "reasons": reasons, "total": self._ledger.total,
                                    "episode_points": self.episode_points,
                                    "confirmed_score": self._score_reader.confirmed,
                                    "pending_score": self._score_reader._pending,
                                    "ocr_failures": self._score_reader.ocr_failures,
                                    "diagnostics": info["reward_diagnostics"],
                                    "motion": motion, "fill": self._fill_ema,
                                    "hub_active": bool(self._hub_active), "hub_source": self._hub.source,
                                    "vision": self._vision.latest() if self._vision else None})
        if truncated:
            info["episode_points"] = self.episode_points
            info["reward_breakdown"] = dict(self.breakdown)
            info["lost_seconds"] = self._lost_seconds
            if self.verbose:
                print(f"\n[episode] {self.episode_points} pts | "
                      + " ".join(f"{k}={v:+.1f}" for k, v in self.breakdown.items())
                      + f" | {self._lost_seconds:.0f}s lost to updates")
        return obs, float(reward), terminated, truncated, info

    def _on_frame(self, frame, planes, elapsed):
        """Extension point for structured observations; v4 pixel mode is unchanged."""

    def _reward_terms(self, components, reasons, **measurements):
        """Extension point for separately versioned learning objectives."""
        return components, reasons, {}

    def _publish_debug(self, frame, elapsed, motion, moved):
        if self._vision:
            self._vision.submit(frame)
        if not self._overlay:
            return
        confirmed = self._score_reader.confirmed
        state = {
            "updated_at": time.perf_counter(), "ledger": self._ledger.snapshot(),
            "mode": self._mode,
            "elapsed": elapsed, "duration": self.episode_seconds,
            "phase": self._hub.schedule.phase(elapsed), "points": self.episode_points,
            "hub_active": bool(self._hub_active), "hub_source": self._hub.source,
            "fill_normalized": normalized_fill(self._fill_ema), "motion": motion,
            "motion_threshold": float(CFG.motion_diff_threshold), "moved": bool(moved),
            "keys": [key for i, key in enumerate(ACTION_KEYS) if self._prev_action[i]],
            "regions": [
                {"box": CFG.score_region, "label": f"Score ROI (OCR): {confirmed if confirmed is not None else 'pending'}"},
                {"box": CFG.hopper_region, "label": f"Hopper ROI: {normalized_fill(self._fill_ema):.0%} (color estimate)"},
                {"box": CFG.hub_region, "label": f"Hub indicator ROI: {self._hub.source}"},
            ],
        }
        state.update(self._debug_state())
        self._overlay.publish(state)

    def _debug_state(self):
        return {}

    def close(self):
        try:
            if self._guard:
                self._guard.close()
            else:
                self._release_all_keys()
        except Exception:
            pass
        for resource in (self._overlay, self._vision, self._reward_log):
            if resource:
                resource.close()
        if self.grabber:
            self.grabber.release()


class SpacesOnlyEnv(gym.Env if gym is not None else object):
    """Real spaces, no game. Lets pretraining build the same network offline."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = build_observation_space()
        self.action_space = spaces.MultiBinary(N_ACTIONS)

    def reset(self, *, seed=None, options=None):
        # Returns zeros so SB3 can construct a model around it; step() still
        # raises, so accidentally trying to TRAIN on this fails loudly.
        return {
            "image": np.zeros((IMAGE_CHANNELS, FRAME_H, FRAME_W), dtype=np.uint8),
            "aux": np.zeros(AUX_DIM, dtype=np.float32),
        }, {}

    def step(self, action):
        raise RuntimeError("SpacesOnlyEnv has no game attached; it only exists "
                           "so pretraining can build the same network offline.")


# --------------------------------------------------------------------------- #
# Demonstration files
#
# Format v2 stores the raw per-tick key state rather than pre-built labels, so
# the reaction-delay alignment and the aux layout can change later without
# making your recordings worthless. Older formats are upgraded on load:
#   v0 - 9 keys, no I column          -> I recovered from the video
#   v1 - 10 keys, prebuilt aux        -> keys recovered from actions + aux
# In both cases hub state and hopper fill are recovered from the stored mask
# plane (or the match schedule), so every demo ends up with the same fields.
# --------------------------------------------------------------------------- #

DEMO_FORMAT_VERSION = 2


def infer_camera_swaps(planes: np.ndarray) -> np.ndarray:
    """
    Find camera swaps in an old demo from the video alone. Returns (T,) uint8
    with a 1 at every tick whose action was (very likely) an I tap, i.e. the
    tick right BEFORE the picture jumped.

    A swap is a frame-to-frame change that is both big in absolute terms and
    much larger than the typical change in the ~1 s around it, so fast driving
    or spinning (large but sustained) isn't mistaken for a cut.
    """
    T = len(planes)
    labels = np.zeros(T, dtype=np.uint8)
    if T < 3:
        return labels
    gray = planes[:, 0].astype(np.int16)
    diffs = np.abs(gray[1:] - gray[:-1]).mean(axis=(1, 2))
    window = CAMERA_SWAP_CONTEXT_STEPS
    for t in range(len(diffs)):
        if diffs[t] <= CAMERA_SWAP_MIN_DIFF:
            continue
        around = np.concatenate([diffs[max(0, t - window):t], diffs[t + 1:t + 1 + window]])
        baseline = float(np.median(around)) if len(around) else 0.0
        if diffs[t] > CAMERA_SWAP_ISOLATION_RATIO * baseline:
            labels[t] = 1
    return labels


def load_demo(path: str) -> dict:
    """Read any demo version and return a normalized record."""
    data = np.load(path)
    keys_in = set(data.files)
    planes = data["planes"]
    T = len(planes)
    schedule = HubSchedule()
    notes = []

    if "keys" in keys_in:                                   # v2
        version = int(data["version"]) if "version" in keys_in else 2
        keys = data["keys"].astype(np.int8)[:T]
        time_frac = data["time_frac"].astype(np.float32)[:T]
        hub = data["hub"].astype(np.int8)[:T]
        fill = data["fill"].astype(np.float32)[:T]
    else:                                                   # v0 / v1
        actions = data["actions"].astype(np.int8)
        aux = data["aux"].astype(np.float32)
        n_keys = actions.shape[1]
        version = 1 if n_keys == N_ACTIONS else 0

        if version == 0:                                    # recorded before the I key
            swaps = infer_camera_swaps(planes)
            actions = np.concatenate([actions, swaps[:len(actions), None]], axis=1)
            notes.append(f"{int(swaps.sum())} camera swaps recovered from the video")

        # aux[:, :n] is the key state at tick t; actions[t] is the state at t+1.
        prev_keys = aux[:, :n_keys].astype(np.int8)
        if version == 0:
            prev_keys = np.concatenate(
                [prev_keys, np.concatenate([[0], actions[:-1, CAMERA_IDX]])[:, None].astype(np.int8)],
                axis=1,
            )
        # prev_keys[t] is the key state at the tick whose frame is planes[t],
        # which is exactly what we want. (actions[t] is the state one tick
        # later; its own frame was dropped when the demo was written.)
        keys = np.zeros((T, N_ACTIONS), dtype=np.int8)
        n = min(T, len(prev_keys))
        keys[:n] = prev_keys[:n]
        if n < T and len(actions):
            keys[n:] = actions[-1]
        time_frac = np.zeros(T, dtype=np.float32)
        n = min(T, aux.shape[0])
        time_frac[:n] = aux[:n, -1]
        if n < T:
            time_frac[n:] = time_frac[n - 1] if n else 0.0

        elapsed = time_frac * schedule.total
        hub = np.array(
            [1 if hub_active_from_planes(planes[t], float(elapsed[t]), schedule) else 0
             for t in range(T)], dtype=np.int8
        )
        fill = np.array([hopper_fill_from_planes(planes[t]) for t in range(T)],
                        dtype=np.float32)
        notes.append("hub state and hopper fill reconstructed from the stored video")

    swap_rate = float(keys[:, CAMERA_IDX].mean()) if len(keys) else 0.0
    if version == 0 and swap_rate > 0.02:
        notes.append(f"WARNING: {swap_rate * 100:.0f}% of ticks were read as camera swaps, "
                     "which is implausibly high - raise CAMERA_SWAP_MIN_DIFF")
    elif version == 0 and keys[:, CAMERA_IDX].sum() == 0:
        notes.append("no camera swaps found - if you did swap views, lower CAMERA_SWAP_MIN_DIFF")

    return {
        "path": path,
        "name": os.path.basename(path),
        "version": version,
        "planes": planes,
        "keys": keys,
        "time_frac": time_frac,
        "hub": hub,
        "fill": fill,
        "notes": notes,
    }


def save_demo(path, planes, keys, time_frac, hub, fill):
    np.savez_compressed(
        path,
        version=np.int32(DEMO_FORMAT_VERSION),
        planes=np.stack(planes),
        keys=np.stack(keys).astype(np.uint8),
        time_frac=np.asarray(time_frac, dtype=np.float32),
        hub=np.asarray(hub, dtype=np.uint8),
        fill=np.asarray(fill, dtype=np.float32),
        control_hz=np.float32(CONTROL_HZ),
    )


def demo_files(include_disabled: bool = False):
    files = sorted(glob.glob(os.path.join(DEMO_DIR, "demo_*.npz")))
    if include_disabled:
        files += sorted(glob.glob(os.path.join(DISABLED_DEMO_DIR, "demo_*.npz")))
    return files


def demo_weight(name: str) -> float:
    return float(CFG.demo_weights.get(name, 1.0))


# --------------------------------------------------------------------------- #
# Behaviour-cloning dataset
# --------------------------------------------------------------------------- #

class DemoDataset:
    """
    All recorded episodes in RAM; stacks and aux vectors are rebuilt on the fly
    so they always match exactly what the live env produces.

    Label alignment: the sample at tick t pairs the frames up to t with the key
    state at t + REACTION_DELAY_STEPS. Your fingers answer what you saw a beat
    earlier, and pairing a frame with the keys held at that same instant teaches
    the network coincidence instead of cause.
    """

    def __init__(self, demo_dir: str = DEMO_DIR, quiet: bool = False):
        files = demo_files()
        if not files:
            raise SystemExit(
                f"No demos found in {demo_dir}.\n"
                "Record some first:  python mosim_env.py record"
            )
        self.episodes = []
        self.names = []
        self.weights = []
        upgraded = 0
        for path in files:
            demo = load_demo(path)
            name = demo["name"]
            weight = demo_weight(name)
            if weight <= 0:
                continue
            if demo["version"] < DEMO_FORMAT_VERSION:
                upgraded += 1
            for note in demo["notes"]:
                if note.startswith(("WARNING", "no camera")) and not quiet:
                    print(f"  {name}: {note}")
            # Precompute camera-swap bookkeeping once, so sampling stays O(1).
            swaps = demo["keys"][:, CAMERA_IDX].astype(np.int64)
            demo["swap_parity"] = (np.cumsum(swaps) % 2).astype(np.int8)
            since = np.empty(len(swaps), dtype=np.int64)
            last = -10 ** 6
            for t, flag in enumerate(swaps):
                if flag:
                    last = t
                since[t] = t - last
            demo["since_swap"] = np.clip(since, 0, 10 ** 6)
            self.episodes.append(demo)
            self.names.append(name)
            self.weights.append(weight)

        if not self.episodes:
            raise SystemExit("Every demo is weighted 0. Run `python mosim_env.py demos` to re-enable some.")

        n = len(self.episodes)
        n_val = 1 if n >= 4 else 0        # hold out one episode once there are enough
        self.train_eps = list(range(n - n_val))
        self.val_eps = list(range(n - n_val, n))

        self.usable = [max(len(e["keys"]) - REACTION_DELAY_STEPS, 0) for e in self.episodes]
        total = sum(self.usable)
        all_keys = np.concatenate([e["keys"] for e in self.episodes], axis=0)
        self.press_rates = all_keys.mean(axis=0)

        if not quiet:
            print(f"Loaded {n} demos, {total} usable steps "
                  f"({n - n_val} train / {n_val} validation).")
            if upgraded:
                print(f"  {upgraded} demo(s) were an older format and were upgraded on load "
                      "(your files on disk are untouched).")
            print("  Your key press rates: "
                  + ", ".join(f"{k}={r:.2f}" for k, r in zip(ACTION_KEYS, self.press_rates)))
            hub_frac = float(np.concatenate([e["hub"] for e in self.episodes]).mean())
            print(f"  Blue hub read as active for {hub_frac * 100:.0f}% of recorded ticks.")

    def positive_weights(self) -> np.ndarray:
        """
        Up-weight the frames where a rare key WAS pressed. Without this, a key
        you press 2% of the time is fitted best by never pressing it, and the
        agent ends up mashing only W/A/S/D.
        """
        p = np.clip(self.press_rates, 1e-3, 1 - 1e-3)
        w = ((1.0 - p) / p) ** BC_BALANCE_POWER
        return np.clip(w, 1.0, BC_MAX_POS_WEIGHT).astype(np.float32)

    def _sample_weights(self, eps):
        """Longer demos contribute more steps; `demos weight` scales that."""
        raw = np.array([self.usable[i] * self.weights[i] for i in eps], dtype=np.float64)
        if raw.sum() <= 0:
            raw = np.ones(len(eps), dtype=np.float64)
        return raw / raw.sum()

    def sample_batch(self, batch_size: int, eps, prev_dropout: float = 0.0,
                     augment: bool = False):
        chosen = np.random.choice(eps, size=batch_size, p=self._sample_weights(eps))

        images = np.empty((batch_size, IMAGE_CHANNELS, FRAME_H, FRAME_W), dtype=np.uint8)
        aux = np.empty((batch_size, AUX_DIM), dtype=np.float32)
        actions = np.empty((batch_size, N_ACTIONS), dtype=np.float32)
        schedule = HubSchedule()

        for b, ep_index in enumerate(chosen):
            ep = self.episodes[ep_index]
            usable = self.usable[ep_index]
            if usable <= 0:
                usable = 1
            t = int(np.random.randint(usable))
            label_t = min(t + REACTION_DELAY_STEPS, len(ep["keys"]) - 1)

            images[b] = ep["planes"][stack_indices(t)].reshape(
                IMAGE_CHANNELS, FRAME_H, FRAME_W)
            actions[b] = ep["keys"][label_t]

            elapsed = float(ep["time_frac"][t]) * schedule.total
            aux[b] = build_aux(
                prev_action=ep["keys"][max(label_t - 1, 0)],
                time_frac=ep["time_frac"][t],
                hub_active=bool(ep["hub"][t]),
                seconds_to_hub_change=schedule.seconds_to_change(elapsed),
                fill=float(ep["fill"][t]),
                view_parity=int(ep["swap_parity"][t]),
                steps_since_swap=int(ep["since_swap"][t]),
            )

        if prev_dropout > 0.0:
            keep = (np.random.rand(batch_size) >= prev_dropout).astype(np.float32)
            aux[:, :N_ACTIONS] *= keep[:, None]

        if augment and BC_BRIGHTNESS_JITTER > 0:
            # Grayscale planes only (channels 0, 2, 4, ...); shifting the fuel
            # mask would change what counts as a ball.
            gray_channels = np.arange(0, IMAGE_CHANNELS, PLANES_PER_FRAME)
            jitter = np.random.uniform(-BC_BRIGHTNESS_JITTER, BC_BRIGHTNESS_JITTER,
                                       size=(batch_size, 1, 1, 1))
            tweaked = images[:, gray_channels].astype(np.int16) + jitter
            images[:, gray_channels] = np.clip(tweaked, 0, 255).astype(np.uint8)

        return images, aux, actions


# --------------------------------------------------------------------------- #
# PPO construction + callbacks
# --------------------------------------------------------------------------- #

def make_ppo(env, finetune: bool):
    PPO, _, _, _, _ = need_sb3()
    torch = need_torch()
    model = PPO(
        policy="MultiInputPolicy",       # CNN for "image", MLP for "aux"
        env=env,
        learning_rate=FINETUNE_LR if finetune else SCRATCH_LR,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=FINETUNE_ENT_COEF if finetune else SCRATCH_ENT_COEF,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device=torch_device(),
        tensorboard_log=TENSORBOARD_DIR,
    )
    # Start the tap-only keys biased off (~5%) instead of 50/50, so a brand-new
    # policy doesn't flicker the camera or the hopper every single step.
    with torch.no_grad():
        model.policy.action_net.bias[CAMERA_IDX] = -3.0
        model.policy.action_net.bias[HOPPER_IDX] = -2.0
    return model


def set_actor_trainable(policy, actor_trainable: bool):
    """False = freeze everything except the value head; True = train everything."""
    for p in policy.parameters():
        p.requires_grad = actor_trainable
    if not actor_trainable:
        for module in (policy.mlp_extractor.value_net, policy.value_net):
            for p in module.parameters():
                p.requires_grad = True


def build_callbacks(vecnorm, warmup: bool):
    _, BaseCallback, _, _, _ = need_sb3()

    class SaveOnEpisodeEnd(BaseCallback):
        """Saves the model and reward-normalization stats once per episode."""

        def __init__(self):
            super().__init__(verbose=1)
            self._episodes = 0

        def _on_step(self) -> bool:
            dones = self.locals.get("dones")
            if dones is not None and bool(dones[0]):
                self._episodes += 1
                out = os.path.join(CHECKPOINT_DIR, f"mosim_ppo_episode_{self._episodes}")
                self.model.save(out)
                vecnorm.save(VECNORM_PATH)
                print(f"[checkpoint] saved after episode {self._episodes}: {out}.zip")
                # Keep only the newest few; a long unattended run would
                # otherwise fill the folder with hundreds of zips.
                saved = sorted(glob.glob(os.path.join(
                    CHECKPOINT_DIR, "mosim_ppo_episode_*.zip")), key=os.path.getmtime)
                for stale in saved[:-KEEP_CHECKPOINTS]:
                    try:
                        os.remove(stale)
                    except OSError:
                        pass
            return True

    class ReleaseKeysOnRolloutEnd(BaseCallback):
        """
        The game keeps running while PPO computes gradients. Without this,
        whatever keys were held at the end of a rollout stay held for the whole
        update and the robot drives into a wall. The env separately discounts
        that time from the match clock.
        """

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            self.training_env.env_method("release_all_keys")

    class ValueWarmup(BaseCallback):
        """Freezes the cloned policy until the value head has learned something."""

        def __init__(self, steps):
            super().__init__(verbose=1)
            self.steps = steps
            self._end = None
            self._frozen = False

        def _on_training_start(self) -> None:
            self._end = self.model.num_timesteps + self.steps
            set_actor_trainable(self.model.policy, False)
            self._frozen = True
            print(f"[warmup] policy frozen for {self.steps} steps; training the value head only")

        def _on_step(self) -> bool:
            if self._frozen and self.model.num_timesteps >= self._end:
                set_actor_trainable(self.model.policy, True)
                self._frozen = False
                print("[warmup] done - policy unfrozen, full PPO fine-tuning from here")
            return True

    callbacks = [SaveOnEpisodeEnd(), ReleaseKeysOnRolloutEnd()]
    if warmup:
        callbacks.append(ValueWarmup(VALUE_WARMUP_STEPS))
    return callbacks


# --------------------------------------------------------------------------- #
# Behaviour cloning (pretraining)
# --------------------------------------------------------------------------- #

def pretrain(args):
    """
    Trains the exact PPO network on your demos by maximizing the likelihood of
    your keypresses, with three guards against the network degenerating into
    "mash whatever is pressed most":
      * rare keys are up-weighted where they WERE pressed,
      * the previous-action input is dropped 60% of the time, so it can't just
        repeat itself instead of looking at the screen,
      * the report at the end shows per-key precision and recall against an
        "always off" baseline, which makes a lazy fit obvious.
    """
    torch = need_torch()
    device = torch_device()
    dataset = DemoDataset()

    _, _, DummyVecEnv, _, _ = need_sb3()
    model = make_ppo(DummyVecEnv([lambda: SpacesOnlyEnv()]), finetune=True)
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=BC_LR)

    iterations = int(args.iterations)
    pos_w = torch.as_tensor(dataset.positive_weights(), device=device)
    print("  Rare-key up-weighting: "
          + ", ".join(f"{k}x{w:.1f}" for k, w in zip(ACTION_KEYS, pos_w.cpu().numpy())))

    def to_obs(images, aux):
        return {"image": torch.as_tensor(images, device=device),
                "aux": torch.as_tensor(aux, device=device)}

    def batch_loss(images, aux, acts_np):
        acts = torch.as_tensor(acts_np, device=device)
        logp = policy.get_distribution(to_obs(images, aux)).distribution.log_prob(acts)
        weights = 1.0 + (pos_w - 1.0) * acts      # only positives are up-weighted
        return -(logp * weights).sum(dim=1).mean(), acts

    val_batch = dataset.sample_batch(512, dataset.val_eps) if dataset.val_eps else None

    def validate():
        policy.set_training_mode(False)
        with torch.no_grad():
            loss, acts = batch_loss(*val_batch)
            probs = policy.get_distribution(
                to_obs(val_batch[0], val_batch[1])).distribution.probs
        policy.set_training_mode(True)
        return float(loss), probs.cpu().numpy(), acts.cpu().numpy()

    best_loss, best_state, stale = float("inf"), None, 0
    policy.set_training_mode(True)
    for it in range(1, iterations + 1):
        # Cosine decay: big steps early, fine adjustments late.
        frac = it / iterations
        lr = BC_LR * (BC_MIN_LR_FRACTION + (1 - BC_MIN_LR_FRACTION)
                      * 0.5 * (1 + math.cos(math.pi * frac)))
        for group in optimizer.param_groups:
            group["lr"] = lr

        images, aux, acts_np = dataset.sample_batch(
            BC_BATCH_SIZE, dataset.train_eps,
            prev_dropout=BC_PREV_ACTION_DROPOUT, augment=True,
        )
        loss, _ = batch_loss(images, aux, acts_np)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

        if it % BC_EVAL_EVERY == 0 or it == 1:
            msg = f"[bc] {it}/{iterations}  train {loss.item():.3f}"
            if val_batch is not None:
                val_loss, _, _ = validate()
                msg += f"  val {val_loss:.3f}"
                if val_loss < best_loss - 1e-4:
                    best_loss, stale = val_loss, 0
                    best_state = {k: v.detach().clone()
                                  for k, v in policy.state_dict().items()}
                else:
                    stale += 1
                    if stale >= 6:
                        print(f"{msg}\n[bc] validation stopped improving - keeping the best network.")
                        break
            print(msg)

    if best_state is not None:
        policy.load_state_dict(best_state)
        print(f"[bc] restored the best network (val loss {best_loss:.3f}).")

    # ---- Report: is it actually learning, or just mashing? ----
    if val_batch is not None:
        _, probs, acts = validate()
        pred = probs > 0.5
        print("\nPer-key report on held-out gameplay")
        print("  key     you press   agent presses   precision  recall   vs always-off")
        for i, key in enumerate(ACTION_KEYS):
            truth, guess = acts[:, i] > 0.5, pred[:, i]
            tp = float(np.sum(truth & guess))
            precision = tp / max(float(guess.sum()), 1.0)
            recall = tp / max(float(truth.sum()), 1.0)
            baseline = max(truth.mean(), 1 - truth.mean())
            acc = float((guess == truth).mean())
            print(f"  {key:<6} {truth.mean():9.2f} {guess.mean():15.2f} "
                  f"{precision:10.2f} {recall:7.2f}   {acc:.2f} vs {baseline:.2f}")
        weak = [ACTION_KEYS[i] for i in range(N_ACTIONS)
                if (acts[:, i] > 0.5).mean() > 0.02 and pred[:, i].mean() < 0.2 * (acts[:, i] > 0.5).mean()]
        if weak:
            print(f"\n  Barely predicted: {', '.join(weak)}. Raise BC_MAX_POS_WEIGHT, or record "
                  "demos where those keys matter more.")
        else:
            print("\n  Every key you use regularly is being predicted at a sane rate.")

    os.makedirs(os.path.dirname(BC_INIT_PATH) or ".", exist_ok=True)
    model.save(BC_INIT_PATH)
    print(f"\nSaved the pretrained network to {BC_INIT_PATH}.zip")
    if glob.glob(os.path.join(CHECKPOINT_DIR, "*.zip")) or os.path.exists(FINAL_MODEL_PATH):
        print("You have existing PPO checkpoints, and `train` prefers those over this "
              "network. Run `python mosim_env.py reset` first if you want a fresh start.")
    else:
        print("Next:  python mosim_env.py train")


# --------------------------------------------------------------------------- #
# Training (PPO)
# --------------------------------------------------------------------------- #

def build_training_env(debug_options=None):
    _, _, DummyVecEnv, VecMonitor, VecNormalize = need_sb3()
    # VecMonitor goes INSIDE VecNormalize so logged episode rewards are the raw
    # ones. Screen capture can't be parallelized on one machine, so exactly one
    # environment instance.
    base = VecMonitor(DummyVecEnv([lambda: MoSimEnv(debug_options=debug_options)]))
    try:
        if os.path.exists(VECNORM_PATH):
            env = VecNormalize.load(VECNORM_PATH, base)
            env.training = True
            env.norm_reward = True
        else:
            # Reward normalization keeps PPO stable when one score event (x10)
            # dwarfs the per-step penalties. Images are not normalized.
            env = VecNormalize(base, norm_obs=False, norm_reward=True,
                               clip_reward=10.0, gamma=GAMMA)
    except BaseException:
        base.close()
        raise
    return env


def load_model_or_explain(PPO, path, env):
    """PPO.load, but with a readable message when the saved net predates v4."""
    try:
        return PPO.load(path, env=env, device=torch_device())
    except (ValueError, RuntimeError, KeyError) as exc:
        raise SystemExit(
            f"Couldn't load {os.path.basename(path)}:\n  {exc}\n\n"
            "This is almost always a network saved before the observation changed.\n"
            "Your demos are unaffected. Start the networks over with:\n"
            "  python mosim_env.py reset --all\n"
            "  python mosim_env.py pretrain\n"
            "  python mosim_env.py train"
        ) from exc


def train(args):
    PPO, _, _, _, _ = need_sb3()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print_startup_banner()
    env = build_training_env(debug_options_from(args))
    model = None
    try:
        candidates = glob.glob(os.path.join(CHECKPOINT_DIR, "mosim_ppo_episode_*.zip"))
        if os.path.exists(FINAL_MODEL_PATH):
            candidates.append(FINAL_MODEL_PATH)

        warmup = False
        if candidates and not args.fresh:
            latest = max(candidates, key=os.path.getmtime)
            print(f"Resuming from {os.path.basename(latest)}")
            model = load_model_or_explain(PPO, latest, env)
        elif os.path.exists(BC_INIT_PATH + ".zip") and not args.fresh:
            print(f"Starting from the behaviour-cloned network "
                  f"({os.path.basename(BC_INIT_PATH)}.zip)")
            model = load_model_or_explain(PPO, BC_INIT_PATH + ".zip", env)
            warmup = True
        else:
            print("Starting a brand-new model. (Pretraining on your demos first learns "
                  "far faster: python mosim_env.py pretrain)")
            model = make_ppo(env, finetune=False)

        print("\nTraining. [ or Ctrl+C stops cleanly and saves.\n")
        model.learn(
            total_timesteps=int(args.steps),
            callback=build_callbacks(env, warmup),
            reset_num_timesteps=False,  # keep the counter continuous across resumes
        )
    except KeyboardInterrupt:
        print("\nInterrupted - saving.")
    finally:
        # Release keys before potentially slow disk writes, including on errors.
        env.env_method("release_all_keys")
        try:
            if model is not None:
                model.save(FINAL_MODEL_PATH.replace(".zip", ""))
                env.save(VECNORM_PATH)
                print(f"Saved {os.path.basename(FINAL_MODEL_PATH)}")
        finally:
            env.close()


# --------------------------------------------------------------------------- #
# Watching the agent play
# --------------------------------------------------------------------------- #

def play(args):
    """
    Runs the trained policy with the full reward function live, so you can see
    what it's being paid for while it plays.

    By default it takes the most likely action instead of sampling, and cancels
    contradictory keys. Sampling from independent per-key probabilities is
    exactly how an agent ends up blending two different routes into one bad
    one; committing to the most likely action avoids that. `--sample` turns the
    stochastic behaviour back on if you want to see the spread.
    """
    PPO, _, _, _, _ = need_sb3()
    torch = need_torch()

    path = args.model
    if path is None:
        candidates = glob.glob(os.path.join(CHECKPOINT_DIR, "mosim_ppo_episode_*.zip"))
        if os.path.exists(FINAL_MODEL_PATH):
            candidates.append(FINAL_MODEL_PATH)
        if not candidates:
            raise SystemExit("No trained model found. Run `pretrain` then `train` first.")
        path = max(candidates, key=os.path.getmtime)
    print(f"Playing with {os.path.basename(path)}")

    model = load_model_or_explain(PPO, path, None)
    if args.dry_run:
        print("DRY RUN: proposed keys are displayed; no keys or match resets are sent.")
    env = MoSimEnv(verbose=False, debug_options=debug_options_from(args), dry_run=args.dry_run)
    try:
        for episode in range(1, int(args.episodes) + 1):
            obs, _ = env.reset()
            total, steps = 0.0, 0
            while True:
                obs_tensor, _ = model.policy.obs_to_tensor(obs)
                with torch.no_grad():
                    dist = model.policy.get_distribution(obs_tensor)
                    probs = dist.distribution.probs.cpu().numpy()[0]
                action = (np.random.rand(N_ACTIONS) < probs).astype(np.int8) if args.sample \
                    else (probs > 0.5).astype(np.int8)
                if not args.raw:
                    action = resolve_conflicts(action, probs)

                obs, reward, _, truncated, info = env.step(action)
                total += reward
                steps += 1
                if steps % CONTROL_HZ == 0:
                    held = "".join(c if action[i] else "."
                                   for i, c in enumerate(KEY_GLYPHS))
                    print(f"\r  {steps // CONTROL_HZ:3d}s  keys {held}  "
                          f"hub {'ON ' if info['hub_active'] else 'off'}({info['hub_source'][:3]})  "
                          f"fill {info['fill']:.2f}  reward {total:+7.1f}  "
                          f"pts {env.episode_points}", end="", flush=True)
                if truncated:
                    break
            print(f"\n[episode {episode}] {env.episode_points} points, reward {total:+.1f}")
            print("  " + "  ".join(f"{k}={v:+.1f}" for k, v in env.breakdown.items()))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        env.close()


def watch(args):
    """Observe human play with the same reward code, without injecting ANY keys."""
    print("WATCH: play yourself. Focus MoSim, restart with R, then press ] to start.")
    print("R clears the reward ledger for a new match; ] pauses; [ stops.")
    print("Score reward tracks BLUE ALLIANCE points, including teammates.")
    env = MoSimEnv(verbose=False, debug_options=debug_options_from(args), inject_inputs=False)
    try:
        env.reset()
        import ctypes
        user = ctypes.windll.user32
        previous_reset = bool(user.GetAsyncKeyState(0x52) & 0x8000)
        while True:
            env._wait_ready()
            reset_down = bool(user.GetAsyncKeyState(0x52) & 0x8000)
            if reset_down and not previous_reset:
                # The human sends R to the game. This command only clears telemetry.
                time.sleep(0.5)
                env.reset()
            previous_reset = reset_down
            _, _, _, truncated, _ = env.step(read_keys_now())
            if truncated:
                print(f"Match complete: {env.episode_points} alliance points, "
                      f"raw return {env._ledger.total:+.3f}. Press R, then ] for another.")
                with env._guard.input_lock:
                    env._guard._pause("MATCH COMPLETE: restart with R, then press ]")
                env._wait_ready()
                env.reset()
    except KeyboardInterrupt:
        print("\nWatch stopped. Reward log saved.")
    finally:
        env.close()


def debug_preview(args):
    """Offline UI preview. Reward values are explicitly synthetic, not measurements."""
    from dataclasses import asdict
    from mosim_overlay import save_preview
    from mosim_vision import ObjectDetector
    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"Could not read {args.image}")
    ledger = RewardLedger()
    ledger.reset()
    for tick in range(1, 121):
        parts = {name: 0.0 for name in REWARD_LABELS}
        parts.update(time=-2/163/10, shaping=-0.008)
        if tick == 100:
            parts["score"] = 10.0
        if tick == 120:
            parts.update(key_change=-0.02)
        ledger.record(parts, tick/10, {"score": "Example: alliance gained 1 point",
                                      "shaping": "Example: discounted fuel / hub potential",
                                      "time": "Example: fixed tick time cost",
                                      "key_change": "Example: two charged key changes"})
    state = {"ledger": ledger.snapshot(), "elapsed": 12.0, "duration": 163,
             "phase": "preview", "points": "--", "hub_active": True,
             "hub_source": "example", "fill_normalized": normalized_fill(hopper_fill_fraction(frame)),
             "motion": 0.0, "motion_threshold": CFG.motion_diff_threshold, "keys": ["w", "shift"],
             "regions": [{"box": CFG.region(name), "label": label} for name, label in (
                 ("score", "Score ROI (OCR)"), ("hopper", "Hopper ROI (color estimate)"),
                 ("hub", "Hub indicator ROI"))]}
    # Use only the zero-label backend for an offline preview; do not discover
    # optional weights or initialize model downloads as a side effect.
    detector = ObjectDetector(CFG, discover_weights=False)
    captured = time.perf_counter()
    started = time.perf_counter()
    boxes = detector.detect(frame)
    vision = {"boxes": [asdict(b) for b in boxes], "captured_at": captured,
              "ms": (time.perf_counter()-started)*1000, "max_age": 0.25,
              "status": detector.status}
    save_preview(args.image, args.output, state, vision, args.corner)
    print(f"Wrote {os.path.abspath(args.output)} (synthetic reward values)")


# --------------------------------------------------------------------------- #
# Recording your own gameplay
# --------------------------------------------------------------------------- #

RECORD_TIPS = """
Recording tips
  * Play the way you want the agent to play. Smooth, decisive inputs get
    copied; so does jerky panic.
  * 10-20 clean matches is a solid start.
  * BE CONSISTENT. The network learns one probability per key, so if half your
    demos go left at a given moment and half go right, it learns "maybe" and
    plays a blend of both, which is usually worse than either. Demos that agree
    with each other are worth far more than demos that merely differ.
  * Keep the same window size, camera setup and HUD layout you'll train with.
  * Every key is recorded, including quick K and H taps between samples.
  * ESC (or Ctrl+C) ends an episode early and keeps what you have.
"""


def record(args):
    """Records matches of YOU playing at the same rate and preprocessing the agent uses."""
    os.makedirs(DEMO_DIR, exist_ok=True)
    print_startup_banner()
    print(RECORD_TIPS)
    grabber = Grabber()
    schedule = HubSchedule()
    tracker = HubTracker()
    duration = schedule.total

    try:
        while True:
            reply = input("\nENTER to record a match, or q + ENTER to finish: ")
            if reply.strip().lower().startswith("q"):
                break

            print("Click the game window now...")
            for n in range(5, 0, -1):
                print(f"  starting in {n}...")
                time.sleep(1)

            tap_key(RESET_KEY)   # the same restart the agent does
            time.sleep(0.5)
            tracker.reset()

            planes_list, keys_list, fracs, hubs, fills = [], [], [], [], []
            start = time.time()
            next_tick = time.perf_counter()
            print(f"RECORDING {duration:.0f}s - play! (ESC ends early and keeps what you have)")
            try:
                while True:
                    elapsed = time.time() - start
                    if elapsed >= duration or escape_pressed():
                        break
                    sleep_for = next_tick - time.perf_counter()
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    next_tick += STEP_PERIOD

                    frame = grabber.grab()
                    hub_active, _ = tracker.update(frame, elapsed)
                    planes_list.append(preprocess_planes(frame))
                    keys_list.append(read_keys_now())
                    fracs.append(min(elapsed / duration, 1.0))
                    hubs.append(1 if hub_active else 0)
                    fills.append(hopper_fill_fraction(frame))
            except KeyboardInterrupt:
                print("\nStopped early - keeping what was recorded.")

            if len(planes_list) < 20:
                print("Too short - discarded.")
                continue
            keys = np.stack(keys_list)
            if not keys.any():
                print("No keypresses detected - discarded. (Was the game window focused?)")
                continue

            path = os.path.join(DEMO_DIR, f"demo_{datetime.now():%Y%m%d_%H%M%S}.npz")
            save_demo(path, planes_list, keys, fracs, hubs, fills)
            rates = keys.mean(axis=0)
            print(f"Saved {len(keys)} steps -> {os.path.basename(path)}")
            print("  key rates: " + ", ".join(f"{k}={r:.2f}" for k, r in zip(ACTION_KEYS, rates)))
    except KeyboardInterrupt:
        print("\nDone recording.")
    finally:
        grabber.release()
        count = len(demo_files())
        print(f"\n{count} demo(s) on file. Next:  python mosim_env.py pretrain")


# --------------------------------------------------------------------------- #
# Key test
# --------------------------------------------------------------------------- #

def testkeys(args):
    """Taps a key at three hold lengths so you can see whether the game reacts."""
    pdi = need_pydirectinput()
    key = args.key
    if key not in VK_CODES:
        raise SystemExit(f"Unknown key '{key}'. One of: {', '.join(VK_CODES)}")
    print(f"Click the game window and watch what '{key}' does.")
    for n in range(5, 0, -1):
        print(f"  starting in {n}...")
        time.sleep(1)
    for hold in (0.05, 0.10, 0.20):
        print(f"  tapping {key}, held {hold:.2f}s")
        pdi.keyDown(key)
        time.sleep(hold)
        pdi.keyUp(key)
        time.sleep(3)
    print("\nIf nothing happened at any hold length, it's an input problem, not a "
          "learning one: check the game has focus and that it reads DirectInput.")
    if key == "i":
        print("If only the longer holds worked, set camera_tap_hold_seconds in "
              f"{os.path.basename(CONFIG_PATH)} to that value.")


# --------------------------------------------------------------------------- #
# Guided calibration
# --------------------------------------------------------------------------- #

# Built for a single screen. You can't click into a terminal and press ENTER
# while the game is fullscreen, so `setup` is split in two:
#
#   PHASE 1, in the game, hands-free. Two global hotkeys (they work from any
#   window) plus beeps and a spoken voice drive it. It only CAPTURES: one
#   screenshot, an idle clip, a driving clip, and a few frames of the hub lit
#   and dark.
#
#   PHASE 2, back on the desktop. All the measuring happens here, from what was
#   captured: you drag the boxes with the mouse on the frozen screenshot, and
#   the Tesseract preprocessing is tuned automatically against it.

REFERENCE_SHOT = os.path.join(HERE, "setup_reference.png")

_VK_NAMES = {
    "[": 0xDB, "]": 0xDD, ";": 0xBA, "'": 0xDE, ",": 0xBC, ".": 0xBE, "/": 0xBF,
    "\\": 0xDC, "-": 0xBD, "=": 0xBB, "`": 0xC0,
    "tab": 0x09, "backspace": 0x08, "enter": 0x0D,
}
_VK_NAMES.update({f"f{n}": 0x6F + n for n in range(1, 13)})
_VK_NAMES.update({c: ord(c.upper()) for c in "abcdefghijklmnopqrstuvwxyz0123456789"})
_SPOKEN = {"[": "left bracket", "]": "right bracket", ";": "semicolon", "'": "apostrophe",
           ",": "comma", ".": "period", "/": "slash", "\\": "backslash", "-": "minus",
           "=": "equals", "`": "backtick"}

SETUP_INTRO = """
Calibration wizard - built for a single screen
  You do NOT need the terminal while the game is fullscreen.

  PHASE 1 (in the game). Two hotkeys work from any window, and sounds plus a
  spoken voice tell you what to do:
      {go}   = go / capture now          {skip}   = skip this step
    1 beep             got your keypress
    3 ticks, long beep measuring right now - do what the voice said
    2 beeps            step finished; the voice explains the next one
    3 rising beeps     everything captured - come back to this window
  Phase 1 only CAPTURES (a screenshot, an idle clip, a driving clip, and a
  few frames of the hub lit and dark).

  PHASE 2 (back on the desktop). The measuring happens here: you drag boxes
  on the frozen screenshot with the mouse, and the Tesseract settings are
  tuned automatically against it.

  Redo just part of it any time, without going back into the game:
    python mosim_env.py setup --only regions,ocr --image setup_reference.png
"""


def _setup_keys():
    go, skip = str(CFG.setup_go_key).lower(), str(CFG.setup_skip_key).lower()
    for name in (go, skip):
        if name not in _VK_NAMES:
            raise SystemExit(f"Setup key '{name}' isn't supported. Edit setup_go_key / "
                             f"setup_skip_key in {os.path.basename(CONFIG_PATH)}. Options: "
                             + ", ".join(sorted(_VK_NAMES)))
    if go == skip:
        raise SystemExit("setup_go_key and setup_skip_key must be different keys.")
    return go, skip


def _key_is_down(name: str) -> bool:
    import ctypes
    return bool(ctypes.windll.user32.GetAsyncKeyState(_VK_NAMES[name]) & 0x8000)


def _beep(freq: int, ms: int):
    try:
        import winsound
        winsound.Beep(int(freq), int(ms))
    except Exception:
        time.sleep(ms / 1000.0)


def cue(kind: str):
    """Sounds, because you can't read the terminal while the game is fullscreen."""
    if kind == "ack":           # your keypress registered
        _beep(1000, 120)
    elif kind == "skip":
        _beep(400, 350)
    elif kind == "tick":        # countdown
        _beep(700, 90)
    elif kind == "measure":     # measuring starts NOW
        _beep(1300, 300)
    elif kind == "done":        # this step is finished
        _beep(1000, 100)
        time.sleep(0.05)
        _beep(1400, 100)
    elif kind == "all_done":
        for freq in (800, 1100, 1500):
            _beep(freq, 160)


_SPEECH = None


def say(text: str, wait: bool = False):
    """
    Print the prompt and speak it through Windows' built-in voice (nothing to
    install). With wait=False it plays in the background, so you can press the
    hotkey before it finishes talking.
    """
    global _SPEECH
    print(f"  >> {text}", flush=True)
    if not CFG.setup_voice:
        return
    try:
        if _SPEECH is not None and _SPEECH.poll() is None:
            _SPEECH.wait(timeout=30)
        clean = text.replace("'", "").replace('"', "")
        script = ("Add-Type -AssemblyName System.Speech; "
                  "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                  f"$s.Speak('{clean}')")
        _SPEECH = subprocess.Popen(
            ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", script],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=0x08000000)   # CREATE_NO_WINDOW
        if wait:
            _SPEECH.wait(timeout=30)
    except Exception:
        pass


def wait_hotkey(go: str, skip: str, announce: str = None) -> bool:
    """Blocks until the go key (True) or the skip key (False), from any window."""
    if announce:
        say(announce)
    while _key_is_down(go) or _key_is_down(skip):    # ignore a key still held from before
        time.sleep(0.02)
    while True:
        if _key_is_down(go):
            cue("ack")
            return True
        if _key_is_down(skip):
            cue("skip")
            return False
        time.sleep(0.02)


def _hotkey_selftest(go: str, seconds: float = 20.0) -> bool:
    print(f"\n  Hotkey check: press  {go}  once (any window) within {seconds:.0f}s ...", flush=True)
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        if _key_is_down(go):
            cue("ack")
            print("  Hotkeys work.")
            return True
        time.sleep(0.02)
    return False


def _tick_countdown(n: int = 3, gap: float = 0.45):
    for _ in range(n):
        cue("tick")
        time.sleep(gap)


def _grab_burst(grabber, n: int, gap: float):
    frames = []
    for _ in range(n):
        frames.append(grabber.grab().copy())
        time.sleep(gap)
    return frames


def _draw_boxes(frame):
    annotated = frame.copy()
    h, w = frame.shape[:2]
    for name, key, color in (("SCORE", "score", (255, 0, 0)),
                             ("HOPPER", "hopper", (0, 255, 0)),
                             ("HUB", "hub", (0, 165, 255))):
        left, top, right, bottom = CFG.region(key)
        p1, p2 = (int(left * w), int(top * h)), (int(right * w), int(bottom * h))
        cv2.rectangle(annotated, p1, p2, color, 2)
        cv2.putText(annotated, name, (p1[0], max(p1[1] - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return annotated


def _ask_region(name, current):
    """Typed fallback, only used if OpenCV can't open a picture window."""
    print(f"\n  Current {name} box (left, top, right, bottom as fractions of the screen):")
    print(f"    {tuple(round(v, 4) for v in current)}")
    reply = input("  ENTER to keep it, or type four numbers separated by spaces: ").strip()
    if not reply:
        return list(current)
    try:
        values = [float(x) for x in reply.replace(",", " ").split()]
        if len(values) != 4 or not all(0.0 <= v <= 1.0 for v in values):
            raise ValueError
        if values[0] >= values[2] or values[1] >= values[3]:
            raise ValueError
        return values
    except ValueError:
        print("  Need four numbers between 0 and 1, left < right and top < bottom. Keeping the old box.")
        return list(current)


def _measure_motion(grabber, seconds):
    diffs, prev = [], None
    end = time.perf_counter() + seconds
    next_tick = time.perf_counter()
    while time.perf_counter() < end:
        sleep_for = next_tick - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)
        next_tick += STEP_PERIOD
        gray = preprocess_planes(grabber.grab())[0].astype(np.int16)
        if prev is not None:
            diffs.append(float(np.mean(np.abs(gray - prev))))
        prev = gray
    return np.array(diffs) if diffs else np.zeros(1)


# ---------------------------- Phase 1: capture ----------------------------- #

def capture_session(grabber, go, skip, want_ref, want_motion, want_hub):
    """
    Hands-free capture. Returns a dict with whatever was captured (None for
    anything skipped): reference (a full frame), idle / moving (motion
    readings), hub_on / hub_off (a few full frames each).
    """
    g, s = _SPOKEN.get(go, go), _SPOKEN.get(skip, skip)
    out = {"reference": None, "idle": None, "moving": None, "hub_on": None, "hub_off": None}
    count = [0]

    def step_name():
        count[0] += 1
        return f"Step {count[0]}"

    say("Setup started. Switch to the game now.", wait=True)

    if want_ref:
        if wait_hotkey(go, skip,
                       f"{step_name()}, screenshot. Get at least one point on the score, "
                       f"show the hub indicator, and load some fuel into the hopper. "
                       f"Press {g} when it looks right. Press {s} to skip."):
            out["reference"] = grabber.grab().copy()
            cue("done")

    if want_motion:
        if wait_hotkey(go, skip,
                       f"{step_name()}, standing still. Let go of every key, then press {g}. "
                       f"You will hear three ticks and a long beep, then four seconds of stillness."):
            _tick_countdown()
            cue("measure")
            out["idle"] = _measure_motion(grabber, 4.0)
            cue("done")
            if wait_hotkey(go, skip,
                           f"{step_name()}, driving. Press {g}, then drive around smoothly "
                           f"starting at the long beep. It measures for five seconds."):
                _tick_countdown()
                cue("measure")
                out["moving"] = _measure_motion(grabber, 5.0)
                cue("done")

    if want_hub:
        if wait_hotkey(go, skip,
                       f"{step_name()}, hub active. Press R to start a match if one is not running. "
                       f"When the blue hub is lit, press {g}. Press {s} to skip hub vision "
                       f"and use the match schedule instead."):
            out["hub_on"] = _grab_burst(grabber, 5, 0.2)
            cue("done")
            if wait_hotkey(go, skip,
                           f"{step_name()}, hub inactive. Wait until the blue hub goes dark, then press {g}. "
                           f"If the match ends first, press R to restart and wait again."):
                out["hub_off"] = _grab_burst(grabber, 5, 0.2)
                cue("done")

    cue("all_done")
    say("All captured. Switch back to the desktop. Picture windows will open on top.", wait=True)
    return out


# ---------------------------- Phase 2: measure ----------------------------- #

def check_tesseract():
    """Terminal-only step: locate tesseract.exe and prove it can read digits."""
    print("\nSTEP 0 - Tesseract (needs the terminal only)")
    path = find_tesseract(refresh=True)
    if not path:
        print("  Couldn't find tesseract.exe on PATH or in the usual install folders.")
        print("  Install it from https://github.com/UB-Mannheim/tesseract/wiki")
        reply = input("  Paste the full path to tesseract.exe here, or ENTER to skip: ").strip().strip('"')
        if reply and os.path.isfile(reply):
            CFG.tesseract_cmd = reply
            path = find_tesseract(refresh=True)
        if not path:
            print("  Skipping OCR setup - the score can't be read without Tesseract.")
            return False
    CFG.tesseract_cmd = path
    try:
        pytesseract = need_pytesseract()
        version = pytesseract.get_tesseract_version()
        img = np.full((90, 300), 255, dtype=np.uint8)
        cv2.putText(img, "4827", (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 0, 5, cv2.LINE_AA)
        read = ocr_read(img)
    except Exception as exc:
        print(f"  Found {path}, but it failed to run: {exc}")
        return False
    print(f"  Found {path} (version {version}).")
    if read == 4827:
        print("  Test read of a rendered 4827: OK.")
        return True
    print(f"  Test read of a rendered 4827 gave {read!r}. Tesseract runs but isn't reading digits;")
    print("  reinstall it and make sure the English language data is included.")
    return False


def _pick_region(frame, label, key, color):
    """Drag a box on the frozen screenshot. ENTER accepts, C keeps the old box."""
    h, w = frame.shape[:2]
    scale = min(1.0, 1280.0 / w, 720.0 / h)
    shown = (cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
             if scale < 1.0 else frame.copy())
    dh, dw = shown.shape[:2]
    left, top, right, bottom = CFG.region(key)
    cv2.rectangle(shown, (int(left * dw), int(top * dh)), (int(right * dw), int(bottom * dh)), color, 1)
    title = f"{label}: drag a box around it, ENTER to accept, C to keep the old one"
    try:
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        try:
            cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass
        x, y, bw, bh = cv2.selectROI(title, shown, showCrosshair=False, fromCenter=False)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    except cv2.error as exc:
        print(f"  (Couldn't open a picture window: {exc})")
        return _ask_region(label, CFG.region(key))
    if bw <= 0 or bh <= 0:
        print(f"  {label}: kept the current box.")
        return list(CFG.region(key))
    box = [round(x / dw, 4), round(y / dh, 4), round((x + bw) / dw, 4), round((y + bh) / dh, 4)]
    print(f"  {label}: {box}")
    return box


def _write_calibration_images(frame):
    cv2.imwrite(os.path.join(HERE, "calib_1_boxes.png"), _draw_boxes(frame))
    lo, hi = CFG.hsv("fuel")
    crop = crop_frac(frame, CFG.region("hopper"))
    if crop.size:
        cv2.imwrite(os.path.join(HERE, "calib_3_hopper_mask.png"),
                    cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), lo, hi))
    lo, hi = CFG.hsv("hub")
    crop = crop_frac(frame, CFG.region("hub"))
    if crop.size:
        cv2.imwrite(os.path.join(HERE, "calib_4_hub_mask.png"),
                    cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), lo, hi))
    print("  Wrote calib_1_boxes.png, calib_3_hopper_mask.png and calib_4_hub_mask.png "
          "next to the script.")


def _ocr_grid(crop, scale):
    """Every candidate preprocessing -> what Tesseract reads with it."""
    results = {}
    for invert in (False, True):
        results[("otsu", 0, invert)] = ocr_read(binarize_with(crop, "otsu", 0, invert, scale))
        for cut in range(70, 231, 10):
            results[("fixed", cut, invert)] = ocr_read(binarize_with(crop, "fixed", cut, invert, scale))
    return results


def _choose_ocr_setting(hits):
    """
    Of all the settings that read the score correctly, pick the most robust:
    the MIDDLE of the working range of brightness cut-offs (so a small change
    in lighting doesn't fall off the edge), unless only the automatic mode works.
    """
    fixed = [h for h in hits if h[0] == "fixed"]
    otsu = [h for h in hits if h[0] == "otsu"]
    if len(fixed) >= 3 or not otsu:
        if not fixed:
            return None
        flipped = sum(1 for h in fixed if h[2])
        invert = flipped > len(fixed) - flipped
        cuts = sorted(h[1] for h in fixed if h[2] == invert)
        return ("fixed", cuts[len(cuts) // 2], invert)
    return otsu[0]


def calibrate_ocr(frame):
    """
    Tunes how the score box is preprocessed for Tesseract, using the frozen
    screenshot. Tries 36 variants, asks you once what the score really is,
    then keeps the setting in the middle of the range that reads it correctly.
    """
    crop = crop_frac(frame, CFG.region("score"))
    if crop.size == 0:
        print("  The score box is empty - pick it again with: setup --only regions,ocr --image setup_reference.png")
        return False
    print("  Trying 36 preprocessing variants on your screenshot (a few seconds)...")
    results = _ocr_grid(crop, 3)

    votes = {}
    for value in results.values():
        if value is not None:
            votes[value] = votes.get(value, 0) + 1
    guess = max(votes, key=votes.get) if votes else None
    if guess is not None and votes[guess] >= 3:
        reply = input(f"  Most variants read the score as {guess}. Press ENTER if that's what the "
                      "screenshot shows, or type the real score: ").strip()
        truth = int(reply) if reply.isdigit() else guess
    else:
        reply = input("  Nothing read it reliably. Type the score shown in the screenshot "
                      "(digits only, ENTER to skip): ").strip()
        if not reply.isdigit():
            print("  Skipped. Keeping the current OCR settings.")
            return False
        truth = int(reply)

    choice, hits = None, []
    for scale in (3, 5):
        if scale != 3:
            results = _ocr_grid(crop, scale)
        hits = [v for v, r in results.items() if r == truth]
        choice = _choose_ocr_setting(hits)
        if choice:
            break

    if not choice:
        big = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(HERE, "calib_2_score_crop.png"), big)
        print(f"  No variant read {truth}. Wrote calib_2_score_crop.png - it should show ONLY the")
        print("  digits, tightly boxed, sharp and not covered by anything. Redo the box with:")
        print("    python mosim_env.py setup --only regions,ocr --image setup_reference.png")
        return False

    mode, cut, invert = choice
    CFG.ocr_mode, CFG.ocr_threshold, CFG.ocr_invert = mode, int(cut), bool(invert)
    CFG.ocr_scale = scale
    cv2.imwrite(os.path.join(HERE, "calib_2_score_ocr.png"), binarize_score_crop(crop))
    reading = ocr_score_crop(crop)
    label = "automatic threshold" if mode == "otsu" else f"cut-off {cut}"
    print(f"  -> {label}{', inverted' if invert else ''}, x{scale} enlargement. "
          f"{len(hits)} of {len(results)} variants read {truth}. "
          f"With the chosen one Tesseract reads: {reading}")
    if len(hits) < 4:
        print("  Only a few variants worked, so this is fragile. A sharper screenshot or a tighter score box helps.")
    if truth == 0:
        print("  Note: the score was 0, which is a weak test. Re-run with a nonzero score for a")
        print("  more trustworthy result:  python mosim_env.py setup --only ocr")
    print("  Wrote calib_2_score_ocr.png (what Tesseract is actually given).")
    return True


def calibrate_fill(frame):
    full_fill = hopper_fill_fraction(frame)
    low_fill = float(crop_frac(preprocess_planes(frame)[1], CFG.region("hopper")).mean()) / 255.0
    print(f"  Hopper fill in the screenshot reads {full_fill:.3f} "
          f"(a full hopper should be near {FUEL_HOLD_THRESHOLD_FRACTION}).")
    if low_fill > 1e-4:
        CFG.lowres_fill_gain = round(full_fill / low_fill, 3)
        print(f"  -> lowres_fill_gain = {CFG.lowres_fill_gain} "
              "(maps old demos' fill onto the same scale as live play)")
    else:
        print("  No fuel visible in the hopper box, so the old-demo scale is unchanged. Load fuel "
              "before the screenshot and re-run:  python mosim_env.py setup --only fill")


def calibrate_motion(idle, moving):
    idle_hi = float(np.percentile(idle, 95))
    moving_lo = float(np.percentile(moving, 25))
    print(f"  Idle    : mean {idle.mean():.2f}, 95th pct {idle_hi:.2f}")
    print(f"  Driving : mean {moving.mean():.2f}, 25th pct {moving_lo:.2f}")
    if moving_lo > idle_hi:
        CFG.motion_diff_threshold = round((idle_hi + moving_lo) / 2, 3)
        print(f"  -> motion_diff_threshold = {CFG.motion_diff_threshold}")
    else:
        CFG.motion_diff_threshold = round(idle_hi * 1.2 + 0.05, 3)
        print(f"  Idle and driving overlap. Using {CFG.motion_diff_threshold} (just above idle); "
              "re-run with a clearer driving test if stillness penalties look wrong.")


def calibrate_hub(on_frames, off_frames):
    active_full = float(np.mean([hub_indicator_coverage(f) for f in on_frames]))
    off_full = float(np.mean([hub_indicator_coverage(f) for f in off_frames]))
    active_low = float(np.mean([hub_coverage_from_planes(preprocess_planes(f)) for f in on_frames]))
    off_low = float(np.mean([hub_coverage_from_planes(preprocess_planes(f)) for f in off_frames]))
    cv2.imwrite(os.path.join(HERE, "calib_hub_active.png"), _draw_boxes(on_frames[0]))
    cv2.imwrite(os.path.join(HERE, "calib_hub_inactive.png"), _draw_boxes(off_frames[0]))
    print(f"  Indicator coverage: active {active_full:.4f}, inactive {off_full:.4f}")
    if active_full > off_full * 1.5 and active_full - off_full > 0.01:
        CFG.hub_active_coverage = round((active_full + off_full) / 2, 4)
        CFG.hub_lowres_coverage = round((active_low + off_low) / 2, 4)
        CFG.hub_calibrated = True
        print(f"  -> hub_active_coverage = {CFG.hub_active_coverage} (vision enabled)")
    else:
        CFG.hub_calibrated = False
        print("  Those two readings are too similar to tell apart, so vision stays off and the")
        print("  match schedule is used instead, which is usually fine. To fix it: put the HUB box")
        print("  tightly on the indicator, or widen hub_hsv_lower/upper if it isn't yellow-ish.")
        print("  calib_hub_active.png / calib_hub_inactive.png show what was captured.")


def setup(args):
    """
    Guided, hands-free calibration for a single screen. Phase 1 captures in the
    game using hotkeys, beeps and voice; phase 2 measures at the desktop.
    Everything measured is written to mosim_config.json.
    """
    print_startup_banner()
    steps = ("tesseract", "regions", "ocr", "motion", "hub", "fill")
    only = getattr(args, "only", None)
    wanted = set(steps) if not only else {p.strip().lower() for p in only.split(",") if p.strip()}
    unknown = wanted - set(steps)
    if unknown:
        raise SystemExit(f"Unknown step(s): {', '.join(sorted(unknown))}. Choose from: {', '.join(steps)}")
    go, skip = _setup_keys()
    print(SETUP_INTRO.format(go=go, skip=skip))

    # ---- Step 0: Tesseract (terminal only) ----
    if wanted & {"tesseract", "ocr"}:
        if not check_tesseract():
            wanted.discard("ocr")

    # ---- Where does the reference screenshot come from? ----
    reference = None
    image_path = getattr(args, "image", None)
    if image_path:
        reference = cv2.imread(image_path)
        if reference is None:
            raise SystemExit(f"Couldn't read the image {image_path}")
        print(f"\nUsing {image_path} ({reference.shape[1]}x{reference.shape[0]}) as the screenshot.")
    needs_ref = bool(wanted & {"regions", "ocr", "fill"})
    want_ref_live = needs_ref and reference is None
    want_motion, want_hub = "motion" in wanted, "hub" in wanted

    # ---- Phase 1: hands-free capture in the game ----
    captured = {}
    if want_ref_live or want_motion or want_hub:
        if os.name != "nt":
            raise SystemExit("Phase 1 uses Windows hotkeys. On another OS, save a screenshot yourself "
                             "and run: setup --only regions,ocr,fill --image <screenshot>")
        print("\nPHASE 1 - capture (hands-free)")
        input("  Press ENTER here ONCE. After that you won't need the terminal until the end. ")
        if not _hotkey_selftest(go):
            raise SystemExit(f"\n  '{go}' wasn't detected. Set setup_go_key / setup_skip_key in "
                             f"{os.path.basename(CONFIG_PATH)} to keys that work on your laptop and re-run.")
        print("\n  Switch to the game now. A voice will guide you; the terminal prompts are the same text.")
        grabber = Grabber()
        try:
            captured = capture_session(grabber, go, skip, want_ref_live, want_motion, want_hub)
        finally:
            grabber.release()
        if captured.get("reference") is not None:
            reference = captured["reference"]
            cv2.imwrite(REFERENCE_SHOT, reference)
            print(f"  Saved the screenshot as {os.path.basename(REFERENCE_SHOT)} "
                  f"({reference.shape[1]}x{reference.shape[0]}).")

    # ---- Phase 2: measure from what was captured ----
    print("\nPHASE 2 - measuring (terminal and mouse are fine to use now)")
    if reference is not None:
        if "regions" in wanted:
            print("\nSTEP - regions. A picture window opens for each box. Drag a box, press ENTER.")
            print("  If a window doesn't appear on top of the game, alt-tab to it.")
            print("  SCORE: tight around the blue alliance digits.  HOPPER: inside the robot's ball cavity.")
            print("  HUB: tight around the blue hub indicator. (The thin coloured line is the current box.)")
            for label, key, color in (("SCORE", "score", (255, 0, 0)),
                                      ("HOPPER", "hopper", (0, 255, 0)),
                                      ("HUB", "hub", (0, 165, 255))):
                CFG[f"{key}_region"] = _pick_region(reference, label, key, color)
        _write_calibration_images(reference)
        if "ocr" in wanted:
            print("\nSTEP - score OCR")
            try:
                calibrate_ocr(reference)
            except Exception as exc:
                print(f"  OCR calibration failed: {exc}")
        if "fill" in wanted:
            print("\nSTEP - hopper fill scale")
            calibrate_fill(reference)
    elif needs_ref:
        print("\n  No screenshot, so regions / OCR / hopper fill were skipped. Redo them later with:")
        print("    python mosim_env.py setup --only regions,ocr,fill")

    if want_motion:
        print("\nSTEP - motion threshold")
        if captured.get("idle") is not None and captured.get("moving") is not None:
            calibrate_motion(captured["idle"], captured["moving"])
        else:
            print("  Skipped (no idle + driving readings). Redo with: python mosim_env.py setup --only motion")

    if want_hub:
        print("\nSTEP - hub indicator")
        if captured.get("hub_on") and captured.get("hub_off"):
            calibrate_hub(captured["hub_on"], captured["hub_off"])
        else:
            print("  Skipped, so hub vision is unchanged and the match schedule covers it. "
                  "Redo with: python mosim_env.py setup --only hub")

    CFG.save()
    print("\nCalibration done. Next:  python mosim_env.py record")


# --------------------------------------------------------------------------- #
# Demo management
# --------------------------------------------------------------------------- #

def demos(args):
    """
    List, weight and disable recordings.

    Weighting is the practical answer to "different runs disagree with each
    other". Rather than deleting a demo, turn down the ones you played badly so
    cloning leans on the runs you'd actually want copied.
    """
    os.makedirs(DEMO_DIR, exist_ok=True)
    files = demo_files()
    disabled = sorted(glob.glob(os.path.join(DISABLED_DEMO_DIR, "demo_*.npz")))

    action = (args.action or "list").lower()

    if action == "list":
        if not files and not disabled:
            print(f"No demos yet in {DEMO_DIR}. Run:  python mosim_env.py record")
            return
        print(f"{'#':>3}  {'file':<28} {'steps':>6} {'weight':>6}  {'hub on':>6}  keys")
        for i, path in enumerate(files):
            demo = load_demo(path)
            rates = demo["keys"].mean(axis=0)
            top = ", ".join(f"{k}={r:.2f}" for k, r in zip(ACTION_KEYS, rates) if r >= 0.02)
            swaps = int(demo["keys"][:, CAMERA_IDX].sum())
            print(f"{i:>3}  {demo['name']:<28} {len(demo['keys']):>6} "
                  f"{demo_weight(demo['name']):>6.2f}  {demo['hub'].mean() * 100:>5.0f}%  {top}")
            print(f"     v{demo['version']}, {swaps} camera swaps"
                  + (f" | {'; '.join(demo['notes'])}" if demo["notes"] else ""))
        for path in disabled:
            print(f"  -  {os.path.basename(path):<28} {'':>6} {'off':>6}  (disabled)")
        print("\n  python mosim_env.py demos weight <#> <0-3>   lean on your better runs")
        print("  python mosim_env.py demos disable <#>         move one out of training")
        print("  python mosim_env.py demos enable <name>       bring it back")
        return

    if action == "weight":
        if len(args.rest) != 2:
            raise SystemExit("Usage: demos weight <#> <weight>   (0 disables, 1 is normal)")
        index, weight = int(args.rest[0]), float(args.rest[1])
        if not 0 <= index < len(files):
            raise SystemExit(f"No demo #{index}. Run `demos list` to see the numbers.")
        name = os.path.basename(files[index])
        CFG.demo_weights[name] = weight
        CFG.save()
        print(f"{name} weight set to {weight}")
        return

    if action == "disable":
        if len(args.rest) != 1:
            raise SystemExit("Usage: demos disable <#>")
        os.makedirs(DISABLED_DEMO_DIR, exist_ok=True)
        index = int(args.rest[0])
        if not 0 <= index < len(files):
            raise SystemExit(f"No demo #{index}. Run `demos list` to see the numbers.")
        path = files[index]
        shutil.move(path, os.path.join(DISABLED_DEMO_DIR, os.path.basename(path)))
        print(f"Moved {os.path.basename(path)} to {DISABLED_DEMO_DIR} (not deleted).")
        return

    if action == "enable":
        if len(args.rest) != 1:
            raise SystemExit("Usage: demos enable <filename>")
        src = os.path.join(DISABLED_DEMO_DIR, args.rest[0])
        if not os.path.exists(src):
            raise SystemExit(f"{src} not found.")
        shutil.move(src, os.path.join(DEMO_DIR, args.rest[0]))
        print(f"Restored {args.rest[0]}")
        return

    raise SystemExit(f"Unknown action '{action}'. Use list, weight, disable or enable.")


# --------------------------------------------------------------------------- #
# Doctor and reset
# --------------------------------------------------------------------------- #

def doctor(args):
    """Checks installs, paths, config and hardware, and says what to do next."""
    print_startup_banner()
    print("Python packages")
    for module, why in (("numpy", "arrays"), ("cv2", "vision"), ("gymnasium", "env API"),
                        ("torch", "neural nets"), ("stable_baselines3", "PPO"),
                        ("bettercam", "screen capture"), ("pydirectinput", "key injection"),
                        ("pytesseract", "score OCR")):
        try:
            __import__(module)
            print(f"  ok      {module:<20} {why}")
        except Exception as exc:
            print(f"  MISSING {module:<20} {why}   ({type(exc).__name__})")

    try:
        torch = need_torch()
        cuda = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda else "CPU only (training will be slow)"
        print(f"\nCompute: {name}")
    except Exception:
        print("\nCompute: torch not importable")

    print(f"\nConfig: {CONFIG_PATH}"
          + ("" if os.path.exists(CONFIG_PATH) else "  (not written yet - run `setup`)"))
    print(f"  hub vision: {'calibrated' if CFG.hub_calibrated else 'NOT calibrated (schedule will be used)'}")
    print(f"  motion threshold: {CFG.motion_diff_threshold}")
    schedule = HubSchedule()
    print(f"  match length from the phase schedule: {schedule.total:.0f}s")

    try:
        pytesseract = need_pytesseract()
        print(f"  tesseract: {pytesseract.get_tesseract_version()}")
    except Exception as exc:
        print(f"  tesseract: NOT working ({exc})")

    n_demos = len(demo_files())
    print(f"\nDemos: {n_demos} in {DEMO_DIR}")
    print(f"Pretrained net: {'yes' if os.path.exists(BC_INIT_PATH + '.zip') else 'no'}")
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, '*.zip'))
    print(f"PPO checkpoints: {len(checkpoints)}")

    print("\nSuggested next step:")
    if not os.path.exists(CONFIG_PATH):
        print("  python mosim_env.py setup")
    elif n_demos == 0:
        print("  python mosim_env.py record")
    elif not os.path.exists(BC_INIT_PATH + ".zip"):
        print("  python mosim_env.py pretrain")
    else:
        print("  python mosim_env.py train")


def reset(args):
    """Deletes networks and checkpoints. Never touches your demos or config."""
    targets = [p for p in (CHECKPOINT_DIR, FINAL_MODEL_PATH, VECNORM_PATH,
                           BC_INIT_PATH + ".zip") if os.path.exists(p)]
    if not args.all:
        targets = [p for p in targets if p != BC_INIT_PATH + ".zip"]

    if not targets:
        print("Nothing to delete.")
        return
    print("This will delete:")
    for path in targets:
        print(f"  {path}")
    print(f"\nYour demos in {DEMO_DIR} and {os.path.basename(CONFIG_PATH)} are NOT touched.")
    if input("Type yes to confirm: ").strip().lower() != "yes":
        print("Cancelled.")
        return
    for path in targets:
        shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
        print(f"  removed {os.path.basename(path)}")
    print("\nNext:  python mosim_env.py pretrain")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def print_startup_banner():
    schedule = HubSchedule()
    hub = "vision + schedule" if CFG.hub_calibrated else "schedule only (run `setup` for vision)"
    print(f"MoSim RL v4 | {CONTROL_HZ} Hz | {schedule.total:.0f}s matches | "
          f"{HISTORY_TICKS / CONTROL_HZ:.1f}s of frame history | hub: {hub}")


MENU = [
    ("doctor", "Check installs, config and what to do next", "doctor"),
    ("setup", "Calibrate screen regions, motion and the hub indicator", "setup"),
    ("testkeys", "Check the game actually receives a key (default: H)", "testkeys"),
    ("record", "Record your own matches as training data", "record"),
    ("demos", "List / weight / disable your recordings", "demos"),
    ("pretrain", "Learn from your demos (behaviour cloning)", "pretrain"),
    ("train", "Train with PPO (auto-resumes)", "train"),
    ("play", "Watch the trained agent, with a live reward readout", "play"),
    ("reset", "Delete checkpoints (never your demos)", "reset"),
]


def interactive_menu():
    print_startup_banner()
    print("\nWhat do you want to do?\n")
    for i, (name, blurb, _) in enumerate(MENU, start=1):
        print(f"  {i}. {name:<9} {blurb}")
    print("  q. quit\n")
    reply = input("Number or name: ").strip().lower()
    if reply.startswith("q") or not reply:
        return None
    if reply.isdigit() and 1 <= int(reply) <= len(MENU):
        return MENU[int(reply) - 1][2]
    names = [m[0] for m in MENU]
    if reply in names:
        return reply
    print(f"Didn't understand '{reply}'.")
    return None


def add_debug_arguments(parser):
    parser.add_argument("--no-overlay", action="store_true", help="hide the in-game debug overlay")
    parser.add_argument("--no-reward-log", action="store_true", help="disable per-tick JSONL logging")
    parser.add_argument("--log-dir", default=os.path.join(HERE, "mosim_debug_logs"))
    parser.add_argument("--weights", help="custom MoSim YOLO best.pt (optional until trained)")
    parser.add_argument("--conf", type=float, default=0.4, help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size")
    parser.add_argument("--vision-fps", type=float, default=30, help="maximum background detection rate (default: 30; adapts to processing cost)")
    parser.add_argument("--vision-device", help="YOLO device, e.g. cpu or 0 (first CUDA GPU)")
    parser.add_argument("--corner", choices=("top-right", "top-left", "bottom-right", "bottom-left"),
                        default="top-right", help="reward panel corner")


def debug_options_from(args):
    if not 0 < args.conf <= 1 or args.imgsz < 32 or args.vision_fps <= 0:
        raise SystemExit("--conf must be in (0,1], --imgsz >= 32, and --vision-fps > 0")
    return {"overlay": not args.no_overlay, "log_rewards": not args.no_reward_log,
            "log_dir": args.log_dir, "weights": args.weights, "conf": args.conf,
            "imgsz": args.imgsz, "vision_fps": args.vision_fps,
            "vision_device": args.vision_device, "corner": args.corner}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="mosim_env.py",
        description="Train an agent to play MoSimulator (FRC REBUILT) from screen capture.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run with no arguments for an interactive menu. "
               "Calibration lives in mosim_config.json, not in the code.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("doctor", help="check installs, config and what to do next").set_defaults(func=doctor)
    sub.add_parser("learn", help="new feature-agent training; run learn --help for options")
    sub.add_parser("agent", help="feature-agent check, bootstrap, smoke, train and play commands")
    # "calibrate" and "testkey" are the old v3 names, kept so muscle memory works.
    p = sub.add_parser("setup", aliases=["calibrate"],
                       help="hands-free calibration wizard (works on a single screen)")
    p.add_argument("--only", default=None,
                   help="comma list of steps to (re)do: tesseract,regions,ocr,motion,hub,fill")
    p.add_argument("--image", default=None,
                   help="use this screenshot for regions/ocr/fill instead of capturing one "
                        "(setup saves its last capture as setup_reference.png)")
    p.set_defaults(func=setup)

    p = sub.add_parser("testkeys", aliases=["testkey"], help="check the game receives a key")
    p.add_argument("--key", default="h", choices=sorted(VK_CODES), help="which key to tap")
    p.set_defaults(func=testkeys)

    sub.add_parser("record", help="record your own matches").set_defaults(func=record)

    p = sub.add_parser("demos", help="list / weight / disable recordings")
    p.add_argument("action", nargs="?", default="list",
                   choices=["list", "weight", "disable", "enable"])
    p.add_argument("rest", nargs="*", help="arguments for the action")
    p.set_defaults(func=demos)

    p = sub.add_parser("pretrain", help="behaviour-clone your demos into the network")
    p.add_argument("--iterations", type=int, default=BC_ITERATIONS)
    p.set_defaults(func=pretrain)

    p = sub.add_parser("train", help="PPO training (auto-resumes)")
    p.add_argument("--steps", type=int, default=10_000_000, help="total timesteps")
    p.add_argument("--fresh", action="store_true",
                   help="ignore existing checkpoints and the pretrained net")
    add_debug_arguments(p)
    p.set_defaults(func=train)

    p = sub.add_parser("play", help="watch the agent play, with a live reward readout")
    p.add_argument("--model", default=None, help="path to a .zip (default: newest)")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--sample", action="store_true",
                   help="sample keys instead of taking the most likely action")
    p.add_argument("--raw", action="store_true",
                   help="don't cancel contradictory keys like W+S")
    p.add_argument("--dry-run", action="store_true", help="display proposed keys; inject NO keys or resets")
    add_debug_arguments(p)
    p.set_defaults(func=play)

    p = sub.add_parser("watch", help="debug rewards while YOU play; sends no game keys")
    add_debug_arguments(p)
    p.set_defaults(func=watch)

    p = sub.add_parser("debug-preview", help="render an offline overlay preview with synthetic rewards")
    p.add_argument("--image", required=True, help="saved game screenshot")
    p.add_argument("--output", default=os.path.join(HERE, "debug_preview.png"))
    p.add_argument("--corner", choices=("top-right", "top-left", "bottom-right", "bottom-left"),
                   default="top-right")
    p.set_defaults(func=debug_preview)

    p = sub.add_parser("reset", help="delete checkpoints (never your demos)")
    p.add_argument("--all", action="store_true", help="also delete the pretrained network")
    p.set_defaults(func=reset)

    return parser


def main():
    argv = sys.argv[1:]
    if argv and argv[0] in ("learn", "agent"):
        from crosim_agent.cli import main as agent_main
        raise SystemExit(agent_main((["train"] if argv[0] == "learn" else []) + argv[1:]))
    parser = build_parser()
    if not argv:
        chosen = interactive_menu()
        if chosen is None:
            return
        argv = [chosen]
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
