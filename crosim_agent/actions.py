"""A small, versioned action vocabulary with coherent, non-conflicting inputs."""
from dataclasses import dataclass

import numpy as np


KEYS = ("w", "a", "s", "d", "j", "l", "shift", "space", "k", "h")


@dataclass(frozen=True)
class Action:
    name: str
    keys: tuple

    def bits(self):
        return np.asarray([key in self.keys for key in KEYS], dtype=np.int8)


# Camera switching uses H with a cooldown; view changes invalidate pose rewards.
ACTIONS = (
    Action("coast", ()),
    Action("collect_forward", ("w", "shift")),
    Action("collect_left", ("a", "shift")),
    Action("collect_right", ("d", "shift")),
    Action("collect_back", ("s", "shift")),
    Action("collect_forward_left", ("w", "a", "shift")),
    Action("collect_forward_right", ("w", "d", "shift")),
    Action("search_left", ("j", "shift")),
    Action("search_right", ("l", "shift")),
    Action("approach_hub", ("w",)),
    Action("retreat", ("s",)),
    Action("shoot", ("space",)),
    Action("aim_left", ("j",)),
    Action("aim_right", ("l",)),
    Action("toggle_hopper", ("k",)),
    Action("return_left", ("a",)),
    Action("return_right", ("d",)),
    Action("camera_swap", ("h",)),
)
ACTION_INDEX = {action.name: index for index, action in enumerate(ACTIONS)}
ACTION_VERSION = "objective-actions-v3-learned-camera-H"


def action_bits(index):
    if not isinstance(index, (int, np.integer)) or not 0 <= int(index) < len(ACTIONS):
        raise ValueError(f"Action must be an integer from 0 through {len(ACTIONS)-1}")
    return ACTIONS[int(index)].bits()


def tick_bits(index, tick=0):
    """Aim actions are one-tick nudges followed by release within a decision."""
    bits = action_bits(index)
    if int(index) in (12, 13) and tick > 0:
        bits[:] = 0
    if int(index) == ACTION_INDEX['camera_swap'] and tick > 0:
        bits[:] = 0
    return bits
