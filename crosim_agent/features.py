"""Small fixed-meaning vision observations, shared by proxy and live training."""
from collections import deque
from dataclasses import dataclass, field
import time

import cv2
import numpy as np

from mosim_landmarks import HubTagDetector, strongest_hub
from mosim_zones import FuelZoneFilter, field_fuel_mask
from mosim_field import FieldPoseTracker
from .actions import ACTIONS
from .objectives import GOAL_FEATURES, MASK_FEATURES


ROWS, COLS = 3, 6
HISTORY = 3
FEATURE_VERSION = "objective-vision-features-v2"
SCENE_NAMES = tuple(f"fuel_r{r}_c{c}" for r in range(ROWS) for c in range(COLS)) + (
    "fuel_visible", "fuel_x", "fuel_y", "fuel_mass",
    "dense_x", "dense_y", "dense_mass",
    "hub_visible", "hub_x", "hub_y", "hub_scale", "hub_age",
    "ally_visible", "ally_x", "ally_size", "opponent_visible", "opponent_x", "opponent_size",
)
STATE_NAMES = ("fill", "fill_known", "hub_active", "hub_trusted", "hub_change_soon",
               "time_left", "motion", "settling", "stuck", "intake_dry", "score_recent", "collect_goal") + GOAL_FEATURES + MASK_FEATURES
FEATURE_NAMES = SCENE_NAMES + STATE_NAMES + tuple(f"previous_{a.name}" for a in ACTIONS)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
FRAME_DIM = len(FEATURE_NAMES)
OBS_DIM = FRAME_DIM * HISTORY


@dataclass
class Scene:
    grid: np.ndarray = field(default_factory=lambda: np.zeros((ROWS, COLS), np.float32))
    fuel_visible: bool = False
    fuel_x: float = 0.0
    fuel_y: float = 0.0
    fuel_mass: float = 0.0
    dense_x: float = 0.0
    dense_y: float = 0.0
    dense_mass: float = 0.0
    hub_visible: bool = False
    hub_x: float = 0.0
    hub_y: float = 0.0
    hub_scale: float = 0.0
    hub_age: float = 1.0
    ally_visible: bool = False
    ally_x: float = 0.0
    ally_size: float = 0.0
    opponent_visible: bool = False
    opponent_x: float = 0.0
    opponent_size: float = 0.0
    fuel_coverage: float = 0.0
    fuel_regions: int = 0
    source: str = "color"
    tag_ids: tuple = ()
    captured_at: float = 0.0
    fuel_excluded: int = 0
    zone_status: str = "unspecified"
    zone_info: dict = field(default_factory=dict)
    field_map: dict = field(default_factory=dict)


def pack_features(scene, *, fill=0.0, fill_known=True, hub_active=True, hub_trusted=True,
                  hub_change=10.0, time_left=1.0, motion=0.0, settling=False,
                  stuck=0.0, intake_dry=0.0, score_recent=0.0, collect_goal=False, previous_action=None, objective=None):
    values = {name: getattr(scene, name) for name in SCENE_NAMES if not name.startswith("fuel_r")}
    values.update(fill=fill, fill_known=fill_known, hub_active=hub_active,
                  hub_trusted=hub_trusted, hub_change_soon=1-float(hub_change)/10,
                  time_left=time_left, motion=float(motion)/20, settling=settling,
                  stuck=float(stuck)/4, intake_dry=float(intake_dry)/4, score_recent=float(score_recent)/10,
                  collect_goal=collect_goal)
    if objective is not None:
        values.update(objective.features())
    else:
        values["allow_coast"] = 1.0
    output = np.zeros(FRAME_DIM, dtype=np.float32)
    output[:ROWS*COLS] = np.asarray(scene.grid, dtype=np.float32).ravel()
    for name, value in values.items():
        output[FEATURE_INDEX[name]] = float(value)
    if previous_action is not None:
        if not 0 <= int(previous_action) < len(ACTIONS):
            raise ValueError("Invalid previous action")
        output[len(SCENE_NAMES)+len(STATE_NAMES)+int(previous_action)] = 1.0
    if not np.isfinite(output).all():
        raise ValueError("Observation contains a non-finite sensor value")
    return np.clip(output, -1, 1)


class FeatureHistory:
    def __init__(self):
        self.frames = deque(maxlen=HISTORY)

    def reset(self, frame):
        self.frames.clear()
        for _ in range(HISTORY):
            self.frames.append(np.array(frame, dtype=np.float32, copy=True))
        return self.observation()

    def push(self, frame):
        if not self.frames:
            return self.reset(frame)
        self.frames.append(np.array(frame, dtype=np.float32, copy=True))
        return self.observation()

    def observation(self):
        return np.concatenate(list(self.frames)) if self.frames else np.zeros(OBS_DIM, np.float32)


class FeatureExtractor:
    """Fuel cells/targets and fresh alliance-specific AprilTag anchors.

    Coordinates and tag size are image-space proxies, not field pose or meters.
    Grid cells keep the same meaning across frames; boxes are never sorted into
    observation slots by confidence.
    """
    def __init__(self, config, alliance="blue", landmarks=True, zones=True,persist_pose=False):
        if alliance not in {"blue", "red"}:
            raise ValueError("Alliance must be blue or red")
        self.config = dict(config)
        self.alliance = alliance
        self.tags = HubTagDetector(max_width=1280, interval=0, hub_only=False) if landmarks else None
        self.zone_filter = FuelZoneFilter(alliance) if zones else None
        self.field_tracker = FieldPoseTracker(alliance,persist=persist_pose)
        self._reversed_view = False

    def reset(self):
        if self.tags:
            self.tags.reset()
        if self.zone_filter:
            self.zone_filter.reset()
        self.field_tracker.reset()

    def analyze(self, frame, captured_at=None, detections=None,pose_enabled=True,reversed_view=False):
        captured_at = time.perf_counter() if captured_at is None else captured_at
        if self._reversed_view != reversed_view:
            # Optical-flow corners from the opposite camera must not be reused.
            if self.tags:self.tags.reset()
            if self.zone_filter:self.zone_filter.reset()
            self._reversed_view = reversed_view
        mask = field_fuel_mask(frame,self.config)
        map_fuel_mask = mask
        sh,sw = mask.shape
        scene = Scene(captured_at=captured_at)
        anchors = self.tags.detect(frame,captured_at) if self.tags else []
        if self.zone_filter:
            access = self.zone_filter.apply(mask,anchors)
            mask = access.eligible
            scene.fuel_excluded = access.excluded_regions
            scene.zone_status = access.status
            scene.zone_info = access.snapshot()
        scene.field_map = self.field_tracker.update(
            self.zone_filter.hypotheses if self.zone_filter and pose_enabled else [],self.config["hopper_region"],
            captured_at=captured_at,detections=detections or (),fuel_mask=map_fuel_mask if pose_enabled else None,
            reversed_view=reversed_view)
        scene.fuel_coverage = float(np.count_nonzero(mask)/mask.size)
        for row in range(ROWS):
            for col in range(COLS):
                cell = mask[int(row*sh/ROWS):int((row+1)*sh/ROWS),
                            int(col*sw/COLS):int((col+1)*sw/COLS)]
                scene.grid[row, col] = min(1.0, float(np.count_nonzero(cell)/cell.size)/0.25)
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
        candidates = []
        for index in range(1, count):
            x, y, bw, bh, area = stats[index]
            if area < 6 or area > sw*sh*0.20:
                continue
            cx, cy = centers[index]/np.array([sw, sh])
            mass = min(float(area)/(sw*sh*0.04), 1.0)
            # Favor low, substantial visible regions; this is a proximity proxy.
            candidates.append((float(cx)*2-1, float(cy), mass,
                               float(cy)+0.15*mass, mass))
        scene.fuel_regions = len(candidates)
        if candidates:
            closest = max(candidates, key=lambda item: item[3])
            densest = max(candidates, key=lambda item: item[4])
            scene.fuel_visible = True
            scene.fuel_x, scene.fuel_y, scene.fuel_mass = closest[:3]
            scene.dense_x, scene.dense_y, scene.dense_mass = densest[:3]
        # Nearby disconnected yellow balls are one useful intake objective.
        # Measure original yellow pixels, not the inflated dilation area.
        if candidates:
            groups = cv2.dilate(mask, np.ones((13, 17), np.uint8))
            group_count, group_labels, _, _ = cv2.connectedComponentsWithStats(groups, 8)
            ys, xs = np.nonzero(mask)
            ids = group_labels[ys, xs]
            counts = np.bincount(ids, minlength=group_count)
            group = int(np.argmax(counts[1:]))+1
            selected = ids == group
            scene.dense_x = float(xs[selected].mean()/sw*2-1)
            scene.dense_y = float(ys[selected].mean()/sh)
            scene.dense_mass = min(float(counts[group])/(sw*sh*0.04), 1.0)
        hub = strongest_hub(anchors, self.alliance, now=time.perf_counter())
        if hub:
            scene.hub_visible = True
            scene.hub_x = hub["x"]*2-1
            scene.hub_y = hub["y"]
            scene.hub_scale = min(hub["scale"]/0.10, 1.0)
            scene.hub_age = min(hub["age"]/0.4, 1.0)
            scene.tag_ids = tuple(hub["tag_ids"])
            scene.source = "color+apriltag_36h11"
        # Optional robot boxes only occupy their own fixed ally/opponent slots.
        for team, prefix in ((self.alliance, "ally"),
                             ("red" if self.alliance == "blue" else "blue", "opponent")):
            selected = [d for d in (detections or []) if d.get("name") == f"{team}_robot"]
            if selected:
                box = max(selected, key=lambda d: (d["box"][2]-d["box"][0])*(d["box"][3]-d["box"][1]))["box"]
                setattr(scene, f"{prefix}_visible", True)
                setattr(scene, f"{prefix}_x", float(box[0]+box[2]-1))
                setattr(scene, f"{prefix}_size", min(1.0, float((box[2]-box[0])*(box[3]-box[1]))/0.2))
        return scene


def latest_frame(observation):
    return np.asarray(observation, dtype=np.float32).reshape(HISTORY, FRAME_DIM)[-1]


def feature_value(observation, name):
    return float(latest_frame(observation)[FEATURE_INDEX[name]])
