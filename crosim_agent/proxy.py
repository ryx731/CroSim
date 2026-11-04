"""Fast, randomized practice world for interface tests and an initial policy.

This is NOT MoSimulator physics or an estimate of real-game playing strength.
It trains coarse visual collection/aiming habits without taking over the screen.
The live game is still required to learn its actual dynamics and validate gains.
"""
import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .actions import ACTIONS, action_bits, tick_bits, KEYS
from .features import Scene, FeatureHistory, OBS_DIM, ROWS, COLS, pack_features
from .rewards import CycleReward, RewardConfig, RewardState, LABELS
from .objectives import ObjectiveController


CONTROL_DT = 0.1
ACTION_REPEAT = 2
RED_ZONE_START = 3.8


class PracticeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, duration=60.0, reward_config=None, action_repeat=ACTION_REPEAT, sensor_dropout=False):
        super().__init__()
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(-1, 1, shape=(OBS_DIM,), dtype=np.float32)
        self.duration = float(duration)
        self.action_repeat = int(action_repeat)
        self.sensor_dropout = sensor_dropout
        if self.duration <= 0 or not 1 <= self.action_repeat <= 5:
            raise ValueError("Invalid practice duration or action repeat")
        self.reward_config = reward_config or RewardConfig()
        self.reward_engine = CycleReward(self.reward_config)
        self.history = FeatureHistory()
        self.objectives = ObjectiveController()
        self.decision = None

    def _relative(self, target):
        delta = np.asarray(target)-self.position
        distance = max(float(np.linalg.norm(delta)), 0.01)
        angle = (math.atan2(delta[1], delta[0])-self.yaw+math.pi) % (2*math.pi)-math.pi
        return distance, angle

    def _scene(self):
        scene = Scene(zone_status="proxy world coordinates")
        candidates = []
        fov = math.radians(60)
        for point, remaining in zip(self.fuel, self.fuel_remaining):
            if remaining <= 0:
                continue
            distance, angle = self._relative(point)
            if abs(angle) > fov or distance > 11:
                continue
            if point[0] >= RED_ZONE_START:
                scene.fuel_excluded += 1
                continue
            x = float(np.clip(-angle/fov, -1, 1))
            y = float(np.clip(1-distance/12, 0.08, 0.84))
            mass = float(np.clip(remaining/40/(1+distance*0.12), 0, 1))
            row, col = min(ROWS-1, int(y*ROWS)), min(COLS-1, int((x+1)*0.5*COLS))
            scene.grid[row, col] = min(1.0, scene.grid[row, col]+mass*0.5)
            candidates.append((distance, x, y, mass))
        if candidates:
            nearest = min(candidates)
            densest = max(candidates, key=lambda c: c[3])
            scene.fuel_visible = True
            scene.fuel_x, scene.fuel_y, scene.fuel_mass = nearest[1:]
            scene.dense_x, scene.dense_y, scene.dense_mass = densest[1:]
        scene.fuel_regions = len(candidates)
        distance, angle = self._relative(self.hub)
        if abs(angle) <= fov and distance < 14:
            scene.hub_visible = True
            scene.hub_x = float(np.clip(-angle/fov, -1, 1))
            scene.hub_y = float(np.clip(0.6+0.12/max(distance, 0.5), 0, 1))
            scene.hub_scale = min(1.0, 1.8/max(distance, 0.5))
            scene.hub_age = 0.0
        return scene

    @property
    def hub_active(self):
        # Randomized alternating availability, with an initial common active period.
        return self.elapsed < 10 or int((self.elapsed+self.shift_offset)/12) % 2 == 0

    def _state(self, scene, action, motion=0, keys=None):
        keys = ACTIONS[action].keys if keys is None else keys
        return RewardState(fill=self.observed_fill,fill_valid=self.inventory_known, hub_active=self.hub_active,
                           hub_visible=scene.hub_visible, hub_x=scene.hub_x, hub_scale=scene.hub_scale,
                           fuel_visible=scene.fuel_visible, fuel_x=scene.fuel_x,
                           fuel_y=scene.fuel_y, fuel_mass=scene.fuel_mass, motion=motion,
                           shooting="space" in keys, movement_requested=any(k in keys for k in "wasdjl"),
                           elapsed=self.elapsed, action=action, goal=self.decision.goal if self.decision else "collect",
                           target_visible=bool(self.decision and self.decision.target_visible),
                           target_x=self.decision.target_x if self.decision else 0.0,
                           target_y=self.decision.target_y if self.decision else 0.0,
                           target_scale=self.decision.target_scale if self.decision else 0.0,
                           rotating="j" in keys or "l" in keys, translating=any(key in keys for key in "wasd"),
                           opponent_zone=bool(self.position[0] >= RED_ZONE_START),zone_sample_at=self.elapsed)

    def _features(self, scene, action=None, motion=0, points=0):
        return pack_features(scene, fill=self.observed_fill,fill_known=self.inventory_known, hub_active=self.hub_active,
                             time_left=max(0, 1-self.elapsed/self.duration), motion=motion,
                             stuck=self.reward_engine.idle_seconds, intake_dry=self.dry_seconds,
                             score_recent=points, previous_action=action, objective=self.decision)

    def _sense_inventory(self):
        self.inventory_known = not self.sensor_dropout or int((self.elapsed+self.sensor_phase)/2)%3 == 0
        if self.inventory_known:
            self.observed_fill = self.hopper/self.capacity

    def _objective(self, keys=None, dt=0.0,points=0):
        self.decision = self.objectives.update(self.scene, fill=self.observed_fill,fill_known=self.inventory_known,
                                               hub_active=self.hub_active, time_remaining=self.duration-self.elapsed,
                                               no_progress=self.reward_engine.idle_seconds,
                                               intake_dry=self.dry_seconds,
                                               keys=None if keys is None else np.array([key in keys for key in KEYS]), dt=dt,points=points)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = self.np_random.uniform([-6, -2.5], [5, 2.5])
        self.yaw = float(self.np_random.uniform(-math.pi, math.pi))
        self.speed = float(self.np_random.uniform(1.8, 3.2))
        self.turn_speed = float(self.np_random.uniform(1.0, 1.9))
        self.capacity = int(self.np_random.integers(15, 36))
        self.hub = np.array([-3.2, 0.0])
        self.fuel = self.np_random.uniform([-7, -3.5], [7, 3.5], size=(30, 2))
        self.fuel_remaining = self.np_random.uniform(5, 35, size=30)
        self.hopper = 0.0
        self.hopper_open = bool(self.np_random.integers(2))
        self.shift_offset = float(self.np_random.uniform(0, 12))
        self.elapsed = self.score = self.collected = self.shot_fraction = self.dry_seconds = 0.0
        self.sensor_phase = float(self.np_random.uniform(0,6))
        self.observed_fill = 0.0
        self._sense_inventory()
        self.previous_action = 0
        self.total_reward = 0.0
        self.finished = False
        self.objectives.reset()
        self.scene = self._scene()
        self._objective()
        self.reward_engine.reset(self._state(self.scene, 0))
        return self.history.reset(self._features(self.scene)), {"domain": "practice_proxy"}

    def _tick(self, action, tick=0):
        keys = tuple(key for key, active in zip(KEYS, tick_bits(action, tick)) if active)
        before = self.position.copy()
        previous_yaw = self.yaw
        previous_fill = self.hopper
        self.yaw += (int("j" in keys)-int("l" in keys))*self.turn_speed*CONTROL_DT
        self.yaw = (self.yaw+math.pi) % (2*math.pi)-math.pi
        forward = np.array([math.cos(self.yaw), math.sin(self.yaw)])
        right = np.array([math.sin(self.yaw), -math.cos(self.yaw)])
        movement = (int("w" in keys)-int("s" in keys))*forward + (int("d" in keys)-int("a" in keys))*right
        norm = float(np.linalg.norm(movement))
        if norm > 1:
            movement /= norm
        self.position = np.clip(self.position+movement*self.speed*CONTROL_DT, [-7.8, -3.8], [7.8, 3.8])
        if action == 14 and self.previous_action != 14:
            self.hopper_open = not self.hopper_open
        if "shift" in keys and self.hopper_open and self.hopper < self.capacity:
            for index, remaining in enumerate(self.fuel_remaining):
                distance, angle = self._relative(self.fuel[index])
                if remaining > 0 and self.fuel[index][0] < RED_ZONE_START and distance < 1.0 and abs(angle) < math.radians(70):
                    amount = min(12*CONTROL_DT, remaining, self.capacity-self.hopper)
                    self.hopper += amount
                    self.fuel_remaining[index] -= amount
                    self.collected += amount
        points = 0
        distance, angle = self._relative(self.hub)
        if "space" in keys and self.hub_active and self.hopper > 0 and distance < 4.0 and abs(angle) < 0.18:
            amount = min(self.hopper, 8*CONTROL_DT)
            self.hopper -= amount
            self.shot_fraction += amount
            points = int(self.shot_fraction)
            self.shot_fraction -= points
            self.score += points
        self.elapsed += CONTROL_DT
        self._sense_inventory()
        motion = float(np.linalg.norm(self.position-before)*40+abs(self.yaw-previous_yaw)*15)
        self.scene = self._scene()
        if "shift" in keys and self.scene.fuel_visible and self.scene.fuel_y > 0.65 and self.hopper-previous_fill < 0.01:
            self.dry_seconds += CONTROL_DT
        else:
            self.dry_seconds = 0.0
        self._objective(keys, CONTROL_DT,points)
        terminal = self.elapsed >= self.duration
        parts, reasons, diagnostics = self.reward_engine.update(self._state(self.scene, action, motion, keys),
                                                               points, dt=CONTROL_DT, terminal=terminal)
        self.previous_action = action
        observation = self.history.push(self._features(self.scene, action, motion, points))
        return observation, parts, reasons, diagnostics, terminal

    def step(self, action):
        raw = np.asarray(action)
        if raw.size != 1 or not np.isfinite(raw).all() or float(raw.item()) != int(raw.item()):
            raise ValueError("Practice agent expects one integer action")
        action = int(raw.item())
        action_bits(action)  # validate before updating any state
        if self.finished:
            raise RuntimeError("Episode is finished; call reset()")
        parts = {name: 0.0 for name in LABELS}
        discount = 1.0
        for tick in range(self.action_repeat):
            observation, tick_parts, reasons, diagnostics, terminal = self._tick(action, tick)
            for name, value in tick_parts.items():
                parts[name] += discount*value
            discount *= self.reward_config.gamma
            if terminal:
                break
        reward = float(math.fsum(parts.values()))
        self.total_reward += reward
        self.finished = terminal
        info = {"domain": "practice_proxy", "points": int(self.score),
                "collected": float(self.collected), "reward_components": parts,
                "reward_reasons": reasons, "reward_diagnostics": diagnostics,
                "objective": self.decision.snapshot()}
        if terminal:
            info["episode"] = {"r": self.total_reward, "l": round(self.elapsed/(CONTROL_DT*self.action_repeat))}
        return observation, reward, terminal, False, info
