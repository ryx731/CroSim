"""Versioned, bounded rewards for fast initial collection/scoring learning.

Potential differences use the same per-control-tick discount as the learner's
action-repeat aggregation. A terminal state settles ALL remaining potential.
Unknown measurements freeze their potential; they do not erase earlier debt.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np


REWARD_VERSION = "score-first-reward-v6"
LABELS = {
    "score": "Confirmed alliance score",
    "fuel": "Fuel potential",
    "approach": "Target potential",
    "time": "Time cost",
    "idle": "No task progress",
    "wasted_shot": "Wasted shot",
    "spin": "Prolonged turning",
    "sensor": "Sensor / camera",
    "red_zone": "Red alliance zone",
    "shift_zone": "Wrong zone for shift",
    "contact": "Structure contact",
}


@dataclass(frozen=True)
class RewardConfig:
    gamma: float = 0.995
    score_per_point: float = 3.6
    fuel_potential: float = .20
    fuel_target_potential: float = .05
    hub_target_potential: float = 0.6
    time_per_second: float = 0.01
    idle_per_second: float = 0.05
    idle_grace_seconds: float = 3.0
    inactive_shot_per_second: float = 0.12
    empty_shot_per_second: float = 0.04
    spin_per_second: float = .60
    spin_grace_seconds: float = 0.6
    sensor_cost: float = 0.03
    score_window_seconds: float = 2.0
    red_zone_entry_cost: float = .75
    red_zone_per_second: float = .08
    zone_confirm_seconds: float = .4
    shift_zone_per_second: float = .10
    shift_zone_grace_seconds: float = 2.0
    blue_bump_per_second: float = .8
    blue_bump_grace_seconds: float = 1.5
    red_structure_per_second: float = .4

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Reward setting {name} must be finite and nonnegative")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")


@dataclass(frozen=True)
class RewardState:
    fill: float = 0.0
    fill_valid: bool = True
    hub_active: bool = True
    hub_trusted: bool = True
    hub_visible: bool = False
    hub_x: float = 0.0
    hub_scale: float = 0.0
    fuel_visible: bool = False
    fuel_x: float = 0.0
    fuel_y: float = 0.0
    fuel_mass: float = 0.0
    motion: float = 0.0
    shooting: bool = False
    movement_requested: bool = False
    elapsed: float = 0.0
    action: int = 0
    settling: bool = False
    camera_changed: bool = False
    goal: str = "collect"
    target_visible: bool = False
    target_x: float = 0.0
    target_y: float = 0.0
    target_scale: float = 0.0
    rotating: bool = False
    translating: bool = False
    opponent_zone: bool | None = None
    zone_sample_at: float | None = None
    blue_zone: bool | None = None
    shift_on: bool | None = None
    blue_bump: bool | None = None
    red_structure: bool | None = None


class CycleReward:
    def __init__(self, config=None):
        self.config = config or RewardConfig()
        self.reset(RewardState())

    def potentials(self, state):
        fill = float(np.clip(state.fill, 0, 1))
        fuel = self.config.fuel_potential * fill
        if state.goal == "collect" and (state.target_visible or state.fuel_visible):
            x = state.target_x if state.target_visible else state.fuel_x
            y = state.target_y if state.target_visible else state.fuel_y
            alignment = max(0.0, 1-abs(x))
            target = self.config.fuel_target_potential * float(np.clip(y, 0, 1)) * alignment
        elif state.goal in {"return", "shoot", "wait"} and state.hub_visible:
            alignment = max(0.0, 1-abs(state.hub_x))
            target = self.config.hub_target_potential * (0.5*float(np.clip(state.hub_scale, 0, 1)) + 0.5*alignment)
        else:
            target = 0.0
        return fuel, target

    def reset(self, state):
        self.previous = state
        self.phi_fuel, self.phi_target = self.potentials(state)
        self.last_valid_shot = -float("inf")
        self.idle_seconds = 0.0
        self.credited_points = 0
        self.uncredited_points = 0
        self.steps = 0
        self.best_fill = state.fill if state.fill_valid else 0.0
        self.best_unload_fill = self.best_fill
        self.best_fuel_approach = self.best_hub_approach = 0.0
        self.rotation_seconds = self.translation_seconds = 0.0
        self.inventory_baselined = state.fill_valid and not state.settling
        self.zone_confirmed = None
        self.zone_candidate = None
        self.zone_seconds = 0.0
        self.zone_last_sample = None
        self.blue_contact_seconds = self.red_contact_seconds = 0.0
        self.contact_last_sample = None
        self.shift_state = None
        self.shift_since = None
        self.shift_zone_seconds = 0.0
        self.shift_zone_last_sample = None

    def shift_zone_cost(self,state,dt):
        """Charge occupancy only with fresh, confirmed arrow and zone evidence."""
        if state.settling or state.shift_on is None or state.blue_zone is None:
            self.shift_zone_seconds = 0.0
            return 0.0,"Shift-zone cost paused: arrow or field boundary uncertain"
        if state.shift_on != self.shift_state:
            self.shift_state,self.shift_since = state.shift_on,state.elapsed
            self.shift_zone_seconds = 0.0
        if state.zone_sample_at is None or (self.shift_zone_last_sample is not None
                                            and state.zone_sample_at <= self.shift_zone_last_sample):
            return 0.0,"Shift-zone cost paused: no new pose"
        self.shift_zone_last_sample = state.zone_sample_at
        misplaced = state.blue_zone != state.shift_on
        self.shift_zone_seconds = self.shift_zone_seconds+dt if misplaced else 0.0
        if not misplaced:
            return 0.0,"Correct zone for observed shift"
        if state.elapsed-self.shift_since < self.config.shift_zone_grace_seconds:
            return 0.0,"Shift changed: two-second travel grace"
        if self.shift_zone_seconds+1e-9 < self.config.zone_confirm_seconds:
            return 0.0,"Confirming shift-zone mismatch"
        reason = "ON arrow present: outside blue alliance zone" if state.shift_on else "OFF arrow absent: inside blue alliance zone"
        return -self.config.shift_zone_per_second*dt,reason

    def zone_cost(self,state,dt):
        """Confirm crossings; uncertainty never invents entries or occupancy."""
        evidence = state.opponent_zone if not state.settling else None
        entry = False
        if evidence is None:
            self.zone_candidate,self.zone_seconds = None,0.0
            return 0.0,"Field pose uncertain: zone cost paused",entry
        if evidence != self.zone_candidate:
            self.zone_candidate,self.zone_seconds = evidence,0.0
        if state.zone_sample_at is not None and self.zone_last_sample is not None and state.zone_sample_at <= self.zone_last_sample:
            return 0.0,"No new field pose: zone cost paused",entry
        self.zone_seconds += dt
        self.zone_last_sample = state.zone_sample_at
        if self.zone_seconds+1e-9 < self.config.zone_confirm_seconds:
            return 0.0,"Confirming field-zone estimate",entry
        if evidence and self.zone_confirmed is False:
            entry = True
        self.zone_confirmed = evidence
        cost = (self.config.red_zone_entry_cost if entry else 0.0)+(self.config.red_zone_per_second*dt if evidence else 0.0)
        reason = "Confirmed crossing into red alliance zone" if entry else "Confidently inside red alliance zone" if evidence else "Outside red alliance zone"
        return -cost,reason,entry

    def update(self, state, points=0, *, dt=0.1, terminal=False):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Reward dt must be finite and positive")
        if not all(math.isfinite(value) for value in (state.fill, state.elapsed, state.motion,
                                                       state.hub_x, state.hub_scale, state.fuel_x,
                                                       state.fuel_y, state.fuel_mass, state.target_x,
                                                       state.target_y, state.target_scale)):
            raise ValueError("Invalid reward observation")
        if any(value is not None and not isinstance(value,bool) for value in (state.opponent_zone,state.blue_zone,state.shift_on)):
            raise ValueError("Field-zone evidence must be true, false or unknown")
        if state.zone_sample_at is not None and not math.isfinite(state.zone_sample_at):
            raise ValueError("Invalid field pose timestamp")
        points = int(points)
        if not 0 <= points <= 40:
            raise ValueError("Score delta must already be debounced and bounded to 0..40")
        cfg = self.config
        parts = dict.fromkeys(LABELS, 0.0)
        reasons = {name: label for name, label in LABELS.items()}
        previous_fill = self.previous.fill if self.previous.fill_valid and not self.previous.settling else 0.0
        loaded = state.fill_valid and not state.settling and max(state.fill, previous_fill) >= 0.03
        # Observed score is stronger evidence than uncertain inventory or hub
        # color. A real Space press (including human assistance) starts credit.
        if state.shooting:
            self.last_valid_shot = state.elapsed
        correlated = 0 <= state.elapsed-self.last_valid_shot <= cfg.score_window_seconds
        credited = points
        self.credited_points += credited
        self.uncredited_points += points-credited
        parts["score"] = cfg.score_per_point * credited
        reasons["score"] = f"{credited} confirmed alliance point(s); all score increases credited"
        if state.fill_valid and not state.settling and not self.inventory_baselined:
            self.phi_fuel = self.potentials(state)[0]
            self.best_fill = self.best_unload_fill = state.fill
            self.inventory_baselined = True
        before_fuel, before_target = self.phi_fuel, self.phi_target
        if terminal:
            fuel, target = 0.0, 0.0
            reason = "Terminal settlement of stored potential"
        elif state.settling:
            fuel, target = before_fuel, before_target
            reason = "Uncertain sensor: keep potential; no new visual credit"
        else:
            fuel, target = self.potentials(state)
            if not state.fill_valid:
                fuel = before_fuel
                if not (state.target_visible or state.fuel_visible or state.hub_visible):
                    target = before_target
            reason = "Discounted change in measured potential"
        parts["fuel"] = cfg.gamma*fuel-before_fuel
        parts["approach"] = cfg.gamma*target-before_target
        reasons["fuel"] = reasons["approach"] = reason
        self.phi_fuel, self.phi_target = fuel, target
        parts["time"] = -cfg.time_per_second*dt
        reasons["time"] = "Small fixed cost for real control time"
        # Raw camera motion is not progress. Use monotone high-water marks so
        # alternating turns or noisy fill oscillations cannot refresh grace.
        fill_progress = state.fill_valid and not state.settling and state.fill >= self.best_fill+0.02
        if fill_progress:
            self.best_fill = state.fill
            self.best_unload_fill = state.fill
        unloading_progress = (loaded and state.shooting and state.hub_active and state.hub_trusted
                              and state.fill < self.best_unload_fill-0.02)
        if unloading_progress:
            self.best_unload_fill = state.fill
            self.best_fill = state.fill
        approach_progress = False
        if not state.settling and state.goal == "collect" and (state.target_visible or state.fuel_visible):
            proximity = state.target_y if state.target_visible else state.fuel_y
            if proximity > self.best_fuel_approach+0.035:
                approach_progress = True
                self.best_fuel_approach = proximity
        elif not state.settling and state.goal in {"return", "shoot"} and state.hub_visible:
            if state.hub_scale > self.best_hub_approach+0.035:
                approach_progress = True
                self.best_hub_approach = state.hub_scale
        waiting_at_hub = (state.goal == "wait" and loaded and state.hub_trusted and not state.hub_active
                          and state.hub_visible and abs(state.hub_x) <= 0.14 and state.hub_scale >= 0.5)
        if credited or fill_progress or unloading_progress or approach_progress or waiting_at_hub:
            self.idle_seconds = 0.0
            if credited:
                self.best_fill = state.fill
                self.best_fuel_approach = self.best_hub_approach = 0.0
        elif not state.settling:
            self.idle_seconds += dt
        if self.idle_seconds > cfg.idle_grace_seconds:
            parts["idle"] = -cfg.idle_per_second*dt
            reasons["idle"] = "No reliable inventory progress, points or closer target; motion alone does not count"
        if state.rotating:
            self.rotation_seconds += dt
        self.translation_seconds = self.translation_seconds+dt if state.translating and not state.rotating else 0.0
        if self.translation_seconds >= 0.6:
            self.rotation_seconds = 0.0
        if state.rotating and self.rotation_seconds > cfg.spin_grace_seconds:
            parts["spin"] = -cfg.spin_per_second*dt
            reasons["spin"] = "Repeated turning without a sustained translation break"
        if state.shooting and state.hub_trusted and not state.hub_active:
            parts["wasted_shot"] -= cfg.inactive_shot_per_second*dt
            reasons["wasted_shot"] = "Shooting while the hub is known inactive"
        if state.shooting and state.fill_valid and state.fill < 0.03:
            parts["wasted_shot"] -= cfg.empty_shot_per_second*dt
            reasons["wasted_shot"] += "; hopper appears empty"
        if state.camera_changed:
            parts["sensor"] = -cfg.sensor_cost
            reasons["sensor"] = "Camera changed; inventory measurement requires settling"
        parts["red_zone"],reasons["red_zone"],zone_entry = self.zone_cost(state,dt)
        parts["shift_zone"],reasons["shift_zone"] = self.shift_zone_cost(state,dt)
        fresh_contact = (not state.settling and state.zone_sample_at is not None
                         and (self.contact_last_sample is None or state.zone_sample_at > self.contact_last_sample))
        if fresh_contact:
            self.contact_last_sample = state.zone_sample_at
            self.blue_contact_seconds = self.blue_contact_seconds+dt if state.blue_bump is True else 0.0
            self.red_contact_seconds = self.red_contact_seconds+dt if state.red_structure is True else 0.0
            if self.blue_contact_seconds > cfg.blue_bump_grace_seconds+1e-9:
                parts["contact"] -= cfg.blue_bump_per_second*dt
                reasons["contact"] = "Estimated blue bump contact beyond crossing grace"
            if self.red_contact_seconds >= cfg.zone_confirm_seconds:
                parts["contact"] -= cfg.red_structure_per_second*dt
                reasons["contact"] = "Estimated red hub/bump/trench contact"
        else:
            self.blue_contact_seconds = self.red_contact_seconds = 0.0
        self.previous = state
        self.steps += 1
        diagnostics = {"reward_version": REWARD_VERSION, "phi_fuel_before": before_fuel,
                       "phi_fuel_after": fuel, "phi_target_before": before_target,
                       "phi_target_after": target, "score_credited": credited,
                       "score_uncredited": points-credited, "idle_seconds": self.idle_seconds,
                       "potential_terminal": bool(terminal), "sensor_valid": state.fill_valid and not state.settling}
        diagnostics.update(goal=state.goal, task_progress=bool(credited or fill_progress or unloading_progress or approach_progress),
                           rotation_seconds=self.rotation_seconds,opponent_zone=state.opponent_zone,
                           red_zone_entry=zone_entry,zone_confirmed=self.zone_confirmed,
                           blue_zone=state.blue_zone,shift_on=state.shift_on)
        diagnostics.update(phi_before=before_fuel+before_target, phi_after=fuel+target, gamma=cfg.gamma)
        return parts, reasons, diagnostics
