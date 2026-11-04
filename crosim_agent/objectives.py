"""Explicit, observable objectives and admissible actions for the learned pilot.

The controller chooses the task and target. PPO learns among the permitted
controls, using the same mask during sampling and gradient updates. Explanations
describe measured inputs and controller rules, not a model's private thoughts.
"""
from dataclasses import dataclass

import numpy as np

from .actions import ACTIONS, ACTION_INDEX


OBJECTIVE_VERSION = "guided-learning-v5"
GUIDANCE_SCALE = .40
HOPPER_TOGGLE_COOLDOWN = 3.0
FULL_HOPPER = 0.65
EMPTY_HOPPER = 0.08
TURN_LIMIT_SECONDS = 0.8
TRANSLATION_RESET_SECONDS = 0.6
GOALS = ("collect", "return", "shoot", "wait", "recover")
GOAL_FEATURES = tuple("goal_"+name for name in GOALS) + (
    "target_visible", "target_x", "target_y", "target_scale", "turn_budget", "no_progress")
MASK_FEATURES = tuple("allow_"+action.name for action in ACTIONS)


@dataclass(frozen=True)
class ObjectiveDecision:
    goal: str
    reason: str
    target_kind: str
    target_visible: bool
    target_x: float
    target_y: float
    target_scale: float
    allowed: tuple
    recommended: int
    turn_budget: float = 1.0
    no_progress: float = 0.0

    def features(self):
        values = {"goal_"+name: GUIDANCE_SCALE*float(self.goal == name) for name in GOALS}
        values.update(collect_goal=GUIDANCE_SCALE*float(self.goal == "collect"), target_visible=GUIDANCE_SCALE*float(self.target_visible),
                      target_x=GUIDANCE_SCALE*self.target_x, target_y=GUIDANCE_SCALE*self.target_y, target_scale=GUIDANCE_SCALE*self.target_scale,
                      turn_budget=self.turn_budget, no_progress=min(self.no_progress/10, 1))
        values.update({name: float(value) for name, value in zip(MASK_FEATURES, self.allowed)})
        return values

    def snapshot(self):
        return {"goal": self.goal, "reason": self.reason, "target_kind": self.target_kind,
                "target_visible": self.target_visible, "target_x": self.target_x,
                "target_y": self.target_y, "target_scale": self.target_scale,
                "allowed_actions": [action.name for action, allowed in zip(ACTIONS, self.allowed) if allowed],
                "recommended_action": ACTIONS[self.recommended].name,
                "guidance_fraction": GUIDANCE_SCALE,
                "turn_budget": self.turn_budget, "no_progress_seconds": self.no_progress}


class ObjectiveController:
    """Advisory targets, bounded exploration and short scoring probes.

    Collection/return labels do not lock the policy into one movement. Masks
    cover opponent-fuel access, camera settling and hopper-toggle cooldown.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.returning = False
        self.elapsed = self.collect_seconds = self.return_seconds = 0.0
        self.turn_seconds = self.translation_seconds = 0.0
        self.full_seconds = self.empty_seconds = 0.0
        self.shot_hold = self.shot_cooldown = self.shot_fill = 0.0
        self.recover_seconds = self.recover_cooldown = 0.0
        self.recoveries = 0
        self.prefer_dense = False
        self.hopper_cooldown = 0.0
        self.camera_cooldown = 0.0

    def update(self, scene, *, fill, fill_known=True, hub_active=True, hub_trusted=True,
               time_remaining=163, no_progress=0.0, intake_dry=0.0, keys=None, dt=0.1,
               settling=False, confirm_seconds=0.0, points=0):
        if not np.isfinite([fill,time_remaining,no_progress,intake_dry,dt]).all() or dt < 0:
            raise ValueError("Invalid objective sensors")
        keys = np.zeros(10,np.int8) if keys is None else np.asarray(keys)
        self.elapsed += dt
        self.camera_cooldown = max(0,self.camera_cooldown-dt)
        if keys[9]:self.camera_cooldown = 3.0
        self.hopper_cooldown = max(0.0,self.hopper_cooldown-dt)
        if keys[8]:
            self.hopper_cooldown = HOPPER_TOGGLE_COOLDOWN
        rotating,translating = bool(keys[4] or keys[5]),bool(np.any(keys[:4]))
        if rotating:
            self.turn_seconds += dt
        self.translation_seconds = self.translation_seconds+dt if translating and not rotating else 0.0
        if self.translation_seconds >= TRANSLATION_RESET_SECONDS:
            self.turn_seconds = 0.0
        self.collect_seconds += dt if not self.returning else 0.0
        self.return_seconds += dt if self.returning else 0.0
        for name in ("shot_hold","shot_cooldown","recover_seconds","recover_cooldown"):
            setattr(self,name,max(0.0,getattr(self,name)-dt))
        full = fill_known and fill >= FULL_HOPPER
        empty = fill_known and fill <= EMPTY_HOPPER
        self.full_seconds = self.full_seconds+dt if full else 0.0
        self.empty_seconds = self.empty_seconds+dt if empty else 0.0
        timed_return = self.collect_seconds >= 16.0
        partial_return = fill_known and fill >= .15 and (time_remaining <= 20 or intake_dry >= 3)
        if not self.returning and ((full and self.full_seconds >= confirm_seconds) or timed_return or partial_return):
            self.returning,self.return_seconds = True,0.0
        if self.returning and not points and ((empty and self.empty_seconds >= confirm_seconds and self.return_seconds >= 3)
                or (self.return_seconds >= 12 and (not fill_known or fill < FULL_HOPPER))):
            self.returning = False
            self.collect_seconds = self.return_seconds = self.shot_hold = 0.0
        if points:
            self.returning,self.return_seconds = True,0.0
            if keys[7]:
                self.shot_hold = max(self.shot_hold,1.0)
        budget = float(np.clip(1-self.turn_seconds/TURN_LIMIT_SECONDS,0,1))
        neutral_face = bool(set(scene.tag_ids) & {19,20})
        only_excluded = scene.fuel_excluded > 0 and not scene.fuel_visible
        goal = "return" if self.returning else "collect"
        if self.returning:
            visible = scene.hub_visible
            x,y,size = scene.hub_x,scene.hub_y,scene.hub_scale
            kind = "blue hub marker"
            why = "Collection time budget reached; seek a scoring attempt" if timed_return else "Load evidence; seek blue hub"
        else:
            dense = scene.fuel_visible and scene.dense_mass >= max(.03,scene.fuel_mass*(1.2 if self.prefer_dense else 1.5))
            self.prefer_dense = bool(dense)
            x,y,size = (scene.dense_x,scene.dense_y,scene.dense_mass) if dense else (scene.fuel_x,scene.fuel_y,scene.fuel_mass)
            visible = scene.fuel_visible
            kind = "dense fuel cluster" if dense else "nearby fuel"
            why = f"Collect guidance; scoring search in {max(0,16-self.collect_seconds):.0f}s"
        actions = {a.name for a in ACTIONS}
        if self.turn_seconds >= 1.6:
            actions = {a.name for a in ACTIONS if not (a.bits()[4] or a.bits()[5])}
        # Keep Space available independently of hopper certainty. Preserve the
        # opponent-zone exclusion and avoid blind forward intake toward piles.
        if only_excluded:
            actions &= {a.name for a in ACTIONS if not a.bits()[6] and not a.bits()[0]}
            if self.returning and visible:
                actions.add("approach_hub")
            why = "Red/unverified fuel excluded; seek landmarks and scoring opportunities"
        if self.hopper_cooldown > 0:
            actions.discard("toggle_hopper")
        if self.camera_cooldown > 0:
            actions.discard('camera_swap')

        def decision(recommended, reason=why, selected_goal=goal, permitted=None):
            allowed = actions if permitted is None else set(permitted)
            if recommended not in allowed:
                recommended = next((name for name in ("retreat","return_left","return_right","coast") if name in allowed),"coast")
            mask = tuple(a.name in allowed for a in ACTIONS)
            return ObjectiveDecision(selected_goal,reason,kind,bool(visible),float(x),float(y),float(size),
                                     mask,ACTION_INDEX[recommended],budget,no_progress)

        if settling:
            return decision("coast","Camera settling; release input","wait",("coast",))
        own_hub_ready = scene.hub_visible and not neutral_face and abs(scene.hub_x) <= .5 and scene.hub_scale >= .1
        blind_probe = not fill_known and self.returning and self.return_seconds >= 4 and not scene.hub_visible and not neutral_face
        probe_due = self.returning or (fill_known and fill >= .15) or (not fill_known and self.elapsed >= 4)
        if (own_hub_ready or blind_probe) and probe_due and (hub_active or not hub_trusted) and self.shot_cooldown <= 0:
            self.shot_hold,self.shot_cooldown,self.shot_fill = 2.0,10.0,fill
        if fill_known and keys[7] and fill < self.shot_fill-.02:
            self.shot_hold,self.shot_fill = max(self.shot_hold,1.0),fill
        if self.shot_hold > 0 and not neutral_face and (hub_active or not hub_trusted):
            kind,visible = "blue hub marker",scene.hub_visible
            x,y,size = scene.hub_x,scene.hub_y,scene.hub_scale
            why = "Scoring probe" if scene.hub_visible else "Exploratory shot; hub pose unknown"
            return decision("shoot",f"{why}: optional Space attempt","shoot")
        at_inactive_hub = self.returning and own_hub_ready and scene.hub_scale >= .5 and hub_trusted and not hub_active
        if at_inactive_hub:
            return decision("coast","At blue hub; await active shift","wait")
        if no_progress >= 8 and not self.recover_seconds and not self.recover_cooldown:
            self.recover_seconds,self.recover_cooldown = 1.0,6.0
            self.recoveries += 1
        if self.recover_seconds:
            escape = ("retreat","return_left","return_right")
            return decision(escape[(self.recoveries-1)%3],"No task progress; translation suggested","recover")
        if not visible:
            if budget <= 0:
                return decision("retreat",why+"; search turn complete, translate to explore")
            search = "aim_left" if int(self.elapsed/6)%2 == 0 else "aim_right"
            return decision(search,why+"; target missing, explore")
        if self.returning:
            return decision(("aim_left" if x < 0 else "aim_right") if abs(x) > .14 else "approach_hub")
        if intake_dry >= 2 and self.hopper_cooldown <= 0:
            return decision("toggle_hopper","Intake stalled; try deployment or another approach")
        if abs(x) > .45:
            return decision("search_left" if x < 0 else "search_right")
        if abs(x) > .16:
            return decision(("collect_left" if x < 0 else "collect_right") if y > .65 else
                            ("collect_forward_left" if x < 0 else "collect_forward_right"))
        return decision("collect_forward")
