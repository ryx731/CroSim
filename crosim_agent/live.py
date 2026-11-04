"""Screen-only live feature environment with the existing capture/input guard."""
from dataclasses import asdict, replace
import math
import time

from gymnasium import spaces
import numpy as np

from mosim_env import MoSimEnv, CFG, normalized_fill, crop_frac
from .actions import ACTIONS, KEYS, action_bits, tick_bits
from .features import FeatureExtractor, FeatureHistory, OBS_DIM, Scene, pack_features
from .proxy import ACTION_REPEAT, CONTROL_DT
from .rewards import CycleReward, RewardConfig, RewardState, LABELS
from .sensors import AsyncScoreReader
from .timer import MatchClock,HARD_MATCH_SECONDS
from .objectives import ObjectiveController
from mosim_zones import FUEL_ACCESS_VERSION
from mosim_field import FIELD_MAP_VERSION
from .hud import HudPanelExtractor, HudReader, InventoryTracker, ShiftArrowTracker, HUD_SENSORS_VERSION


class EpisodeInterrupted(Exception):
    """A focus pause or a long control gap makes the pending transition invalid."""


class LiveFeatureEnv(MoSimEnv):
    reward_labels = LABELS

    def __init__(self, debug_options=None, reward_config=None, dry_run=False, inject_inputs=True):
        self.feature_history = FeatureHistory()
        self.extractor = FeatureExtractor(CFG,persist_pose=True)
        self.scene = Scene()
        self.reward_config = reward_config or RewardConfig()
        self.reward_engine = CycleReward(self.reward_config)
        self.objectives = ObjectiveController()
        self.decision = None
        self.inventory = InventoryTracker(CFG)
        self.hud_panel = HudPanelExtractor(CFG)
        self.shift_arrow = ShiftArrowTracker()
        self._physical_space = False
        self._async_score = None
        self._episode_active = False
        self._interrupted = False
        self.match_clock = MatchClock()
        self._live_start = None
        self._capture_stalled = False
        self._last_control_end = None
        self._current_macro = 0
        self._manual_signature = None
        self._dry_intake_seconds = 0.0
        self._last_fill = 0.0
        self._last_motion = 0.0
        self._camera_was_down = False
        self._score_recent = 0.0
        self._feature_ready = False
        super().__init__(verbose=False, debug_options=debug_options, dry_run=dry_run, inject_inputs=inject_inputs)
        try:
            self.action_space = spaces.Discrete(len(ACTIONS))
            self.observation_space = spaces.Box(-1, 1, shape=(OBS_DIM,), dtype=np.float32)
            diagnostic = self._reward_log.path.with_suffix(".hud.png") if self._reward_log else None
            self._async_score = AsyncScoreReader(lambda: HudReader(diagnostic),
                                                 frame_transform=self.hud_panel,
                                                 crop_reader=True)
            self._score_reader = self._async_score
            if self._reward_log:
                from .checkpoints import schema
                self._reward_log.write({"type": "agent_schema", "spec": schema(self.reward_config),
                                        "fuel_access_version": FUEL_ACCESS_VERSION,
                                        "field_map_version":FIELD_MAP_VERSION,
                                        "hud_sensors_version": HUD_SENSORS_VERSION})
        except BaseException:
            self.close()
            raise
        if dry_run:
            print("[agent] DRY RUN: displays proposed actions; no game keys or resets are sent.")

    def _elapsed(self):
        # The simulator continues during optimizer updates and focus pauses.
        # Do not subtract those seconds from its hub schedule or match length.
        return max(0.0, time.perf_counter()-self._live_start) if self._live_start is not None else 0.0

    def _account_for_lost_time(self):
        now = time.perf_counter()
        if self._inject_inputs and self._episode_active and self._last_control_end is not None and now-self._last_control_end > 0.75:
            self._interrupted = True
        self._last_step_wall = now

    def _wait_ready(self):
        started = time.perf_counter()
        super()._wait_ready()
        if self._inject_inputs and self._episode_active and time.perf_counter()-started > 0.25:
            self._interrupted = True

    def _apply_action(self, action):
        if self._interrupted:
            raise EpisodeInterrupted
        if self._capture_stalled or self._episode_has_ended(self._elapsed()):
            action[:] = 0
        return super()._apply_action(action)

    def _hard_timeout(self):
        return self._live_start is not None and self._elapsed() >= HARD_MATCH_SECONDS

    def _restart_due(self):
        return self.match_clock.can_restart() or self._hard_timeout()

    def _tap_guarded(self, key, hold=None):
        if key == "r" and not self._restart_due():
            return False
        started = time.perf_counter()
        sent = super()._tap_guarded(key, hold)
        if sent and key == "r":
            self._live_start = started
        return sent

    def _sync_match_timer(self):
        snapshot = getattr(self._async_score,"snapshot",None)
        self.match_clock.observe(getattr(snapshot,"timer_seconds",None),getattr(snapshot,"timer_at",0.0))

    def _reset_game(self):
        self._sync_match_timer()
        if self._inject_inputs and self.match_clock.ended:
            self.release_all_keys()
            while self.match_clock.ended and not self._restart_due():
                self._wait_ready()
                frame = self._grab_focused()
                self._score_reader.update(frame,time.perf_counter())
                self._sync_match_timer()
                self._publish_debug(frame,self._elapsed(),0,False)
                time.sleep(.05)
        if self._inject_inputs and self._restart_due() and self._tap_guarded("r"):
            self.match_clock.restarted()
            self._capture_stalled = False
            time.sleep(.5)

    def _episode_has_ended(self,elapsed):
        return self.match_clock.ended or (self._live_start is not None and elapsed >= HARD_MATCH_SECONDS)

    def _remaining_seconds(self):
        remaining = self.match_clock.remaining()
        deadline = max(0,HARD_MATCH_SECONDS-self._elapsed())
        return min(remaining,deadline) if remaining is not None else deadline

    def _await_capture_or_match_end(self):
        """A frozen end screen must reach the watchdog, rather than aborting."""
        if not self._inject_inputs or time.perf_counter()-getattr(self,"_latest_capture_at",time.perf_counter()) <= 3:
            return
        self.release_all_keys()
        self._capture_stalled = True
        while not self._episode_has_ended(self._elapsed()):
            self._wait_ready()
            frame = self._grab_focused()
            self._score_reader.update(frame,time.perf_counter())
            self._sync_match_timer()
            self._publish_debug(frame,self._elapsed(),0,False)
            if time.perf_counter()-self._latest_capture_at <= .3:
                self._capture_stalled = False
                # Reobserve before selecting another action after the gap.
                raise EpisodeInterrupted
            time.sleep(.05)

    def _on_frame(self, frame, planes, elapsed):
        self._sync_match_timer()
        if self._interrupted:
            raise EpisodeInterrupted
        self._capture_stalled = self._inject_inputs and time.perf_counter()-getattr(self,"_latest_capture_at",time.perf_counter()) > 3
        if self._capture_stalled:
            self.release_all_keys()
        self.scene = self._analyze_frame(frame)

    def _observed_shift(self):
        if (self._view_change_steps
                or time.perf_counter()-getattr(self,"_latest_capture_at",0) > .3
                or not self.hud_panel.status.startswith("auto blue scorecard")):
            return None
        return self.shift_arrow.observed()

    def _analyze_frame(self, frame):
        self.extractor.config["hopper_region"] = self.inventory.read_region
        now = time.perf_counter()
        detections = []
        if self._vision:
            snapshot = self._vision.latest()
            age = now-float(snapshot.get("captured_at", 0))
            if 0 <= age <= snapshot.get("max_age", 0.8):
                detections = [{**box,"captured_at":box.get("captured_at") or snapshot.get("captured_at",0)}
                              for box in snapshot.get("boxes", [])]
        captured = getattr(self, "_latest_capture_at", now)
        return self.extractor.analyze(frame,captured_at=captured,detections=detections,
                                      pose_enabled=not self._view_change_steps,
                                      reversed_view=bool(self._view_parity))

    def _reward_state(self, *, action=None, motion=0, elapsed=0, settling=False):
        settling = settling or self._capture_stalled
        action = action_bits(self._current_macro) if action is None else action
        fill_valid = self.inventory.snapshot.known and not settling
        field_map = self.scene.field_map
        pose = field_map.get("pose") or {}
        from mosim_contacts import structure_contact
        blue_bump,red_structure = structure_contact(field_map)
        return RewardState(fill=normalized_fill(self._fill_ema), fill_valid=fill_valid,
                           hub_active=bool(self._hub_active), hub_trusted=self._hub.source != "assumed",
                           hub_visible=self.scene.hub_visible, hub_x=self.scene.hub_x,
                           hub_scale=self.scene.hub_scale, fuel_visible=self.scene.fuel_visible,
                           fuel_x=self.scene.fuel_x, fuel_y=self.scene.fuel_y, fuel_mass=self.scene.fuel_mass,
                           motion=motion, shooting=bool(action[7]) or self._physical_space,
                           movement_requested=bool(np.any(action[:6])),
                           elapsed=elapsed,
                           action=self._current_macro if self._manual_signature is None else self._manual_signature,
                           settling=settling, goal=self.decision.goal if self.decision else "collect",
                           target_visible=bool(self.decision and self.decision.target_visible),
                           target_x=self.decision.target_x if self.decision else 0.0,
                           target_y=self.decision.target_y if self.decision else 0.0,
                           target_scale=self.decision.target_scale if self.decision else 0.0,
                           rotating=bool(action[4] or action[5]), translating=bool(np.any(action[:4])),
                           opponent_zone=field_map.get("opponent_zone") if field_map.get("pose_current") else None,
                           zone_sample_at=pose.get("captured_at"),
                           blue_zone=field_map.get("blue_zone") if field_map.get("pose_current") else None,
                           shift_on=self._observed_shift(),blue_bump=blue_bump,red_structure=red_structure)

    def _update_shaping(self, frame, hub_active, rebaseline):
        captured = getattr(self,"_latest_capture_at",time.perf_counter())
        self.shift_arrow.update(frame,self.hud_panel.arrow_region,captured_at=captured)
        shift = self._observed_shift()
        if shift is not None:
            self._hub_active,self._hub.source = shift,"arrow"
        inventory = self.inventory.update(frame,captured_at=getattr(self,"_latest_capture_at",None),settling=rebaseline or bool(self._view_parity))
        # Keep the shared debug API's normalized_fill convention, but never
        # feed raw yellow coverage back into feature-agent inventory.
        from mosim_env import FUEL_HOLD_THRESHOLD_FRACTION
        self._fill_ema = inventory.fill*FUEL_HOLD_THRESHOLD_FRACTION
        return 0.0

    def _reward_terms(self, components, reasons, *, frame, action, elapsed, points, motion, moved, view_changing):
        fill = normalized_fill(self._fill_ema)
        if bool(action[6]) and self.scene.fuel_visible and self.scene.fuel_y > 0.65 and fill-self._last_fill < 0.005:
            self._dry_intake_seconds += CONTROL_DT
        else:
            self._dry_intake_seconds = 0.0
        self._last_fill = fill
        self._last_motion = motion
        self._score_recent = float(points)
        self._physical_space = False
        user = getattr(self._guard,"user",None)
        if self._inject_inputs and user is not None:
            self._physical_space = bool(user.GetAsyncKeyState(0x20) & 0x8000)
        previous_goal = self.decision.goal if self.decision else None
        self._update_objective(action, dt=CONTROL_DT, settling=view_changing,points=points)
        if self._reward_log and self.decision.goal != previous_goal:
            self._reward_log.write({"type": "objective", "elapsed": elapsed, **self.decision.snapshot()})
        state = self._reward_state(action=action, motion=motion, elapsed=elapsed, settling=view_changing)
        state = replace(state, camera_changed=bool(action[9]) and not self._camera_was_down)
        self._camera_was_down = bool(action[9])
        self._sync_match_timer()
        parts, reasons, diagnostics = self.reward_engine.update(state, points, dt=CONTROL_DT,
                                                               terminal=self._episode_has_ended(elapsed))
        diagnostics["objective"] = self.decision.snapshot()
        diagnostics["fuel_zones"] = self.scene.zone_info
        diagnostics["field_map"] = self.scene.field_map
        diagnostics["inventory"] = asdict(self.inventory.snapshot)
        diagnostics["observed_space"] = bool(action[7]) or self._physical_space
        diagnostics["score_confirmed"] = self._score_reader.confirmed
        diagnostics["score_ocr_error"] = getattr(getattr(self._async_score,"snapshot",None),"error",None)
        diagnostics["hud_source"] = self.hud_panel.status
        diagnostics["timer_seconds"] = self.match_clock.remaining()
        diagnostics["timer_ocr_error"] = getattr(getattr(self._async_score,"snapshot",None),"timer_error",None)
        diagnostics["match_end_confirmed"] = self.match_clock.ended
        diagnostics["watchdog_expired"] = self._hard_timeout()
        return parts, reasons, diagnostics

    def _update_objective(self, keys=None, *, dt=0.0, settling=False,points=0):
        self.decision = self.objectives.update(
            self.scene, fill=normalized_fill(self._fill_ema), fill_known=self.inventory.snapshot.known and not settling,
            hub_active=self._hub_active, hub_trusted=self._hub.source != "assumed",
            time_remaining=self._remaining_seconds(),
            no_progress=self.reward_engine.idle_seconds, intake_dry=self._dry_intake_seconds, keys=keys, dt=dt,
            settling=settling,confirm_seconds=.6,points=points)
        # H can survey the alternate view, then restore the calibrated view.
        # This prevents indefinite camera-based evasion of pose/contact costs.
        if self._view_parity and self._steps_since_swap >= 40:
            from .actions import ACTION_INDEX
            index=ACTION_INDEX['camera_swap']
            self.decision=replace(self.decision,goal='recover',reason='Camera survey complete; restore calibrated view',
                                  allowed=tuple(i==index for i in range(len(ACTIONS))),recommended=index)

    def _feature_frame(self):
        return pack_features(self.scene, fill=normalized_fill(self._fill_ema),
                             fill_known=self.inventory.snapshot.known,
                             hub_active=self._hub_active, hub_trusted=self._hub.source != "assumed",
                             hub_change=self._hub_change_in, time_left=min(1.0,self._remaining_seconds()/self.episode_seconds),
                             motion=self._last_motion, settling=bool(self._view_change_steps or self._view_parity),
                             stuck=self.reward_engine.idle_seconds, intake_dry=self._dry_intake_seconds,
                             score_recent=self._score_recent, previous_action=self._current_macro, objective=self.decision)

    def _get_obs(self):
        if self._feature_ready:
            self.feature_history.push(self._feature_frame())
        return self.feature_history.observation()

    def reset(self, *, seed=None, options=None):
        self._episode_active = self._interrupted = self._feature_ready = False
        self._last_control_end = None
        self._current_macro = 0
        self._manual_signature = None
        self._camera_was_down = False
        self._dry_intake_seconds = self._last_motion = self._score_recent = 0.0
        self.extractor.reset()
        self.objectives.reset()
        self.decision = None
        self.inventory.reset()
        self.shift_arrow.reset()
        self._physical_space = False
        # Arm before starting the wall clock; no game input is sent while waiting.
        self._guard.wait()
        if self._live_start is None:
            self._live_start = time.perf_counter()
        super().reset(seed=seed, options=options)
        self.scene = self._analyze_frame(self._latest_frame)
        self._update_shaping(self._latest_frame,self._hub_active,False)
        self._last_fill = normalized_fill(self._fill_ema)
        self._update_objective()
        self.reward_engine.reset(self._reward_state(elapsed=self._elapsed()))
        self._feature_ready = self._episode_active = True
        self._last_control_end = time.perf_counter()
        observation = self.feature_history.reset(self._feature_frame())
        self._score_reader.update(self._latest_frame, time.perf_counter())
        self._publish_debug(self._latest_frame, self._elapsed(), 0.0, False)
        return observation, {"domain": "live_screen", "hub_landmark_visible": self.scene.hub_visible}

    def _debug_state(self):
        warning = None
        if self._async_score and self._async_score.confirmed is None and self._async_score.ocr_failures >= 5:
            warning = "OCR pending: training continues; score credit waits"
        reader_error = getattr(getattr(self._async_score,"snapshot",None),"error",None)
        if reader_error:
            warning = reader_error
        inventory = self.inventory.snapshot
        if not inventory.known and not warning:
            warning = "Hopper view uncertain; load goal held stable"
        remaining = 0 if self.match_clock.ended else self.match_clock.remaining()
        reset_delay = max(0,self.match_clock.zero_hold-(time.perf_counter()-self.match_clock.zero_since)) if self.match_clock.ended else 0
        timer_status = "2:45 WATCHDOG" if self._hard_timeout() else f"RESET IN {reset_delay:.1f}s" if reset_delay > 0 else "MATCH ENDED" if self.match_clock.ended else "MATCH ACTIVE" if remaining is not None else "WATCHDOG ACTIVE"
        if remaining is None and not warning:
            warning = "Timer OCR pending; 2:45 reset watchdog active"
        if self._capture_stalled:
            warning = "Capture frozen; inputs released; reset watchdog active"
        shift = self._observed_shift()
        targets = []
        if self.decision and self.decision.target_visible:
            targets.append({"x": (self.decision.target_x+1)/2, "y": self.decision.target_y,
                            "label": self.decision.goal.upper()+": "+self.decision.target_kind,
                            "color": "#f5bc58" if self.decision.goal == "collect" else "#55d9f5"})
        regions = [{"box":self.inventory.read_region,"label":f"Hopper: {inventory.fill:.0%} estimate / {inventory.visual_balls} ball cores"},
                   {"box":self.hud_panel.score_region,"label":f"Score OCR: {self._score_reader.confirmed}"},
                   {"box":self.hud_panel.timer_region,"label":f"Timer OCR: {remaining//60}:{remaining%60:02d}" if remaining is not None else "Timer OCR: pending"},
                   {"box":self.hud_panel.arrow_region or CFG.hub_region,"label":f"Shift arrow: {'ON' if shift is True else 'OFF' if shift is False else 'pending'}"}]
        keys = [key for key,down in zip(KEYS,self._prev_action) if down]
        if self._physical_space and "space" not in keys:
            keys.append("space (observed)")
        return {"credited_points": self.reward_engine.credited_points,
                "human_demo_samples":getattr(self,'_human_demo_samples',None),
                "uncredited_points": self.reward_engine.uncredited_points,
                "policy_action": ACTIONS[self._current_macro].name, "sensor_warning": warning,
                "agent_targets": targets, "feature_captured_at": self.scene.captured_at,
                "objective": self.decision.snapshot() if self.decision else None,
                "fuel_zones": self.scene.zone_info,
                "field_map":{**self.scene.field_map,"shift_on":shift,
                             "shift_zone_per_second":self.reward_config.shift_zone_per_second,
                             "zone_cost":{"entry":self.reward_config.red_zone_entry_cost,
                             "per_second":self.reward_config.red_zone_per_second}},
                "inventory_known":inventory.known,"inventory_fill":inventory.fill,"inventory_cores":inventory.visual_balls,
                "score_confirmed":self._score_reader.confirmed,"keys":keys,"regions":regions,
                "timer_seconds":remaining,"timer_status":timer_status,"shift_on":shift,
                "watchdog_remaining":max(0,HARD_MATCH_SECONDS-self._elapsed())}

    def step(self, action):
        # SB3 returns a numpy scalar for Discrete policies.
        value = np.asarray(action)
        if value.size != 1 or not np.isfinite(value).all() or float(value.item()) != int(value.item()):
            raise ValueError("Feature agent expects one integer action")
        macro = int(value.item())
        action_bits(macro)
        self._current_macro = macro
        self._manual_signature = None
        components = dict.fromkeys(LABELS, 0.0)
        discount = 1.0
        info = {}
        terminal = truncated = False
        observation = self.feature_history.observation()
        try:
            self._await_capture_or_match_end()
            for tick in range(ACTION_REPEAT):
                observation, reward, _, match_over, info = super().step(tick_bits(macro, tick))
                for name, amount in info["reward_components"].items():
                    components[name] += discount*amount
                discount *= self.reward_config.gamma
                self._last_control_end = time.perf_counter()
                if match_over:
                    # Allow two asynchronous confirmations of the final score
                    # before reset invalidates the OCR worker's generation.
                    if self._inject_inputs and hasattr(self._async_score,"thread"):
                        self.release_all_keys()
                        deadline = time.perf_counter()+1.2
                        while time.perf_counter() < deadline:
                            self._wait_ready()
                            extra = self._score_reader.update(self._latest_frame,time.perf_counter())
                            if extra:
                                amount = extra*self.reward_config.score_per_point
                                components["score"] += discount*amount
                                self.episode_points += extra
                                self.reward_engine.credited_points += extra
                                final_parts = dict.fromkeys(LABELS,0.0)
                                final_parts["score"] = amount
                                self._ledger.record(final_parts,self._elapsed(),{"score":"Final confirmed score before reset"})
                                if self._reward_log:
                                    self._reward_log.write({"type":"final_score","points":extra,"reward":amount})
                            time.sleep(.05)
                        self._publish_debug(self._latest_frame,self._elapsed(),0,False)
                    terminal = True  # Natural finite match end, with time-left in the state.
                    self._episode_active = False
                    self.release_all_keys()
                    reason = "2:45 wall-clock watchdog" if self._hard_timeout() else "OCR confirmed match clock at 0:00"
                    info.update(auto_restart=True,reason=reason)
                    break
        except EpisodeInterrupted:
            self.release_all_keys()
            self._episode_active = self._interrupted = False
            truncated = True
            info = {"interrupted": True, "reason": "focus pause or control gap; reset before grading another action"}
        info.update(domain="live_screen", action_name=ACTIONS[macro].name,
                    reward_components=components, decision_discount=discount,
                    alliance_points_total=self.episode_points,
                    credited_points=self.reward_engine.credited_points,
                    uncredited_points=self.reward_engine.uncredited_points)
        info["objective"] = self.decision.snapshot() if self.decision else None
        return observation, float(math.fsum(components.values())), terminal, truncated, info

    def watch_step(self, keys):
        """Grade actual human keys with the revised rewards, without injecting."""
        if self._inject_inputs:
            raise RuntimeError("watch_step requires inject_inputs=False")
        keys = np.asarray(keys)
        if keys.shape != (10,) or not np.isin(keys, (0, 1)).all():
            raise ValueError("Expected ten physical key bits")
        keys = keys.astype(np.int8)
        self._current_macro = min(range(len(ACTIONS)), key=lambda index: int(np.count_nonzero(action_bits(index) != keys)))
        self._manual_signature = 100+sum(int(bit)*(1 << index) for index, bit in enumerate(keys))
        observation, reward, _, over, info = super().step(keys)
        self._last_control_end = time.perf_counter()
        return observation, reward, bool(over), False, info

    def close(self):
        try:
            super().close()  # Release input before waiting for an active OCR call.
        finally:
            if self._async_score:
                self._async_score.close()
