# Train the feature agent

The new agent learns collection and scoring from a small, structured vision
observation. It runs its policy on CPU, uses the current fuel detector and hub
AprilTags, and automatically creates an initial policy without human demos.

The local starter is initialized in a randomized practice world. **Practice
results are not proof of real MoSimulator skill.** Live training is needed to
learn the game's actual control, shooting, collision and camera behavior.

## Start

From the project folder in PowerShell:

```powershell
.\train_agent.cmd
```

The command checks dependencies and calibration, resumes a compatible feature
checkpoint or automatically bootstraps one, then opens the HUD. Focus MoSim and
press **]**. The learner joins the current match and begins training. **[** stops and
saves; **]** pauses. Ctrl+C also saves.

Use borderless/windowed mode and start in the calibrated chase-camera view.
Camera swap is **H**, including in the learned 18-action policy. It is a tap,
with a three-second cooldown. Alternate-view surveys automatically return to
calibration after four seconds; alternate-view inventory remains
untrusted. Old 17-action weights and Adam moments expand by named columns/rows,
preserving all previous learned values. The legacy ten-key commands remain separate.

The launcher uses the existing `run_mosim.cmd` interpreter selection. If you want
a particular Python environment, set `$env:MOSIM_PYTHON` to its python.exe path.
Preflight prints the interpreter path when packages are missing. No installation
or sign-in is performed automatically.

Live training continues until stopped. Matches restart three seconds after
confirmed `0:00`, or at a 165-second wall-clock deadline if OCR misses the end. A
checkpoint is saved after each 512 decisions with completed PPO updates (about
102 seconds of control time). Each decision spans two 10 Hz ticks. To run a
bounded session, use `--steps 10000`; `--steps 0` selects continuous training.
Offline training defaults to 10,000 decisions. Only one trainer can run per
project at a time.

## Check or inspect before training

```powershell
.\run_mosim.cmd agent check
.\run_mosim.cmd agent watch
.\run_mosim.cmd agent play --dry-run
```

`agent watch` shows the **revised learning rewards** while you play; it injects no
keys and needs no checkpoint. Restart manually with R, then press ].

`agent play --dry-run` requires a compatible feature checkpoint and displays its
proposed actions without sending keys or match resets. Key-dependent reward
terms in that mode describe proposals, not evidence that actions occurred.

## What the model sees and does

- 95 values per control tick, three ticks of history: a 285-value observation.
- Fixed spatial fuel cells, nearby/dense fuel targets, hub marker position/size,
  optional ally/opponent slots, fill, hub state, time remaining and prior action.
- Current-state features plus explicit change signals feed a small MLP; old
  frames are not blindly averaged into target positions.
- Eighteen purposeful actions: intake movement, search/aim turns, approach,
  retreat, shoot, coast, hopper toggle and return strafes without intake.
- Fine aim is a short nudge followed by release. Scoring never automatically
  combines with random movement keys.

Hub positions come from decoded 36h11 markers on REBUILT hubs. The HUD labels
them as marker anchors, not complete hub outlines or calibrated field pose.
Markers are decoded at up to 1280 pixels wide, with native-resolution retries
when a pair is missing. Current-image optical flow can bridge a decoding gap
for at most 0.6 seconds, with forward/backward and texture checks. Occluded or
expired landmarks remain missing.
The HUD draws fresh crosshairs on the selected fuel target and hub aim point,
and names the proposed purposeful action beside its keys. Optional fresh robot
detections fill the ally/opponent slots when compatible weights are available.

## Objectives and explanations

1. **Collect:** select a substantial cluster when it is larger than the nearby
   isolated fuel target. Nearby disconnected balls are grouped automatically.
2. **Return:** seek the blue hub after 16 seconds of collection, or when a load
   estimate above 65% persists for 0.6 seconds. A confirmed empty basket releases
   this guidance after a minimum three-second attempt. An uncertain or partial
   load gets another collection pass after a twelve-second scoring search.
3. **Shoot / recover:** these remain visible suggestions. The learner chooses
   whether to shoot, turn, collect or recover; suggestions cannot force actions.

Guideline goal/target inputs have 40% strength; direct visual sensor inputs keep
normal scale. Training samples a mixture of 55% learned policy, 40% sensor-based
guidance and 5% legal random exploration. This keeps a collapsed policy trying
navigation and scoring while it learns. Guidance does not recommend hopper
toggles; those remain learned choices with a three-second cooldown. Opponent-fuel
access and camera settling still constrain every part of the distribution.
Sampling and PPO evaluation use the same mixture and mask, so the learner
receives the correct log probability for the action actually executed.

The HUD and JSONL show the objective, target, sensor-based explanation, policy
choices, no-progress timer and remaining turn budget. These are actual
controller rules and sensor facts, not an invented transcript of neural thought.
Hub marker position/size is a navigation proxy; precise alliance-zone boundaries
and shooting distance are not calibrated field pose.

Hopper fullness combines ball-shaped cores, their spread over the basket's
interior and contained yellow coverage. A local cyan-rim search follows the
selected robot's basket when it shifts inside the calibrated view. The black front wall supports the
measurement; scenery at the box edges is excluded. A seven-frame temporal median
rejects brief glimpses, and stale or obscured views preserve the last estimate
as uncertain. This is a camera-specific estimate, not an exact ball count.
The small scoreboard fuel tally is not used as hopper inventory.

## Field minimap and red-zone cost

A minimap sits beside the reward panel. Blue is on the left and red on the
right. It shows your estimated position, chase-view heading, uncertainty outline
and recent position trail. Fresh optional robot detections appear as small
alliance-colored squares; hub markers are fixed map landmarks.

The map projects the lower basket anchor onto the floor under each accepted
camera hypothesis, with an allowance for robot-center offset. It does not use
the camera's position as the robot's position. Heading assumes the calibrated
chase view. Both H camera views support live pose corrections after camera settling.
The pose tracker holds its last given position between visual updates and
uses covariance-weighted corrections when tag estimates return. The last pose
persists across launches in `mosim_pose_state.json`. If no saved or measured pose
exists, the map uses a labeled assumed starting pose. This screen-only setup has
no gyro or wheel-encoder feed, so it holds position instead of inventing motion.
Uncertainty bounds remain estimates rather than calibrated ground truth.

Drag either panel by its title bar. Their independent positions are saved in
`mosim_overlay_layout.json` and restored on the next launch. The rest of the
overlay remains click-through.

The compact reward panel retains total reward, shot credit, clock/watchdog,
shift, hopper, actual controls, advice, per-component tick/totals and three recent
events. Full sensor diagnostics remain in JSONL. The attached field-state box
under the map tracks Blue Alliance, Neutral Field or Red Alliance with boundary
hysteresis and a short crossing confirmation. It holds the last region through
missing tags and labels the estimate source.

The blue agent loses **0.75 reward for a confirmed crossing into the red zone**
and **0.08 per control second while confidently inside**. All accepted position
bounds must be past the boundary margin, with 0.4 seconds of fresh confirmation.
Uncertain boundaries, stale frames and missing poses incur no zone cost. A
localization gap does not create a second entry charge. First localizing inside
the zone incurs occupancy cost without inventing a crossing event.

`Red alliance zone` appears in the reward table, recent events and JSONL log.
The `red_zone_entry_cost`, `red_zone_per_second` and `zone_confirm_seconds`
reward settings can be overridden using the existing `--rewards` option.

The **observed yellow arrow**, rather than the schedule, controls shift-position
costs. Outside blue during ON and inside blue during OFF cost **0.10 reward per
second** after a **two-second travel grace** for each shift change and fresh-zone
confirmation. Missing arrow-view evidence, uncertain boundaries, stale or
repeated poses pause this cost. These settings are `shift_zone_per_second` and
`shift_zone_grace_seconds`. The blue-boundary evidence is independent of the
red-zone exclusion, so neutral field correctly counts as outside blue.

## Opponent-zone fuel exclusion

For the blue agent, red-alliance-zone fuel is excluded before spatial-grid,
nearest-target and largest-cluster calculations. Optional YOLO fuel detections
use the same restriction. Fuel behind the opponent outpost cannot win target
selection merely because it is a large yellow pile.

A fresh pair of decoded field markers and standard field geometry provide
camera/ground-projection hypotheses. The filter accounts for the game's smaller
rendered markers and varied camera focal lengths. A candidate must remain
inside the field and outside the opponent zone, with a boundary margin, across
all accepted hypotheses. A clump crossing the exclusion is discarded whole.
Missing, stale or inconsistent marker geometry leaves fuel unverified; it is
skipped rather than assumed accessible. This can conservatively reject valid
neutral fuel and is not a calibrated guarantee of exact zone membership.

The HUD draws excluded fuel in gray and reports the ignored-region count.
If only excluded fuel is visible, the empty robot searches or backs off without
intaking. Excluded clusters alone do not switch a partial load into return mode.
Blind forward searches toward those
clusters are disabled even while looking for the blue hub.

The access-filter version is recorded in logs and new checkpoint metadata.
The guided-exploration revision has a new objective/reward schema and uses fresh
weights. No manual ball labels are needed.

## Revised rewards

| Component | Behavior |
|---|---|
| Confirmed alliance score | Pays 3.6 per confirmed alliance point (+80%), within two seconds of observed Space, including manual assistance. Uncertain hopper/hub estimates do not veto points. This is correlation, not individual attribution. |
| Fuel potential | Dense guidance from reliable hopper changes, with repayment when potential is consumed or the match ends. |
| Target potential | Small bounded guidance from a visible target, even when hopper readings are unknown. |
| Time cost | Small fixed control-time cost. |
| No task progress | Penalizes lack of new fill, supported unloading, credited points or a closer target. Camera motion alone never clears it. |
| Wasted shot | Known inactive-hub and reliably empty-hopper shots cost a little. |
| Prolonged turning | Costs repeated rotation without a sustained translation break. |
| Sensor / camera | Reserved for known disruptive camera changes. |
| Red alliance zone | Confirmed entry and occupancy costs from reliable field-pose bounds. |
| Wrong zone for shift | -0.10/s outside blue during observed ON, inside blue during observed OFF; two-second travel grace. |

There is no positive reward merely for moving or mashing keys. Uncertain
inventory readings keep existing potential instead of deleting its repayment.
Inventory uncertainty does not disable navigation rewards from visible targets.
Natural match termination settles all remaining potential. Potential shaping
uses a per-tick discount of 0.995; two-tick rewards are discounted consistently
with PPO's decision discount of `0.995 ** 2`.
The smooth-movement bonus is removed from both the feature and legacy paths.
The feature learner also has no cost for changing purposeful actions; holding
one key is not preferred just because changing controls would cost reward.

Per-control-tick components appear in the HUD and JSONL log. PPO receives the
discounted sum over each two-tick decision. The schema is recorded in the log;
raw control totals and discounted decision totals should not be confused.

An optional JSON file can override `RewardConfig` values:

```powershell
.\train_agent.cmd --rewards path/to/reward-settings.json
```

The default launcher automatically builds a new starter when the latest policy
has an old feature/action/reward schema. Previous checkpoint files are preserved.
An explicit `--model` must match; it is never silently replaced.
The v3-to-v4 red-zone extension keeps the same observations, actions and network,
so existing v3 learned weights continue automatically. Their prior reward
settings must match; unrelated schema differences remain incompatible. New
checkpoints record the v4 reward settings and field-map version.

## Pause, focus loss and timing

Input is released immediately on pause/focus loss, independently of inference.
The game may keep running while observation is paused. After a substantial
pause or control gap, training truncates the interrupted transition and resumes
the same match with fresh observations. It does not grade an old action against a much
later image. The match clock includes optimizer time rather than pretending the
simulator stopped during updates.

The actor is briefly held fixed while the value head warms up when transferring
from a practice initialization. Keys are released during PPO updates and before
checkpoint disk writes. Score OCR runs asynchronously on a small cropped region
with a timeout, so it cannot hold down a key while waiting for Tesseract.
The score reader locates the paired blue/red scorecards and crops the large blue
alliance score separately from smaller tallies. The configured score box is a
fallback. The gap between the scorecards supplies a separate `MM:SS` timer OCR
box. The HUD displays its confirmed remaining time. Tight numeric-word OCR reads isolated and multiple digits; the HUD
draws the actual reading box and displays the confirmed number.
Zero is a valid score baseline; a match without points continues normally. If a
baseline is still unreadable, training continues with the other reward terms
and a HUD warning. Unreadable text is never assumed to mean zero. The first
confirmed score establishes a baseline without awarding retroactive points.
Repeated failures save a small `*.hud.png` crop beside the session JSONL log for
diagnosis. `agent watch` uses these same sensors and credits actual human Space.
Capture timestamps are preserved. After three seconds without a fresh capture,
keys are released and the learner waits for a fresh frame or match end. A recovered
capture triggers a fresh observation before another action is graded. Confirmed
`0:00` starts a three-second reset delay that survives OCR gaps or frozen frames.
An independent **165-second watchdog**, counted from arming or the last injected
match restart, resets regardless of OCR. Focus pauses and optimizer time count
toward this deadline; input still requires the armed game to be focused. Each
injected R starts a new deadline. Repeated zero frames cannot create an immediate
second restart. Initial attach/control-gap resets only refresh learner state
until an actual match-end or watchdog deadline.

Fuel overlays target up to 30 Hz, with a processing-time limit that reserves at
least a third of the worker's time. Score OCR checks changed/pending values
promptly and unchanged values every 0.5s (formerly 2s). Its worker uses latest
crops only, has a 10 Hz ceiling and at most 50% processing duty on slow hardware.
Timer OCR remains throttled separately. Tesseract defaults to one thread to
avoid unnecessary parallel-process overhead; an explicit OMP_THREAD_LIMIT is
respected. Feature decisions remain two 10 Hz control ticks each.

## Checkpoints and progress

Feature-agent artifacts live under `crosim_agent_models/live`; offline practice
uses `crosim_agent_models/practice`. Both are excluded from Git. Old v4 pixel
models are never overwritten or loaded as feature policies.

Weights and schema metadata are written before the atomic `latest.json` pointer
is updated. Integrity hashes catch incomplete/corrupted weights. Feature names,
policy architecture, action vocabulary, repeat timing and reward settings are
checked before loading. A `status.json` file reports the current phase and latest
episode results. Checkpoints preserve the domain in which they were trained.

An interrupted launch that never starts collecting decisions does not relabel a
practice starter as live-trained. Checkpoints with non-finite weights are refused.

To reset all active project training data:

```powershell
.\run_mosim.cmd agent reset
```

This moves checkpoints, demonstrations, normalizers and reward logs into
`training_archive/reset_*`, outside all resume paths. It preserves source,
calibration and vision weights. The next training launch automatically creates
a fresh starter. Reset is refused while another trainer holds the project lock.

## Offline work without the game

```powershell
.\run_mosim.cmd agent smoke
.\run_mosim.cmd agent bootstrap
.\train_agent.cmd --offline --steps 20000
.\run_mosim.cmd agent evaluate --episodes 12
```

These commands do not open native capture or send game input. Bootstrap uses a
local heuristic teacher, behavior cloning and automatic dataset aggregation on
states reached by the learner. Offline PPO then practices in a simplified,
randomized world. The game physics and renderer are not reproduced.
Initialization includes intermittent inventory dropouts, and cloned action
logits are softened before PPO so the starter retains exploration.

Practice candidates are compared with their starting policy on development
validation seeds. A regressing candidate is saved for inspection without being
promoted to the latest pointer. Real-game validation is still required before
calling a policy good at MoSimulator.

Evaluation uses deterministic actions on 12 repeatable practice seeds by default
and writes `evaluation.json` beside the selected checkpoint folder. It reports
mean/median score, the number of episodes that scored, and per-episode results.
Use `--practice` to load the practice folder, `--seed` to choose the first seed,
or `--model` for an explicit compatible checkpoint. It never controls the game.


Score-first revision: every confirmed alliance score increase pays 3.6 per point, without a shooting-window filter (including teammate points). OCR requires two agreeing numeric layouts and temporal confirmation; pending final score reads are drained before a live reset. Hopper potential is 0.20 and fuel-target potential 0.05. Turning costs 0.60/s after grace; continuous turns are masked after 1.6s until 0.6s translation. Estimated blue-bump contact costs 0.80/s after 1.5s crossing grace; red hub/bump/trench proximity costs 0.40/s after confirmation. These contact proxies require fresh tight pose bounds and nominal FIRST geometry; they are not collision telemetry. Existing model weights continue unchanged.


Minimap update: partial camera hypotheses may update the displayed pose without enabling zone costs. Moving-pose smoothing has a minimum correction gain; three consistent fresh observations can reacquire a rejected jump. Large visible fuel clusters are projected onto the field, shown with confidence-dependent true alpha transparency, and expire after two seconds. Confidence reflects camera-hypothesis agreement, not a calibrated probability. Red-zone fuel remains excluded from collection targets; visual map clusters include both alliances.


Bidirectional pose update: normal and reversed H views both localize after the brief camera-settling period. Switching views clears old tracked tag corners but retains the robot pose/trail and covariance. Reversed camera bearing is corrected by 180 degrees to preserve robot heading. Joint fitting uses tags across hub/field/alliance groups when reprojection checks succeed, with pair fallback otherwise. Remaining camera ambiguity and contact-anchor uncertainty stay visible in the pose bounds; no artificial reduction of confidence bounds is applied.
