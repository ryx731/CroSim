# Learning design and evidence

The immediate objective is to learn useful collection-and-score cycles quickly,
with errors visible in the HUD and startup requiring one command plus the global
arming key. Fine strategy optimization is a later stage.

## Compact policy and stable observations

The old pixel CNN and independent binary key outputs are preserved for existing
models. The new path uses a small CPU MLP, fixed spatial slots and purposeful
discrete actions. The objective revision adds goal/target slots and legal-action
bits to the current-plus-deltas input. It remains a small two-layer CPU network.
The current policy/value network has 22,290 parameters. A local CPU benchmark
measured a 0.223 ms median decision over 120 predictions; this is not a live
capture-to-action latency guarantee.

Structured features are normalized to bounded ranges. Current target positions
are supplied directly; history is exposed as explicit changes rather than
forcing the network to average outdated coordinates. A three-tick observation
keeps short temporal context without an additional recurrent dependency.

Stable-Baselines3 supports a CPU MLP and discrete actions, and recommends checking
custom environments and using simple temporal stacking before adding recurrent
complexity. See [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
and [RL implementation guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html).

## Automatic initialization

A deterministic heuristic supplies labels from observable features in a small
randomized practice world. Initial cloning is followed by three data-aggregation
rounds on the learner's own visited states, mixing teacher and learner control.
No user demonstrations are required for this initializer.

The reason for aggregation is distribution shift: a model can agree with a
teacher on recorded states while failing in its own closed-loop trajectories.
The first development seed demonstrated exactly that problem (81.5% action
agreement but zero score on three practice validation episodes). The earlier
v1 state-plus-deltas architecture and aggregated data raised action agreement to
95.5%; subsequent proxy PPO improved performance on a separate seed sweep.
These observations concern the proxy only.

Historical v1 comparison on 12 fixed practice seeds (9001–9012); these weights
are not the objective revision's initializer:

| Initializer | Mean proxy points | Episodes with points |
|---|---:|---:|
| Cloning plus aggregation | 50.7 | 10 / 12 |
| Plus 100,096 PPO decisions | 62.7 | 10 / 12 |
| Plus 300,160 total PPO decisions | 57.1 | 9 / 12 |

The 100,096-step candidate was selected before checking 20 fresh seeds
(13001–13020). It averaged 64.75 proxy points, scoring in 19 / 20 episodes.
The longer run was retained locally but did not become the v1 live initializer.
This demonstrates why the default evaluation covers more than one good run;
it does not measure sim-to-real transfer. Local weights/results are excluded
from Git, and a fresh checkout builds its own initializer automatically.

Dataset aggregation follows the central idea of [Ross, Gordon and Bagnell](https://proceedings.mlr.press/v15/ross11a.html).
Our teacher is a heuristic, not an optimal expert, so the method's theoretical
guarantees should not be claimed for real MoSimulator performance.

The objective revision's initializer averaged 28.17 practice points on the
12 development seeds. After 30,080 proxy PPO decisions it averaged 31.42,
scoring in 9 / 12 episodes. That candidate was selected before a fresh 20-seed
check (14001–14020): mean 23.9, scoring in 15 / 20. These proxy-only results
remain modest. Objectives constrain the available actions, often to one choice,
so high cloning agreement is not evidence of independent strategic reasoning.
The earlier v1 numbers do not validate this changed controller or real-game skill.

## Reward integrity

The objective revision removes the movement bonus from legacy rewards and
removes control-change cost from the feature learner. Its no-progress timer
uses inventory high-water marks, supported unloading, points and closer-target
measurements. Large camera motion does not clear it. Consecutive rotation also
has a cost and must be interrupted by sustained translation.

The current controller supplies advisory collect/return targets and short
scoring/recovery probes. Collection and return retain several movement choices,
and Space is available independently of hopper certainty. Time budgets prevent
indefinite collection while noisy inventory stays unknown. Masks enforce fuel
access, full-basket and rotation limits and are observation data. They are
used in the actor distribution for sampling, prediction and minibatch log
probabilities. Actions are never substituted after sampling. This is informed
by [Huang and Ontañón's action-masking analysis](https://arxiv.org/abs/2006.14171).
Our controller's restrictions are task scaffolding, not claims that every
excluded action is invalid under simulator rules.

The HUD's explanation is an audit of controller inputs and rules. It does not
claim access to a neural model's internal reasoning.

The latest guided-exploration revision was prompted by a live log with uncertain
inventory on 729 of 1,003 ticks, rejected/unverified fuel on 861 ticks, and no AI
Space presses. The old state restrictions prevented useful exploration. The new
initializer is built from fresh examples with inventory dropouts and softened
cloned logits; old live/practice weights are archived outside resume paths.
Historical v1/v2 practice scores above do not validate this changed controller.
Fresh starters are checked on twelve development practice episodes. This is
an initializer check, with no real-game training or performance claim; results
are recorded beside each generated checkpoint. The starter's recorded live PPO
timestep count is zero.

Most early guidance is the difference of bounded state potentials. The same
discount is used in shaping, action-repeat aggregation and PPO. Natural terminal
states have zero potential; unknown measurements preserve prior potential.
Tests explicitly check that fill/drain cycles and sensor gaps cannot accumulate
discounted shaping reward without external progress.

The basis for this construction is [Ng, Harada and Russell's potential-based
reward shaping](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).
That result does not make our noisy screen measurements true simulator state,
or turn the other task costs into policy-invariant transformations.

Alliance-score OCR cannot identify which robot caused a point. The reward
therefore gates score credit on a recent observed Space press, including human
assistance, and logs credited and uncredited points separately. A confirmed score
increase outweighs uncertain hopper or hub-color estimates. Coincident teammate
scores can still pass the gate. They are labeled shoot-correlated, not individual
ground truth.

Navigation potential and visible-target progress remain available when inventory
is unknown; only inventory potential is frozen. The first reliable inventory
baseline never overwrites existing target-potential debt.

The scorecard locator separates the large alliance score from small tallies.
The hopper sensor instead checks the basket interior, ball cores, spatial
occupancy and front-wall support, with a temporal median and full/empty dwell
times. A scoreboard tally is not assumed to be robot inventory. These spatial
heuristics were checked on synthetic cases and a saved game image; they still
need validation during real driving and unloading.

The reward does not pay just for moving, increasing the number of detected balls,
or changing camera views. No exact count of overlapping field fuel is required.

## Zero-label hub landmarks

The screenshot supplied during development contains decodable tags 25 and 26 on
the blue hub. A 36h11 OpenCV detector reads them without trained weights.
Blue hub IDs are 18–21 and 24–27; red hub IDs are 2–5 and 8–11, based on the
[official 2026 field drawings](https://firstfrc.blob.core.windows.net/frc2026/FieldAssets/2026-field-dimension-dwgs.pdf).
Trench, tower and outpost tags are excluded from hub classification.

Policy hub features use marker position and apparent size. A tag rectangle is
not a whole-hub detection. A separate minimap now exposes an approximate field
pose from the camera hypotheses, with displayed uncertainty; it is not calibrated
ground truth. Other robots still need optional learned detections.

The opponent-fuel gate separately fits conservative camera hypotheses using
marker-pair spacing and the [WPILib 2026 welded field layout](https://raw.githubusercontent.com/wpilibsuite/allwpilib/v2026.2.1/apriltag/src/main/native/resources/edu/wpi/first/apriltag/2026-rebuilt-welded.json).
Zone depth comes from the [official field/zone definitions](https://firstfrc.blob.core.windows.net/frc2026/Manual/HTML/2026GameManual.htm).
The archived MoSim frame has rotated textures and smaller rendered marker
squares than the official target, so texture-relative corners or a fixed
official-size single-tag range estimate are insufficient. Upright square
geometry, a size/focal-length hypothesis set, a boundary margin and unanimous
eligibility are used instead. Ambiguous candidates are discarded. Synthetic
cameras exercise red/neutral separation, opponent outpost floor and chute fuel,
stale geometry, component boundaries and the optional learned-detector path.
This remains an estimated restriction that can discard valid neutral fuel.

The robot-map point is a projected basket-floor anchor with a center allowance,
rather than the estimated camera location. A conservative red-zone cost requires
the entire position range to lie inside the opponent zone, with fresh temporal
confirmation. Pose gaps freeze crossing history and pause charges. This cost is
additive task guidance; the potential-shaping invariance claim does not apply to
it. The observation/action/network schema stays fixed, preserving existing
learned weights through the explicitly supported v3 reward extension.

## Timing, interruption and persistence

Match end is a natural finite-horizon termination, with remaining time included
in the observation. Focus/control interruptions are truncations. This distinction
follows [Gymnasium's termination/truncation guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).
Only observed control ticks are graded across a partial interrupted decision.

PPO updates are short on the small CPU network, and keys are released during
updates. The match schedule uses elapsed real time because the game keeps
running. Large gaps end the pending training segment rather than grading stale
observations. A new match starts after the session is re-armed.

Checkpoint files have explicit architecture/feature/action/reward fingerprints,
weight hashes and atomic latest pointers. Offline and live domains are kept
separate. Code checks observations, rewards and model weights for non-finite
values; see [SB3's failure-detection guidance](https://stable-baselines3.readthedocs.io/en/master/guide/checking_nan.html).

## What remains unverified

No live game input or capture was used for feature-agent development in the
restricted coding session. Native live integration is exercised with fake
capture, score and input drivers. A real-game run is required to validate
control direction/rates, shooting behavior, calibration, pause/reset behavior,
and actual learning speed. Practice score is never reported as MoSimulator score.
