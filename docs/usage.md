# Usage and troubleshooting

## Calibration

Run `python mosim_env.py setup` with the game available. The guided flow captures
reference frames using global hotkeys and voice prompts, then lets you choose
regions on a frozen screenshot. Calibration writes `mosim_config.json` locally.

The included `mosim_config.example.json` documents the settings. Its regions are
starting defaults; use setup for your resolution, aspect ratio and HUD layout.
Tesseract is discovered from the configured path, PATH or standard Windows install
locations. `python mosim_env.py doctor` checks the environment and existing config.

To redo selected steps from a saved screenshot:

```powershell
python mosim_env.py setup --only regions,ocr --image setup_reference.png
```

## Human watch mode

```powershell
python mosim_env.py watch
```

Focus the game, restart with R and press ]. Pause/resume with ]; stop with [.
During a live watch, pressing R yourself also clears the ledger for the new
match. At match completion, restart with R and press ] to continue. The command
does not send game input.

To inspect the new feature learner's revised rewards during your own play:

```powershell
.\run_mosim.cmd agent watch
```

This uses the same HUD and global keys, adds shot-correlated versus alliance
point totals, and runs score OCR asynchronously. `watch` without `agent` retains
the legacy pixel-agent rewards with the movement bonus removed. Camera swap
is H in both watch paths and recording. The objective HUD names its current
goal and target, gives the sensor-based reason, and lists permitted controls.

```powershell
python mosim_env.py watch --corner top-left
python mosim_env.py watch --vision-fps 30
python mosim_env.py watch --no-overlay
python mosim_env.py watch --no-reward-log
```

`--no-overlay` disables detection and leaves reward logging available. The
`--vision-fps` setting is a maximum background rate, not a guaranteed frame rate.

## Logs

Each session creates `mosim_debug_logs/rewards_<timestamp>.jsonl`. Session records
include config and reward settings; reset records separate matches; step records
contain the full component amounts, reasons, observations and diagnostic values.
Files flush every ten reward steps and on reset/close.

Raw component sums equal the reward returned by the environment before PPO
normalization. Blue-alliance score changes can include teammates' points. Neither
hopper color coverage nor view-motion difference is a direct simulator sensor.

## Optional YOLO

Install `ultralytics` into the same Python environment, then provide custom MoSim
weights containing `fuel`, `blue_robot`, `red_robot`, and `hub`:

```powershell
python -m pip install ultralytics
python mosim_env.py watch --weights path/to/best.pt
```

For a compatible CUDA-enabled PyTorch install, `--vision-device 0` selects the
first GPU. No GPU is required by the current color-based fuel detector.

Decoded hub marker anchors are also available without YOLO weights. They use
OpenCV's 36h11 AprilTag backend, show the marker ID, and exclude non-hub tags.
Robot boxes and whole-hub object boxes still require learned weights.

Custom weights are auto-discovered under `vision_weights/best.pt` or the newest
`runs/detect/**/weights/best.pt`. No trained weights or datasets are distributed
with this repository. Automatic labeling is still a roadmap item.

The standalone collector saves varied gameplay screenshots without injecting
keys. ] starts/pauses and [ finishes:

```powershell
python mosim_vision.py check
python mosim_vision.py collect
```

Collected images are written to `vision_data/images` and excluded from Git.

## Train the feature agent

```powershell
.\train_agent.cmd
```

Focus the game and press ] to begin. [ stops and saves; the next launch resumes.
Initialization needs no human demonstrations or manual ball labels. See the
[training guide](training.md) for the revised rewards, offline practice,
checkpoints and startup checks. Practice scores are not real-game scores.

## Existing pixel-agent support

The underlying environment also retains recording, behavior cloning and PPO
commands. These are separate from the vision showcase. Existing v4 checkpoints
can be passed explicitly:

```powershell
python mosim_env.py play --model path/to/model_v4.zip
python mosim_env.py play --model path/to/model_v4.zip --dry-run
```

Normal agent mode sends actions after arming; dry run displays proposed actions
without sending keys or resets. Key-dependent rewards in dry run describe those
proposed actions, not proof that the game performed them.

## Common issues

| Symptom | Check |
|---|---|
| ] does not start observation | Focus the actual game. The panel reports the wrong foreground title; both MoSim and MoSimulator are recognized. |
| Overlay is invisible | Use borderless fullscreen or windowed mode. Exclusive fullscreen may hide it. |
| Score/hub/hopper boxes are misplaced | Re-run setup at the current resolution and HUD layout. |
| Score remains pending | Run OCR calibration and confirm Tesseract is installed. |
| Fuel boxes appear over scenery | They are color candidates; inspect the HSV settings and exclusion regions. |
| Boxes lag or disappear | Inspect processing time and capture age. Expired results are hidden. A lower-cost YOLO model or image size may help optional inference. |
| Capture/overlay initialization fails in a restricted environment | Run locally on the interactive Windows desktop; native DXGI access and capture exclusion must be available. |

## Offline preview

```powershell
python mosim_env.py debug-preview --image path/to/game_screenshot.png --output preview.png
```

This renders the same layout on a saved image with explicitly synthetic rewards.
The source screenshot is supplied by the user and is not bundled with the repo.
