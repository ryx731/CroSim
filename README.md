# CroSim

A screen-based learning agent for MoSimulator, with a computer-vision HUD,
human demonstration learning, and autonomous PPO training.

**Experimental work in progress.** Known perception and control bugs remain.
Training results vary; current models are not validated for consistent high scores.

## Setup and use

Windows, Python 3.11+, Tesseract OCR, and MoSimulator are required.

```powershell
python -m pip install -r requirements.txt
.\run_mosim.cmd setup
.\run_mosim.cmd agent check
```

- **Train autonomously:** `.\train_agent.cmd`; focus MoSim and press **]**.
- **Teach by playing:** `.\teach_agent.cmd`; start a match with **R**, press **]**, then play. **[** saves recordings and fits the baseline.
- **Hand control back:** `.\train_agent.cmd` resumes from the fitted baseline.
- **Refit saved demonstrations:** `.\fit_demonstrations.cmd`.
- **Inspect without learning:** `.\run_mosim.cmd agent play --episodes 3`.
- **Pause/resume:** **]**. **Stop/save:** **[**. Camera swap: **H**.

Demonstrations use complete categorical actions conditioned on scene/history,
not averaged key bits. Unsupported control combinations are retained raw but
excluded from fitting and reported. Autonomous PPO continues the fitted actor.

## Vision and debugging

Shared HUD components provide fuel detection, score/timer OCR, a draggable reward
panel, estimated robot pose in both camera views, field-state tracking, and
confidence-shaded fuel clusters. Pose and contact are estimates, not simulator
telemetry. Match resets have an OCR end gate and a 165-second watchdog.
The public HUD showcase is [CroSim Vision](https://github.com/ryx731/CroSim-Vision).

See [training](docs/training.md), [architecture](docs/architecture.md), and
[learning design](docs/learning-design.md) for behavior and limitations.

## Source and local data

`crosim_agent/` contains the feature policy, PPO integration, imitation learner,
versioned checkpoints, sensors, objectives, and rewards. `mosim_*.py` contains
shared capture/control, CV, field localization, and HUD rendering.

Calibration, demonstrations, reward logs, model weights, archives, and local
credentials are deliberately excluded. Set up local calibration before live use.
The repository includes source and tests, not a pretrained model.

```powershell
python -m unittest discover -s tests -q
```
