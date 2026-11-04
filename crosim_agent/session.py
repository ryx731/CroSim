"""One trainer per project, with an explicit reset of local learned artifacts."""
from datetime import datetime
import json
import os
from pathlib import Path


class TrainingLease:
    def __init__(self, root):
        self.path = Path(root)/".crosim-training.lock"
        self.file = None

    def __enter__(self):
        self.file = self.path.open("a+b")
        self.file.seek(0,2)
        if not self.file.tell():
            self.file.write(b"0")
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            raise RuntimeError("Another CroSim trainer/reset is running. Stop it with [ first.") from exc
        return self

    def __exit__(self,*exc):
        try:
            self.file.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(),fcntl.LOCK_UN)
        finally:
            self.file.close()


def reset_training(root):
    """Move active learning data out of every automatic resume path.

    Preserve calibration, vision weights and source. Validate every resolved
    source and destination before moving anything, including Windows junctions.
    """
    root = Path(root).resolve()
    names = ("crosim_agent_models","mosim_demos","mosim_debug_logs","mosim_tensorboard",
             "mosim_checkpoints","mosim_checkpoints_v2","mosim_checkpoints_v4")
    candidates = [root/name for name in names if (root/name).exists()]
    for pattern in ("mosim_bc_init*.zip","mosim_ppo_model*.zip","mosim_vecnormalize*.pkl"):
        candidates.extend(root.glob(pattern))
    destination = root/"training_archive"/datetime.now().strftime("reset_%Y%m%d_%H%M%S_%f")
    for path in [destination,*candidates]:
        if not path.resolve().is_relative_to(root) or path.resolve() == root:
            raise ValueError(f"Training reset path escapes the project: {path}")
    with TrainingLease(root):
        destination.mkdir(parents=True,exist_ok=False)
        moved = []
        for source in candidates:
            target = destination/source.name
            source.rename(target)
            moved.append(source.name)
        (destination/"reset.json").write_text(json.dumps({"archived":moved,"active_training_reset":True},indent=2)+"\n")
    return destination,moved
