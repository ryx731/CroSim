"""Reward telemetry shared by the environment, overlay and JSONL recorder.

Amounts here are the raw environment rewards, before PPO normalization.
The overlay groups uninterrupted repetitions; the log keeps every single tick.
"""
from collections import deque
from datetime import datetime
import json
import math
from pathlib import Path


REWARD_LABELS = {
    "score": "Alliance score",
    "shaping": "Fuel / hub potential",
    "stillness": "Stillness",
    "time": "Match time cost",
    "wasted_shot": "Wasted shot",
    "key_change": "Key changes",
    "conflict": "Opposing keys",
}


class RewardLedger:
    def __init__(self, max_events=80, labels=None):
        self.max_events = max_events
        self.labels = dict(REWARD_LABELS if labels is None else labels)
        self.episode = -1
        self.reset()

    def reset(self):
        self.episode += 1
        self.step = 0
        self.total = 0.0
        self.positive = 0.0
        self.negative = 0.0
        self.latest = {name: 0.0 for name in self.labels}
        self.totals = dict(self.latest)
        self.events = deque(maxlen=self.max_events)
        self._active = {}
        self.trend = deque(maxlen=100)
        self.last_score = None

    def record(self, parts, elapsed, reasons=None):
        if set(parts) != set(self.labels):
            raise ValueError("Every reward component must be present exactly once")
        parts = {k: float(v) for k, v in parts.items()}
        if not all(math.isfinite(v) for v in parts.values()):
            raise ValueError("Non-finite reward component")
        self.step += 1
        self.latest = parts
        reasons = reasons or {}
        delta = math.fsum(parts.values())
        self.total = math.fsum((self.total, delta))
        for name, amount in parts.items():
            self.totals[name] += amount
            self.positive += max(amount, 0.0)
            self.negative += min(amount, 0.0)
            reason = reasons.get(name, self.labels[name])
            if abs(amount) <= 1e-12:
                self._active.pop(name, None)
                continue
            signature = (name, amount > 0, reason)
            active = self._active.get(name)
            if (name != "score" and active is not None
                    and active[0] == signature and active[1]["last_step"] == self.step - 1
                    and any(event is active[1] for event in self.events)):
                event = active[1]
                event["amount"] += amount
                event["count"] += 1
                event["end"] = elapsed
                event["last_step"] = self.step
            else:
                event = {"name": name, "amount": amount, "count": 1,
                         "start": elapsed, "end": elapsed, "last_step": self.step,
                         "reason": reason}
                self.events.append(event)
                self._active[name] = (signature, event)
            if name == "score":
                self.last_score = dict(event)
        if self.step % 10 == 0:
            self.trend.append(self.total)
        return delta

    def snapshot(self):
        # Detached objects: the UI reads this from another thread.
        return {"episode": self.episode, "step": self.step, "total": self.total,
                "labels": dict(self.labels),
                "positive": self.positive, "negative": self.negative,
                "latest": dict(self.latest), "totals": dict(self.totals),
                "events": [dict(e) for e in self.events], "trend": list(self.trend),
                "last_score": dict(self.last_score) if self.last_score else None}


class RewardLog:
    """Append all ticks in order. Flushed once a second and at every reset/close."""
    def __init__(self, directory, metadata=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.path = directory / f"rewards_{stamp}.jsonl"
        self.file = self.path.open("x", encoding="utf-8")
        self.write({"type": "session", "schema": 1, "started": datetime.now().isoformat(),
                    "reward_units": "raw_before_normalization", **(metadata or {})})

    def write(self, record):
        self.file.write(json.dumps(record, allow_nan=False) + "\n")
        if record.get("type") != "step" or record.get("step", 0) % 10 == 0:
            self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.flush()
            self.file.close()
