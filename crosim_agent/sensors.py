"""Latest-frame score OCR with nonblocking, exactly-once point delivery."""
from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True)
class ScoreSnapshot:
    confirmed: int | None = None
    pending: int | None = None
    failures: int = 0
    error: str | None = None
    captured_at: float = 0.0
    timer_seconds: int | None = None
    timer_at: float = 0.0
    timer_error: str | None = None


class AsyncScoreReader:
    """Adapter for the legacy reader API. A slow read cannot block action ticks.

    reset() invalidates in-flight results with a generation number. New points
    are accumulated until update() drains them once; dropping pending images
    never drops a previously confirmed score event.
    """
    def __init__(self, reader_factory, interval=0.10, frame_transform=None, crop_reader=False):
        self.factory = reader_factory
        self.interval = interval
        self.frame_transform = frame_transform or (lambda frame: frame)
        self.crop_reader = crop_reader
        self.condition = threading.Condition()
        self.generation = 0
        self.pending_frame = None
        self.closed = False
        self.snapshot = ScoreSnapshot()
        self.points = 0
        self.thread = threading.Thread(target=self._run, name="crosim-score", daemon=True)
        self.thread.start()

    @property
    def confirmed(self):
        with self.condition:
            return self.snapshot.confirmed

    @property
    def _pending(self):
        with self.condition:
            return self.snapshot.pending

    @property
    def ocr_failures(self):
        with self.condition:
            return self.snapshot.failures

    def reset(self):
        with self.condition:
            self.generation += 1
            self.pending_frame = None
            self.points = 0
            self.snapshot = ScoreSnapshot()
            self.condition.notify_all()

    def update(self, frame, now):
        with self.condition:
            if self.closed:
                raise RuntimeError("Score reader is closed")
            self.pending_frame = (self.frame_transform(frame).copy(), float(now), self.generation)
            points, self.points = self.points, 0
            self.condition.notify_all()
            return points

    def _run(self):
        reader = None
        generation = -1
        next_at = 0.0
        while True:
            with self.condition:
                while not self.closed and (self.pending_frame is None or time.perf_counter() < next_at):
                    self.condition.wait(max(0.001, next_at-time.perf_counter()) if self.pending_frame else None)
                if self.closed:
                    return
                frame, captured, requested_generation = self.pending_frame
                self.pending_frame = None
            started = time.perf_counter()
            try:
                if generation != requested_generation:
                    reader = None  # Never expose the old match if factory initialization fails.
                    reader = self.factory()
                    generation = requested_generation
                points = (reader.update_crop(frame, captured) if self.crop_reader else reader.update(frame, captured))
                if not isinstance(points, int) or not 0 <= points <= 40:
                    raise ValueError("OCR reader returned an invalid score delta")
                timer = getattr(reader,"timer",None)
                snapshot = ScoreSnapshot(reader.confirmed, reader._pending, reader.ocr_failures,
                                         getattr(reader,"error",None), captured,getattr(timer,"seconds",None),
                                         getattr(timer,"at",0.0),getattr(timer,"error",None))
            except Exception as exc:
                points = 0
                snapshot = ScoreSnapshot(getattr(reader, "confirmed", None), getattr(reader, "_pending", None),
                                         self.snapshot.failures+1,
                                         f"{type(exc).__name__}: {exc}", captured)
            with self.condition:
                if requested_generation == self.generation:
                    self.points += int(points)
                    self.snapshot = snapshot
            finished = time.perf_counter()
            # At most half of this worker's time goes to OCR on slow hardware.
            # Keep only the latest pending crop, rather than queuing old frames.
            next_at = max(started+self.interval,finished+(finished-started))

    def close(self):
        with self.condition:
            self.closed = True
            self.pending_frame = None
            self.condition.notify_all()
        self.thread.join(timeout=5.0)
