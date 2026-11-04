"""Optional YOLO observer with zero-label fuel candidates and hub markers.

This module only observes; the feature learner can consume fresh results.
Custom MoSim weights must
contain fuel, blue_robot, red_robot, hub. Generic COCO weights are rejected.
"""
import argparse
from dataclasses import asdict, dataclass
import importlib.util
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from mosim_landmarks import HubTagDetector
from mosim_zones import FuelZoneFilter, field_fuel_mask
from mosim_field import FieldPoseTracker


HERE = Path(__file__).resolve().parent
CLASSES = {"fuel", "blue_robot", "red_robot", "hub"}


@dataclass(frozen=True)
class Detection:
    name: str
    box: tuple  # normalized xyxy, relative to the exact captured monitor
    confidence: float | None = None
    source: str = "yolo"
    tag_id: int | None = None
    alliance: str | None = None
    captured_at: float | None = None
    collectible: bool = True
    excluded_reason: str | None = None


def find_weights(explicit=None):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"No weights at {path}")
        return path
    candidates = list((HERE / "runs" / "detect").glob("**/weights/best.pt"))
    candidates += list((HERE / "vision_weights").glob("best.pt"))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def color_candidates(frame, config):
    """Yellow connected regions, NOT a count of individual fuel balls."""
    return mask_candidates(field_fuel_mask(frame,config),False,"unverified color-only candidate")


def mask_candidates(mask, collectible=True, reason=None, limit=100):
    if limit <= 0:
        return []
    sh,sw = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if not 6 <= area <= sw * sh * 0.08:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        boxes.append(Detection("fuel? color", (x/sw, y/sh, (x+bw)/sw, (y+bh)/sh),
                               source="color_candidate",collectible=collectible,excluded_reason=reason))
        if len(boxes) >= limit:
            break
    return boxes


class ObjectDetector:
    def __init__(self, config, weights=None, conf=0.4, imgsz=960, device=None, discover_weights=True):
        self.config = dict(config)
        self.weights = find_weights(weights) if weights or discover_weights else None
        self.conf, self.imgsz, self.device = conf, imgsz, device
        self.model = None
        self.hub_tags = HubTagDetector(max_width=1280, interval=0, hub_only=False)
        self.zone_filter = FuelZoneFilter("blue")
        self.zone_info = {}
        self.field_tracker = FieldPoseTracker("blue")
        self.field_map = {}
        self.status = "COLOR + HUB TAGS: robot model not loaded" if self.hub_tags.available else \
            "COLOR ONLY: AprilTag backend unavailable"
        if self.weights:
            if importlib.util.find_spec("ultralytics") is None:
                raise RuntimeError("Weights found, but ultralytics is missing. Run: pip install ultralytics")
            from ultralytics import YOLO
            self.model = YOLO(str(self.weights))
            names = self.model.names
            values = names.values() if isinstance(names, dict) else names
            self.names = {int(k): str(v).lower().replace("-", "_").replace(" ", "_")
                          for k, v in (names.items() if isinstance(names, dict) else enumerate(values))}
            if not CLASSES.issubset(set(self.names.values())):
                raise ValueError("Use custom MoSim weights with classes: fuel, blue_robot, red_robot, hub. "
                                 f"This model has: {', '.join(self.names.values())}")
            self.status = f"YOLO: {self.weights.name}"

    def detect(self, frame, captured_at=None):
        captured_at = time.perf_counter() if captured_at is None else captured_at
        anchors = self.hub_tags.detect(frame,captured_at)
        self._map_fuel_mask = field_fuel_mask(frame,self.config)
        access = self.zone_filter.apply(self._map_fuel_mask,anchors)
        self.zone_info = access.snapshot()
        if self.model is None:
            found = mask_candidates(access.eligible)
            found += mask_candidates(access.excluded,False,"red / unverified zone",limit=100-len(found))
            for anchor in anchors:
                if anchor.kind != "hub":
                    continue
                found.append(Detection(f"hub_{anchor.alliance}_anchor", anchor.box,
                                       source="apriltag_36h11_tracked" if anchor.tracked else "apriltag_36h11", tag_id=anchor.tag_id,
                                       alliance=anchor.alliance, captured_at=anchor.captured_at))
            self._update_field_map(found,captured_at)
            return found
        kwargs = {"conf": self.conf, "imgsz": self.imgsz, "verbose": False, "max_det": 300}
        if self.device:
            kwargs["device"] = self.device
        result = self.model.predict(source=frame, **kwargs)[0]
        h, w = frame.shape[:2]
        found = []
        if result.boxes is not None:
            for row in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2, confidence, cls = row[:6]
                name = self.names.get(int(cls))
                if name in CLASSES:
                    box = tuple(float(v) for v in np.clip([x1/w, y1/h, x2/w, y2/h], 0, 1))
                    collectible = True
                    reason = None
                    if name == "fuel":
                        sh,sw = access.eligible.shape
                        samples = np.array([[(box[0]+box[2])/2,(box[1]+box[3])/2],
                                            [(box[0]+box[2])/2,box[3]]])*[sw,sh]
                        collectible = bool(self.zone_filter.point_eligibility(samples,self.zone_filter.hypotheses).all())
                        reason = None if collectible else "red / unverified zone"
                    found.append(Detection(name, box, float(confidence),collectible=collectible,excluded_reason=reason))
        self._update_field_map(found,captured_at)
        return found

    def _update_field_map(self,found,captured_at):
        self.field_map = self.field_tracker.update(self.zone_filter.hypotheses,self.config["hopper_region"],
                                                   captured_at=captured_at,detections=[asdict(d) for d in found],
                                                   fuel_mask=getattr(self,"_map_fuel_mask",None))

    def reset(self):
        self.hub_tags.reset()
        self.zone_filter.reset()
        self.field_tracker.reset()
        self.field_map = {}


class DetectionWorker:
    """Latest-frame inference; optionally capture independently of reward/OCR.

    A frame source returns (frame, capture timestamp). It must serialize native
    capture with the environment. Paused sources return None; no game input is
    ever injected by this worker.
    """
    def __init__(self, detector, fps=30, frame_source=None, enabled=None):
        self.detector = detector
        self.period = 1.0 / max(0.5, fps)
        self.frame_source = frame_source
        self.enabled = enabled or (lambda: True)
        self.max_age = 0.25 if getattr(detector, "model", None) is None else 0.8
        self.condition = threading.Condition()
        self.pending = None
        self.closed = False
        self.generation = 0
        self.result = self._empty_result()
        self.thread = threading.Thread(target=self._run, name="mosim-vision", daemon=True)
        self.thread.start()

    def reset(self):
        with self.condition:
            self.generation += 1
            self.pending = None
            self.result = self._empty_result()

    def _empty_result(self):
        return {"boxes": [], "captured_at": 0.0, "ms": 0.0,
                "status": self.detector.status, "max_age": self.max_age}

    def submit(self, frame, captured_at=None):
        if self.frame_source is not None:
            return  # The live source supplies newer frames than the reward loop.
        with self.condition:
            self.pending = (frame.copy(), time.perf_counter() if captured_at is None else captured_at,
                            self.generation)
            self.condition.notify()

    def latest(self):
        with self.condition:
            return dict(self.result)

    def _run(self):
        next_at = 0.0
        detector_generation = -1
        while True:
            with self.condition:
                while not self.closed and (time.perf_counter() < next_at
                                            or (self.frame_source is None and self.pending is None)):
                    timeout = max(0.001, next_at - time.perf_counter()) \
                        if self.pending is not None or self.frame_source is not None else None
                    self.condition.wait(timeout)
                if self.closed:
                    return
                pending = self.pending
                self.pending = None
                generation = self.generation
            started = time.perf_counter()
            captured = started
            try:
                if not self.enabled():
                    with self.condition:
                        self.result = self._empty_result()
                    next_at = time.perf_counter() + self.period
                    continue
                if self.frame_source is not None:
                    sample = self.frame_source()
                    if sample is None:
                        next_at = time.perf_counter() + self.period
                        continue
                    frame, captured = sample
                else:
                    frame, captured, generation = pending
                if generation != detector_generation and isinstance(self.detector,ObjectDetector):
                    self.detector.reset()
                    detector_generation = generation
                detected = self.detector.detect(frame,captured_at=captured) if isinstance(self.detector,ObjectDetector) \
                           else self.detector.detect(frame)
                boxes = [asdict(box) for box in detected]
                status = self.detector.status
            except Exception as exc:
                boxes, status = [], f"VISION ERROR: {type(exc).__name__}: {exc}"
            with self.condition:
                if generation == self.generation:
                    self.result = {"boxes": boxes, "captured_at": captured,
                                   "ms": (time.perf_counter() - started)*1000, "status": status,
                                   "max_age": self.max_age, "fuel_zones": dict(getattr(self.detector,"zone_info",{})),
                                   "field_map":dict(getattr(self.detector,"field_map",{}))}
            finished = time.perf_counter()
            # Allow higher refresh while reserving at least one third of this
            # worker's time for the game/learner when inference is expensive.
            next_at = max(started+self.period,finished+(finished-started)*.5)

    def close(self):
        with self.condition:
            self.closed = True
            self.pending = None
            self.condition.notify_all()
        # A cold first YOLO inference can exceed two seconds. Input has already
        # been released before close; give the active call time to finish.
        self.thread.join(timeout=30.0)


def collect(args):
    """No injected keys. ] toggles saving; [ stops; images exclude our overlay."""
    import mosim_env as env
    from mosim_controls import InputGuard
    grabber = None
    guard = InputGuard(lambda: None, env.CFG.get("game_window_title", "MoSim"))
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    print("] starts/pauses capture when MoSim is focused; [ quits. No game keys are injected.")
    previous, next_at, count = None, 0.0, 0
    try:
        guard.wait()
        grabber = env.Grabber()
        while True:
            guard.wait()
            if time.perf_counter() < next_at:
                time.sleep(0.02)
                continue
            frame = grabber.grab()
            sample = cv2.resize(frame, (160, 90)).astype(np.float32)
            change = float(np.abs(sample - previous).mean()) if previous is not None else float("inf")
            if change >= args.min_change:
                path = directory / f"capture_{time.time_ns()}.jpg"
                if not cv2.imwrite(str(path), frame):
                    raise OSError(f"Could not save {path}")
                previous = sample
                count += 1
                print(f"Saved {count}: {path.name}", flush=True)
            next_at = time.perf_counter() + args.every
    except KeyboardInterrupt:
        print(f"Stopped. {count} images saved in {directory.resolve()}")
    finally:
        guard.close()
        if grabber:
            grabber.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="show custom weights and optional dependency status")
    p = sub.add_parser("collect", help="collect real screenshots while you play")
    p.add_argument("--output", default=str(HERE / "vision_data" / "images"))
    p.add_argument("--every", type=float, default=1.0)
    p.add_argument("--min-change", type=float, default=3.0)
    p = sub.add_parser("overlay", help="watch detections and rewards while you play")
    p.add_argument("--weights")
    args = parser.parse_args()
    if args.command == "check":
        print(f"Custom weights: {find_weights() or 'not trained / not found'}")
        print(f"Ultralytics installed: {importlib.util.find_spec('ultralytics') is not None}")
        print(f"Collected JPGs: {len(list((HERE / 'vision_data' / 'images').glob('*.jpg')))}")
        print("Without weights: HUD regions, yellow fuel candidates, and decoded hub AprilTags when available.")
    elif args.command == "collect":
        if args.every <= 0 or args.min_change < 0:
            parser.error("--every must be positive and --min-change nonnegative")
        collect(args)
    else:
        import mosim_env
        argv = ["watch"] + (["--weights", args.weights] if args.weights else [])
        options = mosim_env.build_parser().parse_args(argv)
        mosim_env.watch(options)


if __name__ == "__main__":
    main()
