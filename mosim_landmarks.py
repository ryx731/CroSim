"""Decode REBUILT hub AprilTags directly from pixels; no trained weights.

IDs follow FIRST's 2026 field drawings. These are marker anchors, not inferred
whole-hub boxes or a calibrated 3-D pose. Unknown marker IDs are never relabeled
as hubs. OpenCV builds without aruco simply report an unavailable backend.
"""
from dataclasses import dataclass, replace
import time

import cv2
import numpy as np


HUB_TAG_IDS = {
    "blue": frozenset((18, 19, 20, 21, 24, 25, 26, 27)),
    "red": frozenset((2, 3, 4, 5, 8, 9, 10, 11)),
}
TAG_ALLIANCE = {tag: alliance for alliance, tags in HUB_TAG_IDS.items() for tag in tags}
FIELD_TAG_ALLIANCE = {tag: "red" if tag <= 16 else "blue" for tag in range(1, 33)}
FIELD_REFERENCE = "https://firstfrc.blob.core.windows.net/frc2026/FieldAssets/2026-field-dimension-dwgs.pdf"


@dataclass(frozen=True)
class HubAnchor:
    tag_id: int
    alliance: str
    box: tuple
    center: tuple
    scale: float  # mean tag edge / image width: an apparent-size proxy only
    captured_at: float
    corners: tuple = ()
    kind: str = "hub"
    tracked: bool = False


class HubTagDetector:
    def __init__(self, max_width=960, interval=0.15, hub_only=True):
        self.max_width = max_width
        self.interval = interval
        self.hub_only = hub_only
        self.detector = None
        self.error = None
        self.reset()
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "ArucoDetector"):
            self.error = "OpenCV aruco backend unavailable"
            return
        parameters = cv2.aruco.DetectorParameters()
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    def reset(self):
        self._last_run = -float("inf")
        self._last_native = -float("inf")
        self._anchors = []
        self._previous_gray = None
        self._previous_at = -float("inf")
        self._decoded_at = {}

    @property
    def available(self):
        return self.detector is not None

    def detect(self, frame, captured_at=None):
        now = time.perf_counter() if captured_at is None else float(captured_at)
        if self.detector is None:
            return []
        if 0 <= now - self._last_run < self.interval:
            return list(self._anchors)
        self._last_run = now
        height, width = frame.shape[:2]
        ratio = min(1.0, self.max_width / width)
        native_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = native_gray
        if ratio < 1:
            gray = cv2.resize(gray, (round(width*ratio), round(height*ratio)), interpolation=cv2.INTER_AREA)
        anchors = list(self._decode(gray,now))
        paired = any(sum(a.kind == b.kind and a.alliance == b.alliance for b in anchors) >= 2 for a in anchors)
        if not paired and ratio < 1 and now-self._last_native >= .3:
            self._last_native = now
            native_anchors = self._decode(native_gray,now)
            anchors = list({a.tag_id:a for a in [*anchors,*native_anchors]}.values())
        for anchor in anchors:
            self._decoded_at[anchor.tag_id] = now
        detected = {a.tag_id for a in anchors}
        anchors.extend(a for a in self._track(gray,now) if a.tag_id not in detected)
        self._previous_gray,self._previous_at = gray,now
        self._anchors = anchors
        return list(anchors)

    def _track(self,gray,now):
        previous = self._previous_gray
        candidates = [a for a in self._anchors if now-self._decoded_at.get(a.tag_id,-1e9) <= .6]
        if previous is None or previous.shape != gray.shape or not 0 < now-self._previous_at <= .25 or not candidates:
            return []
        height,width = gray.shape
        points = np.asarray([a.corners for a in candidates],np.float32).reshape(-1,1,2)*[width,height]
        points = points.astype(np.float32)
        moved,status,error = cv2.calcOpticalFlowPyrLK(previous,gray,points,None,winSize=(21,21),maxLevel=3)
        if moved is None:
            return []
        back,back_status,_ = cv2.calcOpticalFlowPyrLK(gray,previous,moved,None,winSize=(21,21),maxLevel=3)
        if back is None:
            return []
        good = status.ravel().astype(bool) & back_status.ravel().astype(bool)
        good &= np.linalg.norm(back-points,axis=2).ravel() <= .8
        good &= error.ravel() <= 25
        result = []
        for index,anchor in enumerate(candidates):
            group = slice(index*4,index*4+4)
            quad = moved[group].reshape(4,2)
            if not good[group].all() or not np.isfinite(quad).all() or not cv2.isContourConvex(quad):
                continue
            if (quad < 0).any() or (quad >= [width,height]).any():
                continue
            old_area = abs(cv2.contourArea(points[group]))
            new_area = abs(cv2.contourArea(quad))
            if old_area < 16 or not .6 <= new_area/old_area <= 1.6:
                continue
            left,top = np.floor(quad.min(axis=0)).astype(int)
            right,bottom = np.ceil(quad.max(axis=0)).astype(int)
            if np.std(gray[top:bottom,left:right]) < 25:
                continue
            normalized = quad/[width,height]
            low,high = normalized.min(axis=0),normalized.max(axis=0)
            result.append(replace(anchor,box=tuple([*low,*high]),center=tuple(normalized.mean(axis=0)),
                                  corners=tuple(tuple(p) for p in normalized),
                                  scale=float(np.linalg.norm(quad-np.roll(quad,1,axis=0),axis=1).mean()/width),
                                  captured_at=now,tracked=True))
        return result

    def _decode(self,gray,now):
        sh,sw = gray.shape
        corners, ids, _ = self.detector.detectMarkers(gray)
        anchors = []
        for marker_id, corner in zip(ids.flatten() if ids is not None else [], corners):
            alliance = (TAG_ALLIANCE if self.hub_only else FIELD_TAG_ALLIANCE).get(int(marker_id))
            if alliance is None:
                continue
            points = corner.reshape(4, 2).astype(np.float64)
            normalized = points / np.array([sw, sh])
            left, top = normalized.min(axis=0)
            right, bottom = normalized.max(axis=0)
            center = normalized.mean(axis=0)
            edge = float(np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1).mean() / sw)
            kind = "hub" if int(marker_id) in TAG_ALLIANCE else \
                   "outpost" if int(marker_id) in (13, 14, 29, 30) else \
                   "tower" if int(marker_id) in (15, 16, 31, 32) else "trench"
            # MoSim's printed tag textures can be rotated relative to the field
            # mount. Pose geometry needs the upright physical square corners,
            # not the decoder's texture-relative first corner.
            top_corners = sorted(normalized[np.argsort(normalized[:,1])[:2]], key=lambda point: point[0])
            bottom_corners = sorted(normalized[np.argsort(normalized[:,1])[2:]], key=lambda point: point[0])
            upright = [top_corners[0],top_corners[1],bottom_corners[1],bottom_corners[0]]
            anchors.append(HubAnchor(int(marker_id), alliance,
                                     tuple(float(v) for v in np.clip([left, top, right, bottom], 0, 1)),
                                     tuple(float(v) for v in center), edge, now,
                                     tuple(tuple(float(v) for v in point) for point in upright), kind))
        return anchors


def strongest_hub(anchors, alliance="blue", now=None, max_age=0.4):
    """Group only fresh markers from the requested alliance into one fixed slot."""
    now = time.perf_counter() if now is None else now
    selected = [a for a in anchors if a.kind == "hub" and a.alliance == alliance and 0 <= now-a.captured_at <= max_age]
    if not selected:
        return None
    weights = np.array([max(a.scale, 1e-6)**2 for a in selected])
    center = np.average([a.center for a in selected], axis=0, weights=weights)
    return {"x": float(center[0]), "y": float(center[1]),
            "scale": max(a.scale for a in selected),
            "age": max(0.0, now-max(a.captured_at for a in selected)),
            "tag_ids": [a.tag_id for a in selected], "source": "apriltag_36h11"}
