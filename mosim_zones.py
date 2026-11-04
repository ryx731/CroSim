"""Conservative screen-to-field fuel eligibility; opponent and unknown are out.

No fixed screen rectangle or guessed color-side boundary is used. Decoded field
tags establish camera-pose hypotheses over a range of pinhole focal lengths.
Fuel must lie inside the field and outside the opponent zone in EVERY accepted
hypothesis. Without a fresh, plausible pose no fuel is offered for collection.
These are estimates for the standard field, not calibrated ground-truth pose.
"""
from dataclasses import dataclass
import math
import time

import cv2
import numpy as np


FUEL_ACCESS_VERSION = "field-tag-opponent-exclusion-v1"
LAYOUT_SOURCE = "https://raw.githubusercontent.com/wpilibsuite/allwpilib/v2026.2.1/apriltag/src/main/native/resources/edu/wpi/first/apriltag/2026-rebuilt-welded.json"
ZONE_SOURCE = "https://firstfrc.blob.core.windows.net/frc2026/Manual/HTML/2026GameManual.htm"
FIELD_LENGTH, FIELD_WIDTH = 16.541, 8.069
ALLIANCE_ZONE_DEPTH = 158.6*0.0254
TAG_SIZE = 8.125*0.0254
ZONE_MARGIN = 0.40


def field_fuel_mask(frame, config):
    """One shared HSV/HUD-exclusion path for the learner and Vision HUD."""
    height, width = frame.shape[:2]
    ratio = min(1.0,960/width)
    small = cv2.resize(frame,(round(width*ratio),round(height*ratio)),interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,np.asarray(config["fuel_hsv_lower"],np.uint8),np.asarray(config["fuel_hsv_upper"],np.uint8))
    sh,sw = mask.shape
    mask[:int(sh*0.08)] = 0
    mask[int(sh*0.84):] = 0
    for name in ("score","hopper","hub"):
        left,top,right,bottom = config[name+"_region"]
        mask[int(top*sh):int(bottom*sh),int(left*sw):int(right*sw)] = 0
    return mask
# Factual tag center coordinates (m) and face yaw (degrees), WPILib 2026 welded.
TAG_POSES = {
    1:(11.8779798,7.4247756,0.889,180), 2:(11.9154194,4.6380400,1.12395,90),
    3:(11.3118646,4.3902376,1.12395,180), 4:(11.3118646,4.0346376,1.12395,180),
    5:(11.9154194,3.4312352,1.12395,-90), 6:(11.8779798,0.6444996,0.889,180),
    7:(11.9528844,0.6444996,0.889,0), 8:(12.2710194,3.4312352,1.12395,-90),
    9:(12.5191774,3.6790376,1.12395,0), 10:(12.5191774,4.0346376,1.12395,0),
    11:(12.2710194,4.6380400,1.12395,90), 12:(11.9528844,7.4247756,0.889,0),
    13:(16.5333172,7.4033126,0.55245,180), 14:(16.5333172,6.9715126,0.55245,180),
    15:(16.5329616,4.3235626,0.55245,180), 16:(16.5329616,3.8917626,0.55245,180),
    17:(4.6630844,0.6444996,0.889,0), 18:(4.6256194,3.4312352,1.12395,-90),
    19:(5.2291742,3.6790376,1.12395,0), 20:(5.2291742,4.0346376,1.12395,0),
    21:(4.6256194,4.6380400,1.12395,90), 22:(4.6630844,7.4247756,0.889,0),
    23:(4.5881798,7.4247756,0.889,180), 24:(4.2700194,4.6380400,1.12395,90),
    25:(4.0218614,4.3902376,1.12395,180), 26:(4.0218614,4.0346376,1.12395,180),
    27:(4.2700194,3.4312352,1.12395,-90), 28:(4.5881798,0.6444996,0.889,180),
    29:(0.0077470,0.6659626,0.55245,0), 30:(0.0077470,1.0977626,0.55245,0),
    31:(0.0080772,3.7457126,0.55245,0), 32:(0.0080772,4.1775126,0.55245,0),
}


def tag_world_corners(tag_id, size=TAG_SIZE):
    x, y, z, yaw = TAG_POSES[tag_id]
    angle = math.radians(yaw)
    horizontal = np.array([-math.sin(angle), math.cos(angle), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    center = np.array([x, y, z])
    return np.array([center + size/2*(side*horizontal+vertical*up)
                     for side, vertical in ((-1,1),(1,1),(1,-1),(-1,-1))], np.float64)


@dataclass
class CameraHypothesis:
    rotation: np.ndarray  # world -> camera
    center: np.ndarray
    focal: float
    width: int
    height: int
    error: float

    def ground_points(self, pixels, plane_height=0.075):
        pixels = np.asarray(pixels, np.float64).reshape(-1, 2)
        rays = np.column_stack(((pixels[:,0]-self.width/2)/self.focal,
                                (pixels[:,1]-self.height/2)/self.focal, np.ones(len(pixels)))) @ self.rotation
        with np.errstate(divide="ignore", invalid="ignore"):
            distance = (plane_height-self.center[2])/rays[:,2]
            points = self.center+distance[:,None]*rays
        valid = (distance > 0) & (rays[:,2] < -1e-5) & np.isfinite(points).all(axis=1)
        return points, valid


@dataclass
class ZoneResult:
    eligible: np.ndarray
    excluded: np.ndarray
    status: str
    excluded_regions: int = 0
    tag_ids: tuple = ()
    hypotheses: int = 0

    def snapshot(self):
        return {"version": FUEL_ACCESS_VERSION, "status": self.status,
                "excluded_regions": self.excluded_regions, "tag_ids": list(self.tag_ids),
                "hypotheses": self.hypotheses, "eligible_pixels": int(np.count_nonzero(self.eligible))}


class FuelZoneFilter:
    def __init__(self, alliance="blue"):
        if alliance not in {"blue", "red"}:
            raise ValueError("Alliance must be blue or red")
        self.alliance = alliance
        self.hypotheses = []

    def reset(self):
        self.hypotheses = []

    def estimate(self, anchors, width, height, now=None):
        now = time.perf_counter() if now is None else now
        anchors = [anchor for anchor in anchors if anchor.tag_id in TAG_POSES and len(anchor.corners) == 4
                   and 0 <= now-anchor.captured_at <= 0.25 and anchor.scale*width >= 8]
        groups = {}
        for anchor in anchors:
            groups.setdefault((anchor.kind,anchor.alliance),[]).append(anchor)
        # A single square cannot establish the game's rendered marker size.
        # Use the known spacing of a marker pair instead of guessing range.
        groups = sorted((group for group in groups.values() if len(group) >= 2),
                        key=lambda group: min(a.scale for a in group)*len(group), reverse=True)
        # Widely separated tags constrain camera range/focal length much more
        # than a single hub face. Use them jointly when multiple groups exist;
        # retain pair fallback when the joint reprojection test fails.
        if len(anchors) >= 2 and len({(a.kind,a.alliance) for a in anchors}) > 1:
            combined=anchors
            fits=self._fit_group(combined,width,height)
            if fits:
                return fits,tuple(sorted(a.tag_id for a in combined))
        for group in groups:
            fits = self._fit_group(group,width,height)
            if fits:
                return fits, tuple(sorted(anchor.tag_id for anchor in group))
        return [], ()

    def _fit_group(self, anchors, width, height):
        images = np.concatenate([np.asarray(anchor.corners)*[width,height] for anchor in anchors]).astype(np.float64)
        if not np.isfinite(images).all():
            return []
        fits = []
        # The saved game frame uses smaller rendered squares than the official
        # 8.125-inch target. Fit size from pair spacing as well as focal length.
        for size in (0.14,0.15,0.1651,0.18,TAG_SIZE):
            objects = np.concatenate([tag_world_corners(anchor.tag_id,size) for anchor in anchors])
            for fov in (35,50,65,80,95):
                focal = height/(2*math.tan(math.radians(fov)/2))
                intrinsics = np.array([[focal,0,width/2],[0,focal,height/2],[0,0,1]], np.float64)
                try:
                    found, rvecs, tvecs, *_ = cv2.solvePnPGeneric(objects, images, intrinsics, None, flags=cv2.SOLVEPNP_SQPNP)
                except cv2.error:
                    continue
                if not found:
                    continue
                for rvec, tvec in zip(rvecs, tvecs):
                    rotation = cv2.Rodrigues(rvec)[0]
                    center = -rotation.T @ tvec.reshape(3)
                    projected = cv2.projectPoints(objects,rvec,tvec,intrinsics,None)[0].reshape(-1,2)
                    error = float(np.sqrt(np.mean(np.sum((projected-images)**2,axis=1))))
                    depth = (objects @ rotation.T+tvec.reshape(3))[:,2]
                    if not np.isfinite(center).all() or not math.isfinite(error) or not (0.4 <= center[2] <= 8.0) \
                            or not (-2 <= center[0] <= FIELD_LENGTH+2 and -2 <= center[1] <= FIELD_WIDTH+2) \
                            or np.any(depth <= 0) or error > 2.5:
                        continue
                    fits.append(CameraHypothesis(rotation, center, focal, width, height, error))
        if fits:
            best = min(fit.error for fit in fits)
            fits = [fit for fit in fits if fit.error <= best+0.4]
        return fits

    def point_eligibility(self, pixels, hypotheses):
        allowed = np.ones(len(pixels), dtype=bool) if hypotheses else np.zeros(len(pixels), dtype=bool)
        for hypothesis in hypotheses:
            points, valid = hypothesis.ground_points(pixels)
            x, y = points[:,0], points[:,1]
            valid &= (x >= 0) & (x <= FIELD_LENGTH) & (y >= 0) & (y <= FIELD_WIDTH)
            own_side = (x < FIELD_LENGTH-ALLIANCE_ZONE_DEPTH-ZONE_MARGIN) if self.alliance == "blue" \
                       else (x > ALLIANCE_ZONE_DEPTH+ZONE_MARGIN)
            valid &= own_side
            allowed &= valid
        return allowed

    def apply(self, fuel_mask, anchors=(), now=None):
        height, width = fuel_mask.shape
        self.hypotheses, tag_ids = self.estimate(anchors, width, height, now)
        eligible = np.zeros_like(fuel_mask)
        ys, xs = np.nonzero(fuel_mask)
        allowed = self.point_eligibility(np.column_stack((xs+0.5,ys+0.5)), self.hypotheses)
        eligible[ys[allowed],xs[allowed]] = 255
        # Do not retain the near edge of a ball/clump that crosses the boundary.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(fuel_mask, 8)
        total = np.bincount(labels[ys,xs], minlength=count)
        safe = np.bincount(labels[ys[allowed],xs[allowed]], minlength=count)
        rejected_ids = (total != safe)
        rejected_ids[0] = False
        eligible[rejected_ids[labels]] = 0
        excluded = cv2.bitwise_and(fuel_mask, cv2.bitwise_not(eligible))
        rejected = int(np.count_nonzero(rejected_ids & (stats[:,cv2.CC_STAT_AREA] >= 6)))
        status = "field tags: red/unknown fuel excluded" if self.hypotheses and self.alliance == "blue" else \
                 "field tags: blue/unknown fuel excluded" if self.hypotheses else "zone unknown: fuel skipped"
        return ZoneResult(eligible, excluded, status, rejected, tag_ids, len(self.hypotheses))
