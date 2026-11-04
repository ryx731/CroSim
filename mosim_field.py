"""Estimated field map from tag camera poses and the robot's floor contact.

The chase camera's center is not the robot's position. Project the lower basket
anchor onto the floor, allow for the rear-to-center offset, and keep the full
range of accepted camera hypotheses. Heading corrects for H camera parity.
"""
from collections import deque
import math
import time
import json
from pathlib import Path
import uuid

import numpy as np

from mosim_zones import FIELD_LENGTH,FIELD_WIDTH,ALLIANCE_ZONE_DEPTH


FIELD_MAP_VERSION = "bidirectional-multitag-pose-v4"
POSE_PATH = Path(__file__).with_name("mosim_pose_state.json")
FIELD = {"length":FIELD_LENGTH,"width":FIELD_WIDTH,"blue_end":ALLIANCE_ZONE_DEPTH,
         "red_start":FIELD_LENGTH-ALLIANCE_ZONE_DEPTH}


class FieldPoseTracker:
    def __init__(self,alliance="blue",persist=False):
        self.alliance = alliance
        self.persist = persist
        self._saved_at = -math.inf
        self.pose = None
        try:
            saved = json.loads(POSE_PATH.read_text())
            if (np.isfinite([saved[k] for k in ("x","y","heading","uncertainty_m")]).all()
                    and len(saved["bounds"]) == 4 and np.isfinite(saved["bounds"]).all()):
                self.pose = {**saved,"captured_at":-1e9,"source":"restored last pose"}
        except (OSError,ValueError,KeyError,TypeError):
            pass
        self.reset()

    def reset(self):
        if self.pose is None:
            x = 2.0 if self.alliance == "blue" else FIELD_LENGTH-2
            y = FIELD_WIDTH/2
            self.pose = {"x":x,"y":y,"heading":0.0 if self.alliance == "blue" else math.pi,
                         "uncertainty_m":3.0,"bounds":[x-2,y-2,x+2,y+2],"captured_at":-1e9,
                         "hypotheses":0,"source":"assumed initial pose"}
        self._has_measurement = False
        self._variance = 25.0
        self.trail = deque(maxlen=80)
        if not hasattr(self,"_region"):
            self._region = self.region_at(self.pose["x"])
            self._region_since = time.perf_counter()
        self._region_candidate = None
        self._region_candidate_at = None
        self._region_measured = False
        self._reacquire = None
        self._reacquire_count = 0
        self._fuel_clusters = []

    @staticmethod
    def region_at(x):
        return "blue" if x < FIELD["blue_end"] else "red" if x > FIELD["red_start"] else "neutral"

    def track_region(self,current,now):
        """Hold the last region through tag gaps; debounce boundary crossings."""
        if current:
            x = self.pose["x"]
            candidate = self.region_at(x)
            boundary = FIELD["blue_end"] if "blue" in (candidate,self._region) else FIELD["red_start"]
            if candidate != self._region and abs(x-boundary) < .20:
                candidate = self._region
            if candidate != self._region_candidate:
                self._region_candidate,self._region_candidate_at = candidate,now
            if not self._region_measured or now-self._region_candidate_at >= .3:
                if candidate != self._region:
                    self._region_since = now
                self._region = candidate
            self._region_measured = True
        else:
            self._region_candidate,self._region_candidate_at = None,None
        return {"region":self._region,"label":{"blue":"BLUE ALLIANCE","neutral":"NEUTRAL FIELD","red":"RED ALLIANCE"}[self._region],
                "source":"vision estimate" if current else "last region held",
                "seconds":max(0,now-self._region_since)}

    @staticmethod
    def project(pixels,hypotheses,offset=0.0,allow_partial=False):
        points,headings = [],[]
        for h in hypotheses:
            point,valid = h.ground_points(np.asarray(pixels)*[h.width,h.height],plane_height=0.0)
            if not valid.all():
                if allow_partial: continue
                return None
            forward = np.array(h.rotation[2,:2],dtype=float,copy=True)
            norm = float(np.linalg.norm(forward))
            if norm < .1:
                if allow_partial: continue
                return None
            forward /= norm
            xy = point[:,:2]+offset*forward
            if not np.isfinite(xy).all() or (xy[:,0] < -.5).any() or (xy[:,0] > FIELD_LENGTH+.5).any() \
                    or (xy[:,1] < -.5).any() or (xy[:,1] > FIELD_WIDTH+.5).any():
                if allow_partial: continue
                return None
            points.append(xy)
            headings.append(forward)
        return (np.asarray(points),np.asarray(headings)) if points else None

    def update(self,hypotheses,robot_region,*,captured_at=None,now=None,detections=(),fuel_mask=None,reversed_view=False):
        now = time.perf_counter() if now is None else now
        captured_at = now if captured_at is None else captured_at
        current = False
        status = "Holding last given pose"
        zone,opponent,blue_zone = "unknown",None,None
        peers = []
        region_valid = (len(robot_region) == 4 and np.isfinite(robot_region).all()
                        and 0 <= robot_region[0] < robot_region[2] <= 1
                        and 0 <= robot_region[1] < robot_region[3] <= 1)
        ordered = not self._has_measurement or captured_at >= self.pose["captured_at"]
        if hypotheses and region_valid and ordered and 0 <= now-captured_at <= .3:
            l,t,r,b = robot_region
            projection = self.project([[(l+r)/2,b]],hypotheses,offset=.25,allow_partial=True)
            if projection is not None:
                points,headings = projection
                points = points[:,0,:]
                center = np.median(points,axis=0)
                # Include contact-point / robot-center calibration uncertainty.
                low,high = points.min(axis=0)-.55,points.max(axis=0)+.55
                uncertainty = float(np.linalg.norm(np.maximum(center-low,high-center)))
                forward = headings.mean(axis=0)
                heading = math.atan2(float(forward[1]),float(forward[0]))
                if reversed_view:
                    heading += math.pi
                    heading = (heading+math.pi)%(2*math.pi)-math.pi
                plausible = True
                if self._has_measurement and 0 <= captured_at-self.pose["captured_at"] <= .8:
                    jump = float(np.linalg.norm(center-[self.pose["x"],self.pose["y"]]))
                    limit = max(1.0,(captured_at-self.pose["captured_at"])*7)+uncertainty+self.pose["uncertainty_m"]
                    plausible = jump <= limit
                if not plausible:
                    if self._reacquire is not None and captured_at > self._reacquire[1] and captured_at-self._reacquire[1] <= .4 and np.linalg.norm(center-self._reacquire[0]) <= .8:
                        self._reacquire_count += 1
                    else:
                        self._reacquire_count = 1
                    self._reacquire = (center.copy(),captured_at)
                    if self._reacquire_count >= 3:
                        plausible = True
                        self._has_measurement = False
                if plausible:
                    self._reacquire,self._reacquire_count = None,0
                    if self._has_measurement:
                        dt = max(0,captured_at-self.pose["captured_at"])
                        predicted = min(25,self._variance+.25*dt)
                        gain = predicted/(predicted+max(.1,uncertainty**2))
                        # Track a moving robot rather than indefinitely
                        # anchoring it to a once-low covariance estimate.
                        gain = max(.25 if uncertainty <= 2 else .1,gain)
                        previous = np.array([self.pose["x"],self.pose["y"]])
                        center = previous+gain*(center-previous)
                        delta = (heading-self.pose["heading"]+math.pi)%(2*math.pi)-math.pi
                        heading = self.pose["heading"]+gain*delta
                        low,high = np.minimum(low,center-.55),np.maximum(high,center+.55)
                        uncertainty = float(np.linalg.norm(np.maximum(center-low,high-center)))
                        self._variance = (1-gain)*predicted
                    else:
                        self._variance = uncertainty**2
                    self.pose = {"x":float(center[0]),"y":float(center[1]),"heading":heading,
                                 "uncertainty_m":uncertainty,"bounds":[float(low[0]),float(low[1]),float(high[0]),float(high[1])],
                                 "captured_at":float(captured_at),"hypotheses":len(hypotheses),
                                 "heading_source":"reverse-camera corrected bearing" if reversed_view else "vision bearing; held between corrections",
                                 "camera_view":"reversed" if reversed_view else "normal","source":"tag correction"}
                    self._has_measurement = True
                    if self.persist and now-self._saved_at >= 1:
                        temporary = POSE_PATH.with_name(POSE_PATH.name+"."+uuid.uuid4().hex+".tmp")
                        try:
                            temporary.write_text(json.dumps(self.pose))
                            temporary.replace(POSE_PATH)
                            self._saved_at = now
                        except OSError:
                            pass
                        finally:
                            temporary.unlink(missing_ok=True)
                    current = True
                    if not self.trail or math.hypot(center[0]-self.trail[-1][0],center[1]-self.trail[-1][1]) >= .12:
                        self.trail.append([float(center[0]),float(center[1])])
                    confident = uncertainty <= 1.75 and len(points) == len(hypotheses)
                    if confident:
                        blue_zone = True if high[0] < FIELD["blue_end"]-.15 else False if low[0] > FIELD["blue_end"]+.15 else None
                        if low[0] > FIELD["red_start"]+.15:
                            zone = "red"
                        elif high[0] < FIELD["blue_end"]-.15:
                            zone = "blue"
                        elif low[0] > FIELD["blue_end"]+.15 and high[0] < FIELD["red_start"]-.15:
                            zone = "neutral"
                        else:
                            zone = "boundary uncertain"
                        if self.alliance == "blue":
                            opponent = True if low[0] > FIELD["red_start"]+.15 else False if high[0] < FIELD["red_start"]-.15 else None
                        else:
                            opponent = True if high[0] < FIELD["blue_end"]-.15 else False if low[0] > FIELD["blue_end"]+.15 else None
                    status = "Tag pose estimate" if confident else "Pose range too wide for zone reward"
                else:
                    status = "Pose jump rejected; last known position"
            if current:
                for detection in detections:
                    if detection.get("name") not in {"blue_robot","red_robot"}:
                        continue
                    at = detection.get("captured_at") or captured_at
                    if not 0 <= now-at <= .3:
                        continue
                    a,c,d,e = detection["box"]
                    result = self.project([[(a+d)/2,e]],hypotheses)
                    if result is None:
                        continue
                    values = result[0][:,0,:]
                    xy = np.median(values,axis=0)
                    if float(np.max(np.linalg.norm(values-xy,axis=1))) <= 1.75:
                        peers.append({"x":float(xy[0]),"y":float(xy[1]),"alliance":detection["name"].split("_")[0]})
        location = self.track_region(current,now)
        if fuel_mask is not None and hypotheses and 0 <= now-captured_at <= .3:
            self._fuel_clusters = project_fuel_clusters(fuel_mask,hypotheses,captured_at)
        self._fuel_clusters = [v for v in self._fuel_clusters if 0 <= now-v['captured_at'] <= 2]
        return {"version":FIELD_MAP_VERSION,"field":dict(FIELD),"pose":dict(self.pose) if self.pose else None,
                "pose_current":current,"zone":zone,"opponent_zone":opponent,
                "blue_zone":blue_zone,"location":location,
                "fuel_clusters":list(self._fuel_clusters),
                "trail":list(self.trail),"robots":peers,"status":status}


def project_fuel_clusters(mask,hypotheses,captured_at):
    """Map large visible yellow groups; confidence is geometric agreement,
    not a calibrated probability or an individual ball count."""
    import cv2
    groups = cv2.dilate(mask,np.ones((13,17),np.uint8))
    count,labels,_,_ = cv2.connectedComponentsWithStats(groups,8)
    ys,xs = np.nonzero(mask)
    if not len(xs): return []
    ids = labels[ys,xs]
    masses = np.bincount(ids,minlength=count)
    blobs = []
    for index in np.argsort(masses[1:])[::-1][:12]+1:
        if masses[index] < 30: continue
        selected = ids == index
        # Lower yellow pixels approximate the floor contact of a pile.
        pixel = [[float(np.median(xs[selected]))/mask.shape[1],float(np.quantile(ys[selected],.9))/mask.shape[0]]]
        result = FieldPoseTracker.project(pixel,hypotheses)
        if result is None: continue
        positions = result[0][:,0,:]
        center = np.median(positions,axis=0)
        spread = float(np.max(np.linalg.norm(positions-center,axis=1)))
        if spread > 2.5: continue
        blobs.append({'x':float(center[0]),'y':float(center[1]),
                      'radius':min(1.4,.25+math.sqrt(float(masses[index]))*.025),
                      'confidence':float(1/(1+spread)), 'captured_at':float(captured_at)})
    return blobs
