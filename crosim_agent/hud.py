"""Read alliance score and estimate contained hopper fuel conservatively."""
from collections import deque
from dataclasses import dataclass
import time

import cv2
import numpy as np


HUD_SENSORS_VERSION = "fast-scorecard-match-timer-v4"


def read_hud_number(crop, timeout=None):
    """Tight numeric-word layout: line layout can read a clear 75 as 19."""
    from mosim_env import binarize_score_crop, ocr_read
    if crop.size == 0:
        return None
    binary = binarize_score_crop(crop)
    if float(binary[:10].mean()) < 127:
        binary = cv2.bitwise_not(binary)
    ink = cv2.findNonZero(cv2.bitwise_not(binary))
    if ink is None:
        return None
    x,y,w,h = cv2.boundingRect(ink)
    pad = max(10,h//8)
    tight = cv2.copyMakeBorder(binary[y:y+h,x:x+w],pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=255)
    # Independent layouts must agree; a single plausible misread is not a score.
    deadline = time.perf_counter()+(timeout if timeout is not None else 1.0)
    votes = {}
    for psm in (8,7,13):
        budget = deadline-time.perf_counter()
        if budget <= 0:
            break
        value = ocr_read(tight,timeout=budget,psm=psm)
        if value is not None:
            votes[value] = votes.get(value,0)+1
            if votes[value] >= 2:
                return value
    return None



class HudPanelExtractor:
    def __init__(self, config):
        self.config = dict(config)
        self.status = "looking for blue scorecard"
        self.score_region = config["score_region"]
        self.arrow_region = None
        self._arrow_offset = None
        left,top,right,bottom = self.score_region
        self.timer_region = config.get("timer_region") or [right,top,min(1,right+(right-left)*1.25),bottom]

    def __call__(self, frame):
        height,width = frame.shape[:2]
        scale = min(1.0,960/width)
        small = cv2.resize(frame,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        sh,sw = small.shape[:2]
        hsv = cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv,np.array([95,150,65]),np.array([135,255,255]))
        red = cv2.bitwise_or(cv2.inRange(hsv,np.array([0,150,80]),np.array([10,255,255])),
                            cv2.inRange(hsv,np.array([170,150,80]),np.array([179,255,255])))
        def panels(mask):
            mask[:int(sh*0.65)] = 0
            mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,7),np.uint8))
            contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            return [(x,y,w,h) for contour in contours for x,y,w,h in [cv2.boundingRect(contour)]
                    if sw*0.1 <= w <= sw*0.35 and sh*0.025 <= h <= sh*0.15 and 2 <= w/h <= 9]
        choices = []
        for bx,by,bw,bh in panels(blue):
            for rx,ry,rw,rh in panels(red):
                if rx <= bx+bw or abs((by+bh/2)-(ry+rh/2)) > max(bh,rh)*0.25 \
                        or not 0.6 <= bw/rw <= 1.7 or not 0.65 <= bh/rh <= 1.5:
                    continue
                choices.append((by+bh,bx,by,bw,bh,rx))
        if choices:
            _,x,y,w,h,rx = max(choices)
            crop = frame[round(y/scale):round((y+h)/scale),round(x/scale):round((x+w)/scale)]
            self.status = "auto blue scorecard: alliance score"
            self.score_region = [(x+w*0.56)/sw,y/sh,(x+w*0.98)/sw,(y+h)/sh]
            self.timer_region = [(x+w)/sw,y/sh,rx/sw,(y+h)/sh]
            # Some skins include the arrow column in the blue panel's outer
            # frame, others place it just outside. Cover both layouts.
            if self._arrow_offset is None:
                # Acquire a rectangular yellow scoreboard tile once. Circular
                # field fuel cannot establish or relocate this lock.
                al=max(0,round(x-h*1.2));ar=round(x+w*.25)
                yellow=cv2.inRange(hsv[y:y+h,al:ar],np.array([18,110,130]),np.array([40,255,255]))
                contours,_=cv2.findContours(yellow,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                tiles=[]
                for contour in contours:
                    ax,ay,aw,ah=cv2.boundingRect(contour)
                    polygon=cv2.approxPolyDP(contour,.025*cv2.arcLength(contour,True),True)
                    if len(polygon)==4 and cv2.isContourConvex(polygon) and .75 <= aw/ah <= 1.3 and ah >= h*.35 and cv2.contourArea(contour)/(aw*ah) >= .82:
                        tiles.append((aw*ah,ax+al,ay+y,aw,ah))
                if tiles:
                    _,ax,ay,aw,ah=max(tiles)
                    self._arrow_offset=((ax-x)/w,(ay-y)/h,aw/w,ah/h)
            if self._arrow_offset is not None:
                dx,dy,aw,ah=self._arrow_offset
                self.arrow_region=[max(0,(x+dx*w)/sw),max(0,(y+dy*h)/sh),min(1,(x+(dx+aw)*w)/sw),min(1,(y+(dy+ah)*h)/sh)]
            else:
                self.arrow_region=None
            from mosim_env import crop_frac
            timer = crop_frac(frame,self.timer_region)
            return np.concatenate((cv2.resize(crop,(400,96),interpolation=cv2.INTER_CUBIC),
                                   cv2.resize(timer,(320,96),interpolation=cv2.INTER_CUBIC)),axis=1)
        from mosim_env import crop_frac
        score = crop_frac(frame,self.config["score_region"])
        panel = np.zeros((96,720,3),np.uint8)
        if score.size:
            panel[:,224:400] = cv2.resize(score,(176,96))
        left,top,right,bottom = self.config["score_region"]
        self.timer_region = self.config.get("timer_region") or [right,top,min(1,right+(right-left)*1.25),bottom]
        timer = crop_frac(frame,self.timer_region)
        if timer.size:
            panel[:,400:] = cv2.resize(timer,(320,96))
        self.status = "scorecard not found: configured score ROI"
        self.arrow_region = None
        self.score_region = self.config["score_region"]
        return panel


class ShiftArrowTracker:
    """Debounced yellow-arrow presence beside a positively located scorecard."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = self.pending = None
        self.count = 0
        self.at = self.last_sample = -float("inf")
        self.valid = False

    def observed(self,now=None):
        now = time.perf_counter() if now is None else now
        return self.state if self.valid and 0 <= now-self.at <= .35 else None

    def update(self,frame,region,*,captured_at,now=None):
        now = time.perf_counter() if now is None else now
        self.valid = bool(region) and 0 <= now-captured_at <= .3
        if not self.valid:
            self.pending,self.count = None,0
            return None
        if captured_at <= self.last_sample:
            return self.observed(now)
        self.last_sample = captured_at
        from mosim_env import crop_frac
        crop = crop_frac(frame,region)
        if not crop.size:
            self.valid = False
            return None
        hsv = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,np.array([18,110,130]),np.array([40,255,255]))
        count,_,stats,_ = cv2.connectedComponentsWithStats(mask,8)
        reading = any(area >= max(12,mask.size*.06) and h >= mask.shape[0]*.35
                      and .6 <= w/h <= 1.8 and area/(w*h) >= .4
                      for _,_,w,h,area in stats[1:]) if count > 1 else False
        reading = bool(reading)
        if reading == self.state:
            self.at = captured_at
            self.pending,self.count = None,0
        else:
            self.count = self.count+1 if reading == self.pending else 1
            self.pending = reading
            if self.count >= 3:
                self.state,self.at = reading,captured_at
                self.pending,self.count = None,0
        return self.observed(now)


class HudReader:
    """Score-only API; unverified scorecard tallies never control inventory."""
    def __init__(self, diagnostic_path=None):
        from mosim_env import ScoreReader
        self.score = ScoreReader(ocr_timeout=0.65,read_crop=read_hud_number,max_interval=.5)
        self.error = None
        self.diagnostic_path = diagnostic_path
        self._diagnostic_saved = False
        self._score_error = None
        from .timer import TimerReader
        self.timer = TimerReader()

    @property
    def confirmed(self):
        return self.score.confirmed

    @property
    def _pending(self):
        return self.score._pending

    @property
    def ocr_failures(self):
        return self.score.ocr_failures

    def update_crop(self, panel, now):
        self.error = self._score_error
        failures_before = self.score.ocr_failures
        attempt_before = self.score._last_ocr_time
        try:
            points = self.score.update_crop(panel[:,224:392],now)
        except Exception as exc:
            points = 0
            self.score.ocr_failures += 1
            self.error = f"Score OCR: {type(exc).__name__}: {exc}"
        if self.score.ocr_failures > failures_before:
            self._score_error = self.error or "Score OCR unreadable; see saved HUD crop"
        elif self.score._last_ocr_time != attempt_before:
            self._score_error = None
        self.error = self._score_error
        # Retry baseline promptly instead of waiting two seconds on an unchanged
        # but unreadable crop. A real zero still establishes a normal baseline.
        if self.confirmed is None:
            self.score._last_ocr_time = -1e9
        if panel.shape[1] >= 720:
            self.timer.update(panel[:,400:720],now)
        if self.diagnostic_path and not self._diagnostic_saved and self.ocr_failures >= 3:
            self._diagnostic_saved = True
            try:
                cv2.imwrite(str(self.diagnostic_path),panel)
            except cv2.error:
                pass
        return points


@dataclass(frozen=True)
class InventorySnapshot:
    fill: float = 0.0
    known: bool = False
    source: str = "hopper view pending"
    visual_balls: int = 0
    visual_cells: float = 0.0
    edge_yellow_fraction: float = 0.0
    coverage: float = 0.0
    basket_support: float = 0.0
    raw_fill: float = 0.0


class InventoryTracker:
    """Use basket geometry, ball cores, spatial spread and temporal evidence.

    This is a fullness estimate, never a count of overlapping balls. Missing
    basket geometry preserves the estimate without claiming confidence.
    """
    def __init__(self, config):
        self.config = dict(config)
        self.reset()

    def reset(self):
        self.snapshot = InventorySnapshot()
        self.history = deque(maxlen=7)
        self.last_capture = -float("inf")
        self.read_region = list(self.config["hopper_region"])

    def basket_region(self,frame):
        """Follow the selected robot's cyan basket rim inside a local search.

        Require plausible dimensions around the calibrated basket; otherwise
        retain the calibrated region. This does not search the whole field.
        """
        from mosim_env import crop_frac
        left,top,right,bottom = self.config["hopper_region"]
        width,height = right-left,bottom-top
        search = [max(0,left-width*.25),max(0,top-height*.4),min(1,right+width*.25),min(1,bottom+height*.25)]
        crop = crop_frac(frame,search)
        if not crop.size:
            return [left,top,right,bottom],False
        mask = cv2.inRange(cv2.cvtColor(crop,cv2.COLOR_BGR2HSV),np.array([70,150,140]),np.array([100,255,255]))
        points = cv2.findNonZero(mask)
        if points is None or len(points) < 25:
            return [left,top,right,bottom],False
        x,y,w,h = cv2.boundingRect(points)
        fh,fw = frame.shape[:2]
        region = [search[0]+x/fw,search[1]+y/fh,search[0]+(x+w)/fw,search[1]+(y+h)/fh]
        rw,rh = region[2]-region[0],region[3]-region[1]
        if not (.55*width <= rw <= 1.4*width and .55*height <= rh <= 1.5*height
                and abs((region[0]+region[2]-left-right)/2) <= width*.25
                and .001 <= len(points)/(w*h) <= .3):
            return [left,top,right,bottom],False
        return region,True

    def update(self, frame, *, captured_at=None, now=None, settling=False):
        from mosim_env import crop_frac
        now = time.perf_counter() if now is None else now
        captured_at = now if captured_at is None else captured_at
        self.read_region,rim_found = self.basket_region(frame)
        crop = crop_frac(frame,self.read_region)
        balls,cells,edge_fraction,coverage,support,raw = 0,0.0,0.0,0.0,0.0,0.0
        reliable = crop.size > 0 and not settling and 0 <= now-captured_at <= 0.8
        if reliable:
            crop = cv2.resize(crop,(192,192),interpolation=cv2.INTER_AREA)
            mask = cv2.inRange(cv2.cvtColor(crop,cv2.COLOR_BGR2HSV),
                               np.array(self.config["fuel_hsv_lower"],np.uint8),np.array(self.config["fuel_hsv_upper"],np.uint8))
            interior = np.zeros((192,192),np.uint8)
            polygon = np.array([[.16,.06],[.84,.06],[.94,.72],[.06,.72]])*192
            cv2.fillConvexPoly(interior,polygon.astype(np.int32),255)
            total = int(np.count_nonzero(mask))
            mask = cv2.bitwise_and(mask,interior)
            edge_fraction = 1-np.count_nonzero(mask)/total if total else 0.0
            # Close wire-mesh holes before looking for round ball cores.
            mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
            mask = cv2.bitwise_and(mask,interior)
            coverage = float(np.count_nonzero(mask)/np.count_nonzero(interior))
            distance = cv2.distanceTransform(mask,cv2.DIST_L2,5)
            border_distance = cv2.distanceTransform(interior,cv2.DIST_L2,5)
            peaks = ((distance >= cv2.dilate(distance,np.ones((15,15),np.uint8)))
                     & (distance >= 3) & (distance <= 20) & (border_distance >= 6))
            balls = cv2.connectedComponents(peaks.astype(np.uint8))[0]-1
            occupied = []
            rows = set()
            for y in range(0,192,48):
                for x in range(0,192,48):
                    area = int(np.count_nonzero(interior[y:y+48,x:x+48]))
                    if area > 48*48*.25:
                        present = np.count_nonzero(mask[y:y+48,x:x+48])/area >= .12
                        occupied.append(present)
                        if present:
                            rows.add(y//48)
            cells = float(np.mean(occupied)) if occupied else 0.0
            gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
            # A dark front wall anchors this view to a basket rather than a
            # field pile passing behind the calibrated rectangle.
            support = float(np.mean(gray[145:185,20:172] < 65))
            wall_brightness = float(np.median(gray[145:185,20:172]))
            rim_contrast = float(np.percentile(gray[10:136,22:170],90))-wall_brightness
            reliable = support >= .45 and rim_contrast >= 15 and coverage < .85
            raw = .45*min(balls/12,1)+.35*min(cells/.75,1)+.20*min(coverage/.40,1)
            if balls < 4 or sum(occupied) < 4 or len(rows) < 2:
                raw = min(raw,.65)
            if balls == 0 and coverage < .008:
                raw = 0.0
        if reliable and captured_at > self.last_capture:
            self.history.append((captured_at,float(raw)))
            self.last_capture = captured_at
        if not reliable:
            self.history.clear()
        recent = [value for at,value in self.history if 0 <= now-at <= .8]
        known = reliable and len(recent) >= 3
        fill = float(np.median(recent)) if known else self.snapshot.fill
        source = ("tracked basket rim / spatial median" if rim_found else "calibrated basket / spatial median") if known else "hopper view uncertain; previous load held"
        self.snapshot = InventorySnapshot(fill,known,source,int(balls),cells,float(edge_fraction),
                                           coverage,support,float(raw))
        return self.snapshot
