"""Match-clock OCR plus a wall-clock watchdog for missed end frames."""
import math
import re
import time

import cv2
import numpy as np


HARD_MATCH_SECONDS = 165.0


def read_match_timer(crop,timeout=.6):
    from mosim_env import need_pytesseract
    if not crop.size:
        return None
    gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    if float(gray.std()) < 5:
        return None
    gray = cv2.resize(gray,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
    _,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if float(np.concatenate((binary[0],binary[-1],binary[:,0],binary[:,-1])).mean()) < 127:
        binary = cv2.bitwise_not(binary)
    ink = cv2.findNonZero(cv2.bitwise_not(binary))
    if ink is None:
        return None
    x,y,w,h = cv2.boundingRect(ink)
    binary = cv2.copyMakeBorder(binary[y:y+h,x:x+w],12,12,12,12,cv2.BORDER_CONSTANT,value=255)
    contours,hierarchy = cv2.findContours(cv2.bitwise_not(binary),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        boxes = [(i,cv2.boundingRect(c)) for i,c in enumerate(contours) if hierarchy[0,i,3] == -1]
        tallest = max((b[3] for _,b in boxes),default=0)
        digits = [(i,b) for i,b in boxes if b[3] >= tallest*.65 and b[2] >= 5]
        zero_shapes = len(digits) == 3
        for index,(x,y,w,h) in digits:
            children = [cv2.boundingRect(c) for i,c in enumerate(contours) if hierarchy[0,i,3] == index]
            zero_shapes &= (len(children) == 1 and .35 <= (children[0][1]+children[0][3]/2-y)/h <= .65
                            and children[0][3]/h >= .4)
        if zero_shapes:
            return 0
    started = time.perf_counter()
    for psm in (7,8):
        budget = timeout-(time.perf_counter()-started)
        if budget <= 0:
            return None
        text = need_pytesseract().image_to_string(binary,config=f"--psm {psm} -c tessedit_char_whitelist=0123456789:",timeout=budget)
        cleaned = re.sub(r"\s+","",text)
        if cleaned in {"000","0000"}:
            return 0
        match = re.fullmatch(r"([0-3]):([0-5][0-9])",cleaned)
        if match:
            return int(match[1])*60+int(match[2])
    return None


class TimerReader:
    def __init__(self):
        self.seconds = None
        self.at = 0.0
        self.pending = None
        self.pending_at = 0.0
        self.last_attempt = -math.inf
        self.error = None
        self.zero_pending = False

    def update(self,crop,now):
        if now-self.last_attempt < .25:
            return
        self.last_attempt = now
        try:
            value = read_match_timer(crop)
            self.error = None if value is not None else "Match timer unreadable"
        except Exception as exc:
            value = None
            self.error = f"Timer OCR: {type(exc).__name__}: {exc}"
        if value is None:
            self.pending = None
            return
        if value > 180:
            self.error = "Implausible match timer rejected"
            return
        # A previous bad minute digit must never lock out a real 0:00.
        if value == 0:
            if self.zero_pending:
                self.seconds,self.at = 0,now
                self.pending = None
            self.zero_pending = True
            return
        self.zero_pending = False
        if self.seconds is not None:
            drop_limit = max(3,now-self.at+2)
            if self.seconds-value > drop_limit:
                self.error = "Timer jump rejected"
                return
            if value <= self.seconds+2:
                self.seconds,self.at = value,now
                self.pending = None
                return
        # Establish a baseline, or confirm an externally restarted clock.
        if self.pending is not None and 0 <= self.pending-value <= max(2,now-self.pending_at+1):
            self.seconds,self.at = value,now
            self.pending = None
        else:
            self.pending,self.pending_at = value,now


class MatchClock:
    def __init__(self,zero_hold=3.0,zero_reads=1):
        self.zero_hold,self.required_reads = zero_hold,zero_reads
        self.seconds = None
        self.at = -math.inf
        self.ended = False
        self.awaiting_positive = False
        self.zero_since = None
        self.zero_count = 0

    def observe(self,seconds,captured_at,now=None):
        now = time.perf_counter() if now is None else now
        if seconds is None or not 0 <= seconds <= 359 or captured_at <= self.at or not 0 <= now-captured_at <= 2:
            return
        self.seconds,self.at = int(seconds),captured_at
        if seconds > 0:
            self.ended,self.awaiting_positive = False,False
            self.zero_since,self.zero_count = None,0
        elif not self.awaiting_positive:
            if self.zero_since is None:
                self.zero_since = captured_at
            self.zero_count += 1
            self.ended = self.zero_count >= self.required_reads

    def remaining(self,now=None):
        now = time.perf_counter() if now is None else now
        return self.seconds if 0 <= now-self.at <= 3 else None

    def can_restart(self,now=None):
        now = time.perf_counter() if now is None else now
        return (self.ended and self.seconds == 0 and self.zero_since is not None
                and now-self.zero_since >= self.zero_hold and not self.awaiting_positive)

    def restarted(self):
        self.seconds,self.at = None,-math.inf
        self.ended,self.awaiting_positive = False,True
        self.zero_since,self.zero_count = None,0
