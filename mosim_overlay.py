"""Click-through Windows overlay and an offline preview of the SAME layout."""
import ctypes
from ctypes import wintypes
from pathlib import Path
import sys
import threading
import time

from mosim_debug import REWARD_LABELS
from mosim_layout import PanelLayout,panel_geometry


BG, BORDER, FG, MUTED = "#101924", "#304154", "#edf4fa", "#91a5b8"
GREEN, RED, CYAN, AMBER = "#69e8a3", "#ff8291", "#6cd5f7", "#ffd26e"
TRANSPARENT = "#010203"


def enable_dpi_awareness():
    if sys.platform == "win32":
        user = ctypes.WinDLL("user32", use_last_error=True)
        try:
            user.SetProcessDpiAwarenessContext.argtypes = (ctypes.c_void_p,)
            user.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except AttributeError:
            user.SetProcessDPIAware()


def exclude_from_capture(hwnd):
    """Set and verify the exact affinity; fail closed before showing any UI."""
    if sys.getwindowsversion().build < 19041:
        raise RuntimeError("Overlay capture exclusion requires Windows 10 2004 or later")
    user = ctypes.WinDLL("user32", use_last_error=True)
    user.SetWindowDisplayAffinity.argtypes = (wintypes.HWND, wintypes.DWORD)
    user.GetWindowDisplayAffinity.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    if not user.SetWindowDisplayAffinity(hwnd, 0x11):
        raise ctypes.WinError(ctypes.get_last_error())
    affinity = wintypes.DWORD()
    if not user.GetWindowDisplayAffinity(hwnd, ctypes.byref(affinity)) or affinity.value != 0x11:
        raise RuntimeError("Could not verify overlay capture exclusion")


def style_window(root):
    user = ctypes.WinDLL("user32", use_last_error=True)
    user.GetAncestor.argtypes = (wintypes.HWND, wintypes.UINT)
    user.GetAncestor.restype = wintypes.HWND
    hwnd = user.GetAncestor(root.winfo_id(), 2)
    get_long = user.GetWindowLongPtrW if ctypes.sizeof(ctypes.c_void_p) == 8 else user.GetWindowLongW
    set_long = user.SetWindowLongPtrW if ctypes.sizeof(ctypes.c_void_p) == 8 else user.SetWindowLongW
    get_long.argtypes = (wintypes.HWND, ctypes.c_int)
    get_long.restype = ctypes.c_ssize_t
    set_long.argtypes = (wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t)
    set_long.restype = ctypes.c_ssize_t
    # LAYERED | TRANSPARENT | TOOLWINDOW | NOACTIVATE.
    style = get_long(hwnd, -20) | 0x80000 | 0x20 | 0x80 | 0x08000000
    ctypes.set_last_error(0)
    if not set_long(hwnd, -20, style) and ctypes.get_last_error():
        raise ctypes.WinError(ctypes.get_last_error())
    exclude_from_capture(hwnd)
    user.SetWindowPos.argtypes = (wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_int, ctypes.c_int, wintypes.UINT)
    user.SetWindowPos(hwnd, wintypes.HWND(-1), 0, 0, 0, 0, 0x01 | 0x02 | 0x10 | 0x20)
    return hwnd


def shorten(value, n=51):
    text = str(value)
    return text if len(text) <= n else text[:n-3] + "..."


def set_click_through(hwnd,enabled):
    user = ctypes.WinDLL("user32",use_last_error=True)
    get_long = user.GetWindowLongPtrW if ctypes.sizeof(ctypes.c_void_p) == 8 else user.GetWindowLongW
    set_long = user.SetWindowLongPtrW if ctypes.sizeof(ctypes.c_void_p) == 8 else user.SetWindowLongW
    get_long.argtypes = (wintypes.HWND,ctypes.c_int)
    get_long.restype = ctypes.c_ssize_t
    set_long.argtypes = (wintypes.HWND,ctypes.c_int,ctypes.c_ssize_t)
    set_long.restype = ctypes.c_ssize_t
    style = get_long(hwnd,-20)
    updated = style | 0x20 if enabled else style & ~0x20
    if updated != style:
        ctypes.set_last_error(0)
        if not set_long(hwnd,-20,updated) and ctypes.get_last_error():
            raise ctypes.WinError(ctypes.get_last_error())


def amount_color(value):
    return GREEN if value > 1e-12 else RED if value < -1e-12 else MUTED


def paint(p, width, height, state, vision, status="LIVE", corner="top-right", now=None,positions=None):
    """Single renderer: Tk at runtime, Pillow for deterministic screenshot QA."""
    now = time.perf_counter() if now is None else now
    ledger = state.get("ledger", {})
    reward_labels = ledger.get("labels", REWARD_LABELS)
    objective = state.get("objective")
    geometry = panel_geometry(width,height,state,corner,positions)
    scale = geometry["scale"]
    ox,oy,pw,ph = geometry["reward"]

    def rect(x1, y1, x2, y2, fill, outline=None):
        p.rect((ox+x1*scale, oy+y1*scale, ox+x2*scale, oy+y2*scale), fill, outline)

    def text(x, y, value, size=12, color=FG, right=False, bold=False, mono=False):
        p.text(ox+x*scale, oy+y*scale, str(value), max(8, int(size*scale)), color,
               "ne" if right else "nw", bold, mono)

    def line(y):
        p.line([(ox+20*scale, oy+y*scale), (ox+420*scale, oy+y*scale)], BORDER)

    # Detections expire instead of hanging on the screen when inference stalls.
    captured = vision.get("captured_at", 0)
    age = max(0.0, now - captured) if captured else float("inf")
    max_age = vision.get("max_age", 0.8)
    boxes = vision.get("boxes", []) if age <= max_age and (status == "LIVE" or status.startswith("PREVIEW")) else []
    for detection in boxes:
        detection_time = detection.get("captured_at") or captured
        if now - detection_time > (0.4 if detection.get("source","").startswith("apriltag_36h11") else max_age):
            continue
        name = detection["name"]
        color = {"fuel": AMBER, "blue_robot": CYAN, "red_robot": RED,
                 "hub": GREEN, "hub_blue_anchor": CYAN, "hub_red_anchor": RED}.get(name, AMBER)
        excluded_fuel = name in ("fuel", "fuel? color") and not detection.get("collectible",True)
        if excluded_fuel:
            color = MUTED
        l, t, r, b = detection["box"]
        xy = (l*width, t*height, r*width, b*height)
        p.rect(xy, None, color)
        # Avoid hundreds of labels over clumps. All boxes remain visible.
        if name not in ("fuel", "fuel? color") or (r-l)*width > 35:
            conf = detection.get("confidence")
            label = f"{name} {conf:.0%}" if conf is not None else name
            if excluded_fuel:
                label = "ignored fuel: red / unverified"
            if detection.get("tag_id") is not None:
                label = f"{detection.get('alliance', '')} hub tag {detection['tag_id']}"
            p.text(xy[0]+3, max(0, xy[1]-17), label, 11, color, "nw", True, False)

    for region in state.get("regions", []):
        l, t, r, b = region["box"]
        color = region.get("color", CYAN)
        p.rect((l*width, t*height, r*width, b*height), None, color)
        p.text(l*width+2, max(0, t*height-19), region["label"], 12, color, "nw", True, False)

    feature_time = state.get("feature_captured_at", 0)
    if feature_time and 0 <= now-feature_time <= 0.25 and (status == "LIVE" or status.startswith("PREVIEW")):
        for target in state.get("agent_targets", []):
            x, y = target["x"]*width, target["y"]*height
            color = target.get("color", CYAN)
            p.line([(x-10, y), (x+10, y)], color)
            p.line([(x, y-10), (x, y+10)], color)
            p.text(max(0, min(x+12, width-180)), max(0, y-20), target["label"], 11, color, "nw", True, False)

    rect(0,0,440,ph/scale,BG,BORDER)
    rect(0,0,4,ph/scale,CYAN)
    text(18,14,"CROSIM / REWARDS",16,bold=True)
    text(420,15,"drag",9,MUTED,right=True)
    live = status == "LIVE"
    stale = state.get("updated_at") and now-state["updated_at"] > 1.5 and live
    warning = state.get("sensor_warning")
    display_status = warning or ("PPO update / waiting" if stale else status)
    mode = {"watch":"WATCH","dry_run":"DRY RUN","agent":"AGENT"}.get(state.get("mode"),"")
    if state.get('human_demo_samples') is not None:
        mode='YOU DRIVE'
        display_status=f"Recording {state['human_demo_samples']} decisions / "+display_status
    text(18,40,shorten((mode+" / " if mode else "")+display_status,56),10,AMBER if warning or stale else GREEN)
    line(62)
    total = ledger.get("total",0)
    text(18,74,"TOTAL REWARD",10,MUTED,bold=True)
    text(18,91,f"{total:+.2f}",29,amount_color(total),bold=True,mono=True)
    credited = state.get("credited_points")
    text(420,74,"REWARDED / ALLIANCE" if credited is not None else "ALLIANCE PTS",10,MUTED,right=True)
    points = f"{credited} / {state.get('points',0)}" if credited is not None else str(state.get("points",0))
    text(420,94,points,25,FG,right=True,bold=True,mono=True)
    seconds = state.get("timer_seconds")
    timer = f"{int(seconds)//60}:{int(seconds)%60:02d}" if seconds is not None else "--:--"
    text(18,130,f"Match {ledger.get('episode',1)}   {timer}",12,mono=True)
    watchdog = state.get("watchdog_remaining")
    reset_text = f"Reset in <= {watchdog:.0f}s" if watchdog is not None else state.get("timer_status","")
    if state.get("timer_status","").startswith("RESET IN"):
        reset_text = state["timer_status"]
    text(420,131,reset_text,10,MUTED,right=True)
    shift = state.get("shift_on")
    shift_text = "Shift ON / arrow" if shift is True else "Shift OFF / no arrow" if shift is False else "Shift: checking arrow"
    text(18,154,shift_text,11,GREEN if shift is True else AMBER)
    known = state.get("inventory_known",True)
    fill = state.get("inventory_fill",state.get("fill_normalized",0))
    text(420,154,f"Hopper ~{fill:.0%}"+(" held" if not known else ""),11,FG if known else MUTED,right=True)
    keys = " ".join(state.get("keys",[])) or "released"
    action = state.get("policy_action","Controls")
    text(18,178,shorten(f"{action}  [{keys}]",62),11,CYAN,mono=True)
    if objective:
        text(18,200,shorten(f"Advice: {objective.get('goal','').upper()} / {objective.get('reason','')}",45),10,MUTED)
    else:
        text(18,200,shorten(vision.get("status","Vision starting"),65),10,MUTED)
    zones = state.get("fuel_zones") or vision.get("fuel_zones") or {}
    if zones.get("excluded_regions") is not None:
        text(420,200,f"ignored {zones['excluded_regions']}",9,MUTED,right=True)
    line(222)
    text(18,234,"REWARD",10,MUTED,bold=True)
    text(302,234,"TICK",10,MUTED,right=True)
    text(420,234,"TOTAL",10,MUTED,right=True)
    for index,(name,label) in enumerate(reward_labels.items()):
        y = 256+index*20
        latest = ledger.get("latest",{}).get(name,0)
        accumulated = ledger.get("totals",{}).get(name,0)
        text(18,y,label,11)
        text(302,y,f"{latest:+.3f}",10,amount_color(latest),right=True,mono=True)
        text(420,y,f"{accumulated:+.2f}",11,amount_color(accumulated),right=True,mono=True)
    table_end = 256+len(reward_labels)*20
    line(table_end+4)
    text(18,table_end+16,"RECENT REWARDS",10,MUTED,bold=True)
    events = list(reversed(ledger.get("events",[])))[:3]
    if not events:
        text(18,table_end+42,"Waiting for rewards...",11,MUTED)
    for index,event in enumerate(events):
        y = table_end+38+index*24
        count = f" x{event['count']}" if event["count"] > 1 else ""
        label = reward_labels.get(event["name"],event["name"])
        text(18,y,f"{event['end']:.1f}s  {label}{count}",10)
        text(420,y,f"{event['amount']:+.3f}",10,amount_color(event["amount"]),right=True,mono=True)
        text(18,y+12,shorten(event.get("reason",""),65),9,MUTED)
    line(table_end+112)
    text(18,table_end+122,"] pause / resume    [ stop    |    Full detail in JSONL",10,CYAN)
    field_map = state.get("field_map") or vision.get("field_map")
    if field_map:
        from mosim_minimap import paint_minimap
        mx,my,_,_ = geometry["minimap"]
        paint_minimap(p,mx,my,scale,field_map,now)


class TkPainter:
    def __init__(self, canvas):
        self.canvas = canvas
        self.images = []

    def rect(self, box, fill=None, outline=None):
        self.canvas.create_rectangle(*box, fill=fill or "", outline=outline or "", width=2)

    def blob(self,box,alpha):
        from PIL import Image,ImageDraw,ImageTk
        from collections import OrderedDict
        if not hasattr(self.canvas,'_blob_cache'):self.canvas._blob_cache=OrderedDict()
        cache=self.canvas._blob_cache
        width,height=max(2,round(box[2]-box[0])),max(2,round(box[3]-box[1]))
        key=(width,height,round(alpha*20))
        if key not in cache:
            bitmap=Image.new('RGBA',(width,height))
            draw=ImageDraw.Draw(bitmap)
            for inset,factor in ((0,.25),(2,.5),(4,1)):
                if width > inset*2 and height > inset*2:
                    draw.ellipse((inset,inset,width-1-inset,height-1-inset),fill=(244,206,81,round(key[2]/20*255*factor)))
            cache[key]=ImageTk.PhotoImage(bitmap,master=self.canvas)
        cache.move_to_end(key)
        self.canvas.create_image(box[0],box[1],anchor='nw',image=cache[key])
        self.images.append(cache[key])
        while len(cache)>128:cache.popitem(last=False)

    def text(self, x, y, value, size, color, anchor, bold, mono):
        self.canvas.create_text(x, y, text=value, fill=color, anchor=anchor,
                                font=("Consolas" if mono else "Segoe UI", -size,
                                      "bold" if bold else "normal"))

    def line(self, points, color):
        self.canvas.create_line(*[v for point in points for v in point], fill=color, width=1.5)


class DebugOverlay:
    """Tk belongs entirely to its own thread. Producers only replace snapshots."""
    def __init__(self, bounds, vision, guard, corner="top-right"):
        self.bounds, self.vision, self.guard, self.corner = bounds, vision, guard, corner
        self.lock = threading.Lock()
        self.state = {}
        self.layout = PanelLayout()
        self.closed = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, name="mosim-overlay", daemon=True)
        self.thread.start()
        if not self.ready.wait(timeout=5):
            self.closed.set()
            raise RuntimeError("Overlay window did not initialize")
        if self.error:
            raise RuntimeError(f"Overlay could not start: {self.error}. Use --no-overlay to log only.")

    def publish(self, state):
        if self.error:
            raise RuntimeError(f"Debug overlay stopped: {self.error}")
        with self.lock:
            self.state = state

    def _run(self):
        root = canvas = None
        try:
            import tkinter as tk
            user = ctypes.WinDLL("user32", use_last_error=True)
            user.GetForegroundWindow.restype = wintypes.HWND
            prior_foreground = user.GetForegroundWindow()
            root = tk.Tk()
            root.withdraw()
            root.overrideredirect(True)
            root.attributes("-transparentcolor", TRANSPARENT)
            root.configure(bg=TRANSPARENT)
            left, top, right, bottom = self.bounds
            width, height = right-left, bottom-top
            root.geometry(f"{width}x{height}{left:+d}{top:+d}")
            canvas = tk.Canvas(root, width=width, height=height, bg=TRANSPARENT,
                               highlightthickness=0, borderwidth=0)
            canvas.pack(fill="both", expand=True)
            root.update_idletasks()
            hwnd = style_window(root)
            # Tk can activate its hidden bootstrap wrapper before we can set
            # NOACTIVATE. Restore focus before exposing the actual overlay, but
            # only if focus is still owned by our UI thread. A user's deliberate
            # switch to another application must be respected.
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetCurrentThreadId.restype = wintypes.DWORD
            user.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
            user.GetWindowThreadProcessId.restype = wintypes.DWORD
            user.SetForegroundWindow.argtypes = (wintypes.HWND,)
            current = user.GetForegroundWindow()
            if prior_foreground and user.GetWindowThreadProcessId(current, None) == kernel.GetCurrentThreadId():
                if not user.SetForegroundWindow(prior_foreground):
                    raise RuntimeError("Could not restore the foreground window after Tk initialization")
            # Show without activation; subsequent paints also preserve focus.
            user.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
            user.ShowWindow(hwnd, 4)  # SW_SHOWNOACTIVATE
            # Tk interprets negative geometry offsets relative to the far edge.
            # Win32 coordinates place left/upper monitors at their true origin.
            user.SetWindowPos.argtypes = (wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                          ctypes.c_int, ctypes.c_int, wintypes.UINT)
            if not user.SetWindowPos(hwnd, wintypes.HWND(-1), left, top, width, height, 0x10 | 0x40):
                raise ctypes.WinError(ctypes.get_last_error())
            self.ready.set()
            user.GetCursorPos.argtypes = (ctypes.POINTER(wintypes.POINT),)
            user.GetAsyncKeyState.argtypes = (ctypes.c_int,)
            user.GetAsyncKeyState.restype = ctypes.c_short
            user.SetCapture.argtypes = (wintypes.HWND,)
            user.SetCapture.restype = wintypes.HWND
            down_before = False
            click_through = True

            # Keep the event pump and every Tk reference on this thread. A
            # recursive after() closure can otherwise keep the Tcl interpreter
            # alive until main-thread GC, which crashes on Windows at shutdown.
            while not self.closed.is_set():
                with self.lock:
                    state = self.state
                vision = self.vision.latest()
                geometry = panel_geometry(width,height,state,self.corner,self.layout.positions)
                point = wintypes.POINT()
                user.GetCursorPos(ctypes.byref(point))
                px,py = point.x-left,point.y-top
                down = bool(user.GetAsyncKeyState(1) & 0x8000)
                show_map = bool(state.get("field_map") or vision.get("field_map"))
                hot = self.layout.header_at(px,py,geometry,show_map)
                if down and not down_before and hot:
                    self.layout.begin(px,py,geometry,show_map)
                    user.SetCapture(hwnd)
                if down and self.layout.drag:
                    self.layout.move(px,py,width,height)
                if not down and self.layout.drag:
                    self.layout.end()
                    user.ReleaseCapture()
                transparent = not (self.layout.drag or hot)
                if transparent != click_through:
                    set_click_through(hwnd,transparent)
                    click_through = transparent
                canvas.configure(cursor="fleur" if hot or self.layout.drag else "")
                down_before = down
                canvas.delete("all")
                paint(TkPainter(canvas), width, height, state, vision,
                      self.guard.status, self.corner,positions=self.layout.positions)
                root.update_idletasks()
                root.update()
                self.closed.wait(0.033)
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.ready.set()
            if self.layout.drag:
                self.layout.end()
            if root is not None:
                root.destroy()
            canvas = root = None
            import gc
            gc.collect()

    def close(self):
        self.closed.set()
        self.thread.join(timeout=2.0)


def save_preview(image_path, output_path, state, vision, corner="top-right"):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image,"RGBA")

    class Painter:
        fonts = {}

        def blob(self,box,alpha):
            draw.ellipse(box,fill=(244,206,81,round(255*alpha)))

        def rect(self, box, fill=None, outline=None):
            draw.rectangle(box, fill=fill, outline=outline, width=2)

        def text(self, x, y, value, size, color, anchor, bold, mono):
            key = (size, bold, mono)
            if key not in self.fonts:
                filename = ("consolab.ttf" if bold else "consola.ttf") if mono else (
                    "segoeuib.ttf" if bold else "segoeui.ttf")
                self.fonts[key] = ImageFont.truetype(str(Path("C:/Windows/Fonts") / filename), size)
            draw.text((x, y), value, font=self.fonts[key], fill=color,
                      anchor="rt" if anchor == "ne" else "lt")

        def line(self, points, color):
            draw.line(points, fill=color, width=2)

    paint(Painter(), *image.size, state, vision, "PREVIEW: synthetic reward values", corner)
    image.save(output_path)
