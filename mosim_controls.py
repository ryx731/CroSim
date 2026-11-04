"""Global hotkeys and focus gating, independent of the inference/UI threads."""
import ctypes
from ctypes import wintypes
import os
import re
import threading
import time


def windows_api():
    if os.name != "nt":
        raise RuntimeError("Live game controls require Windows")
    user = ctypes.WinDLL("user32", use_last_error=True)
    user.GetForegroundWindow.restype = wintypes.HWND
    user.GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
    user.GetWindowTextLengthW.argtypes = (wintypes.HWND,)
    user.GetAsyncKeyState.argtypes = (ctypes.c_int,)
    user.GetAsyncKeyState.restype = ctypes.c_short
    return user


def foreground_title(user):
    hwnd = user.GetForegroundWindow()
    title = ctypes.create_unicode_buffer(user.GetWindowTextLengthW(hwnd) + 1)
    user.GetWindowTextW(hwnd, title, len(title))
    return title.value


def game_title_matches(expected, actual):
    """Support both released game names without matching mosim_env.py consoles."""
    expected = expected.casefold().strip()
    if not expected:
        return False
    if expected in {"mosim", "mosimulator"}:
        return re.search(r"\bmosim(?:ulator)?\b", actual, re.IGNORECASE) is not None
    return expected in actual.casefold()


class InputGuard:
    """']' arms/pauses, '[' stops. Focus loss requires a fresh ']' to resume.

    The input lock is shared with key injection, so a pause cannot race with a
    new keyDown. The polling thread releases keys even during a long PPO update.
    """
    def __init__(self, release, title="MoSim", user=None, input_lock=None):
        self.user = user if user is not None else windows_api()
        self.title = title.casefold().strip()
        if not self.title:
            raise ValueError("game_window_title must contain part of the game window title")
        self.release = release
        self.input_lock = input_lock or threading.RLock()
        self.stopped = threading.Event()
        self.closed = threading.Event()
        self.paused = True
        self.status = f"WAITING: focus {title}, then press ]"
        self.thread = threading.Thread(target=self._poll, name="mosim-safety", daemon=True)
        self.thread.start()

    def focused(self):
        return game_title_matches(self.title, foreground_title(self.user))

    def _wrong_window_status(self):
        actual = foreground_title(self.user) or "no active window"
        return f"PAUSED: focus {self.title}; current: {actual[:55]}"

    def _pause(self, message):
        self.paused = True
        self.status = message
        self.release()

    def _poll(self):
        previous_go = bool(self.user.GetAsyncKeyState(0xDD) & 0x8000)
        try:
            while not self.closed.wait(0.01):
                with self.input_lock:
                    if self.user.GetAsyncKeyState(0xDB) & 0x8000:
                        self.stopped.set()
                        self._pause("STOPPED: [ pressed")
                        return
                    go = bool(self.user.GetAsyncKeyState(0xDD) & 0x8000)
                    focused = self.focused()
                    if not focused and not self.paused:
                        self._pause(self._wrong_window_status())
                    if go and not previous_go and not focused:
                        self._pause(self._wrong_window_status())
                        print(f"[controls] ] detected, but {self.status}", flush=True)
                    if go and not previous_go and focused:
                        if self.paused:
                            self.paused = False
                            self.status = "LIVE"
                        else:
                            self._pause("PAUSED: press ] to resume")
                    previous_go = go
        except Exception as exc:
            with self.input_lock:
                self.stopped.set()
                self._pause(f"STOPPED: safety monitor failed ({exc})")

    def check_input(self):
        """Call under input_lock immediately before every keyDown."""
        if self.stopped.is_set() or self.closed.is_set():
            raise KeyboardInterrupt
        if not self.focused():
            self._pause(self._wrong_window_status())
            return False
        if self.paused:
            self.status = "PAUSED: game focused; press ] to start / resume"
            return False
        return True

    def wait(self):
        start = time.perf_counter()
        while True:
            with self.input_lock:
                if self.check_input():
                    return time.perf_counter() - start
            time.sleep(0.02)

    def close(self):
        self.closed.set()
        self.thread.join(timeout=1.0)
        with self.input_lock:
            self.release()
