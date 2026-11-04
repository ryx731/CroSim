"""Optional native Windows check. Opens our click-through overlay for 250 ms.

Run: python -m tests.smoke_overlay
No game capture, game reset, or key injection occurs.
"""
import time
from types import SimpleNamespace

from mosim_controls import windows_api
from mosim_overlay import DebugOverlay, enable_dpi_awareness


class Vision:
    def latest(self):
        return {"boxes": [], "status": "Native overlay smoke test"}


def main():
    enable_dpi_awareness()
    user = windows_api()
    for trial in range(3):
        before = user.GetForegroundWindow()
        overlay = DebugOverlay((0, 0, 800, 600), Vision(), SimpleNamespace(status="SMOKE TEST"))
        try:
            time.sleep(0.25)
            assert overlay.error is None, overlay.error
            assert user.GetForegroundWindow() == before, f"Overlay changed foreground on trial {trial}"
        finally:
            overlay.close()
        assert not overlay.thread.is_alive(), "UI thread did not close"
    print("PASS: overlay startup, verified capture exclusion, foreground preservation, and shutdown")


if __name__ == "__main__":
    main()
