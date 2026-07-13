"""In-process session state shared across GUI windows.

Class names and their per-class settings deliberately live in MEMORY, not on
disk: the user wants them to survive moving between images and round trips
through the main menu, but a fresh app launch must always start from one
unnamed class. Nothing here is ever written to ~/.autoannotate.
"""
import sys

STATE = {
    # Box-prompt class names, index == class id. None = the one-class default.
    "box_class_names": None,
    # Per-class threshold overrides, {class_id: {"det", "seg", "max_area"}},
    # each value 0.0-1.0. A class id with no entry uses the global sliders.
    "class_settings": {},
    # "trackpad" | "mouse" | None = pick by platform (macOS gets trackpad).
    "input_scheme": None,
}


def reset():
    """Back to launch state. Exists for tests; the app never calls it."""
    STATE["box_class_names"] = None
    STATE["class_settings"] = {}
    STATE["input_scheme"] = None


def input_scheme():
    """The active pointer scheme: the user's explicit pick from the Image
    Resize menu, else the platform default (trackpads on macOS laptops, mice
    everywhere else)."""
    s = STATE.get("input_scheme")
    if s in ("trackpad", "mouse"):
        return s
    return "trackpad" if sys.platform == "darwin" else "mouse"


def classify_wheel(scheme, dx, dy, ctrl=False, shift=False):
    """What a wheel event means in Image Resize mode, per scheme.

    Returns ("zoom", amount), ("pan", dx, dy) or None.

    Mouse: the wheel ZOOMS, always, with no modifier keys involved; panning is
    a right-button drag handled by the views, so shift/ctrl are deliberately
    ignored here (the user wants plain mouse inputs only).

    Trackpad: a two-finger scroll PANS on both axes (the left button stays
    reserved for drawing on the zoomed view) and Ctrl/Cmd + scroll zooms,
    alongside the native pinch gesture."""
    if scheme == "mouse" or ctrl:
        amount = dy or dx
        return ("zoom", amount) if amount else None
    if dx == 0 and dy == 0:
        return None
    return ("pan", dx, dy)
