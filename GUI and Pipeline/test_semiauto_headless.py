"""Headless logic tests for the semi-automatic SAM segmentation feature.

Runs the real GUI code (autoannotate.gui.*) under an offscreen Qt platform,
with the heavy ML deps (cv2, ultralytics, groundingdino, ...) stubbed. It
exercises the pure GUI logic: gating, point capture, coordinate mapping, key
handling, commit, and toolbar mutual-exclusion, NOT real SAM inference or live
rendering.

Run with the repo venv:
    QT_QPA_PLATFORM=offscreen .venv/bin/python "GUI and Pipeline/test_semiauto_headless.py"
"""
import os, sys, types
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# ── Stub the heavy modules so the package can import without them ─────────
def _stub(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m
cv2 = _stub("cv2")
cv2.cvtColor = lambda *a, **k: None
cv2.COLOR_BGR2RGB = 4
import numpy as _np
cv2.imread = lambda *a, **k: _np.zeros((100, 100, 3), dtype=_np.uint8)
# autoannotate.imageio resolves these at call time (never as default args, which
# would blow up against this stub at import). Present so a real call path works.
cv2.IMREAD_COLOR = 1
cv2.IMREAD_GRAYSCALE = 0

_gd = _stub("groundingdino"); _gdu = _stub("groundingdino.util")
_gdi = _stub("groundingdino.util.inference")
_gd.util = _gdu; _gdu.inference = _gdi
_gdi.load_model = lambda *a, **k: object()
_gdi.load_image = lambda *a, **k: (None, None)
_gdi.predict = lambda *a, **k: ([], [], [])

_ul = _stub("ultralytics")
_ul.SAM = lambda *a, **k: object()

_de = _stub("dotenv"); _de.load_dotenv = lambda *a, **k: None
_hf = _stub("huggingface_hub"); _hf.login = lambda *a, **k: None
# transformers is stubbed EMPTY on purpose: ensure_llm's lazy
# `from transformers import ...` must raise ImportError so the VLM tests
# exercise the degrade-gracefully path instead of downloading a model.
_stub("transformers")

from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np

# ── Import the real GUI modules from the package ──────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import autoannotate.gui.manual_window as _mod_mw
import autoannotate.gui.canvas as _mod_canvas
import autoannotate.gui.dialogs as _mod_dialogs
import autoannotate.gui.side_by_side as _mod_sbs
import autoannotate.gui.splash as _mod_splash
import autoannotate.gui.automated_window as _mod_auto
import autoannotate.gui.llm as _mod_llm
import autoannotate.gui.spatial as _mod_spatial
import autoannotate.gui.style as _mod_style
import autoannotate.pipeline.labels as _mod_labels

class _FlatNamespace:
    """Dict-like view over the module set, mirroring the flat namespace the
    suite used when the GUI was a single exec'd notebook cell. Reads find the
    first module holding the name; writes patch EVERY module holding it, so a
    per-test stub lands in the module where the name is actually used."""
    def __init__(self, mods):
        self._mods = mods
    def __getitem__(self, k):
        for m in self._mods:
            if hasattr(m, k):
                return getattr(m, k)
        raise KeyError(k)
    def __setitem__(self, k, v):
        hit = False
        for m in self._mods:
            if hasattr(m, k):
                setattr(m, k, v)
                hit = True
        if not hit:
            setattr(self._mods[0], k, v)
    def get(self, k, default=None):
        try:
            return self[k]
        except KeyError:
            return default
    def pop(self, k, default=None):
        v = self.get(k, default)
        for m in self._mods:
            if hasattr(m, k):
                delattr(m, k)
        return v

G = _FlatNamespace([_mod_mw, _mod_canvas, _mod_dialogs, _mod_sbs, _mod_splash,
                    _mod_auto, _mod_llm, _mod_spatial, _mod_style, _mod_labels])

# Default stubs, mirroring the old harness: model loads and pipeline calls
# are inert unless a test overrides them.
G["result_clean_polys"] = lambda r: []
G["load_dino_model"] = lambda *a, **k: object()
G["load_sam"] = lambda *a, **k: object()
G["load_yoloe"] = lambda *a, **k: object()
G["save_masks"] = lambda *a, **k: None
G["adjust_masks"] = lambda *a, **k: []
G["overlay_with_borders"] = lambda img, *a, **k: img
G["draw_boxes_on_image"] = lambda img, *a, **k: img
# Image I/O goes through autoannotate.imageio (unicode-safe on Windows), not
# cv2.imread/imwrite. Stub it the way cv2.imread used to be stubbed: decode
# yields a blank image, encode succeeds without touching the disk.
G["imread_unicode"] = lambda *a, **k: _np.zeros((100, 100, 3), dtype=_np.uint8)
G["imwrite_unicode"] = lambda *a, **k: True

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
AnnotationCanvas = G["AnnotationCanvas"]
ManualWindow = G["ManualWindow"]
SemiAutoSettingsDialog = G["SemiAutoSettingsDialog"]
SpatialGrid = G["SpatialGrid"]
SideBySideWindow = G["SideBySideWindow"]


# ── tiny test runner ──────────────────────────────────────────────────────
_results = []
def check(name, cond, detail=""):
    _results.append((name, bool(cond), detail))
    print(("PASS" if cond else "FAIL"), "-", name, ("" if cond else f"  [{detail}]"))

def mk_window():
    """Bare ManualWindow: skips the heavy __init__/model load but initializes
    the QWidget base so Qt signal delivery to bound-method slots works."""
    w = ManualWindow.__new__(ManualWindow)
    QtWidgets.QWidget.__init__(w)
    # __init__ is skipped above, so restore the simple flags the exercised
    # methods rely on (mirrors ManualWindow.__init__). _in_mode_switch guards the
    # tool-select reentry path hit when a model change forces a fallback to Boxes.
    w._in_mode_switch = False
    return w

class StubLabel:
    """Records canvas mode calls without needing a real widget."""
    def __init__(self, semiauto_masks=False):
        self.calls = []
        self._orig_w = 100; self._orig_h = 100
        self._semiauto_masks = semiauto_masks
    def has_semiauto_masks(self):
        return self._semiauto_masks
    def has_editable_masks(self):
        # Any editable polygon (model-generated masks included), independent of
        # whether a SAM model is loaded. Stubbed to the test's mask-presence flag.
        return self._semiauto_masks
    def __getattr__(self, n):
        def rec(*a, **k): self.calls.append((n, a)); return None
        return rec

def mk_draw_button():
    b = QtWidgets.QToolButton(); b.setCheckable(True); return b

def mk_action(checked=False):
    a = QtWidgets.QAction("x"); a.setCheckable(True); a.setChecked(checked); return a

# ══════════════════════════════════════════════════════════════════════════
# T1: _active_interactive_sam_key decision table
# ══════════════════════════════════════════════════════════════════════════
def t1():
    w = mk_window()
    cases = [
        ("DINO (SwinT)", "SAM2 (tiny)", "sam2_t"),
        ("DINO (SwinT)", "SAM3",        "sam3"),
        ("DINO (SwinB)", "SAM2 (tiny)", "sam2_t"),
        ("YOLOE-vis",    "SAM2 (tiny)", "sam2_t"),
        ("YOLOE-vis",    "SAM3",        "sam3"),
        ("YOLOE-vis",    "(none)",      None),
        ("YOLOE-seg (one-shot)", "(none)", None),
        ("YOLOE-seg (one-shot)", "SAM3",   "sam3"),
        ("SAM3 (one-shot)",      "(none)", "sam3"),
        ("SAM3 (one-shot)",      "SAM2 (tiny)", "sam2_t"),
    ]
    for det, seg, expect in cases:
        w.detector_choice = det; w.segmenter_choice = seg
        got = w._active_interactive_sam_key()
        check(f"T1 sam-key {det} + {seg} -> {expect}", got == expect, f"got {got}")
t1()

# ══════════════════════════════════════════════════════════════════════════
# T2: _refresh_mask_draw_enabled gating
# ══════════════════════════════════════════════════════════════════════════
def mk_gating_window(mode, det, seg, images, semiauto_masks=False):
    w = mk_window()
    w.current_mode = mode
    w.detector_choice = det; w.segmenter_choice = seg
    w.images = images
    w.image_label = StubLabel(semiauto_masks=semiauto_masks)
    w._draw_tool = "box"
    w.draw_btn = mk_draw_button()
    w.draw_tool_box_action = mk_action(True)
    w.mask_draw_action = mk_action()
    w.semiauto_edit_action = mk_action()
    return w

def t2():
    # Model-only gating: enabled whenever an interactive SAM model is active,
    # in EITHER view. Greyed only for YOLOE standalone / segmenter (none).
    for mode in ("seg", "bbox"):
        w = mk_gating_window(mode, "DINO (SwinT)", "SAM2 (tiny)", ["a.jpg"])
        w._refresh_mask_draw_enabled()
        check(f"T2 {mode}+SAM2+img -> enabled", w.mask_draw_action.isEnabled())

    w = mk_gating_window("bbox", "SAM3 (one-shot)", "(none)", ["a.jpg"])
    w._refresh_mask_draw_enabled()
    check("T2 SAM3 one-shot (seg none) -> enabled", w.mask_draw_action.isEnabled())

    w = mk_gating_window("bbox", "YOLOE-vis", "SAM2 (tiny)", ["a.jpg"])
    w._refresh_mask_draw_enabled()
    check("T2 YOLOE-vis + SAM2 segmenter -> enabled", w.mask_draw_action.isEnabled())

    # Greyed cases: YOLOE standalone, or segmenter (none) with non-SAM detector.
    w = mk_gating_window("seg", "YOLOE-vis", "(none)", ["a.jpg"])
    w._refresh_mask_draw_enabled()
    check("T2 YOLOE-vis standalone -> disabled", not w.mask_draw_action.isEnabled())

    w = mk_gating_window("seg", "YOLOE-seg (one-shot)", "(none)", ["a.jpg"])
    w._refresh_mask_draw_enabled()
    check("T2 YOLOE-seg standalone -> disabled", not w.mask_draw_action.isEnabled())

    w = mk_gating_window("seg", "DINO (SwinT)", "SAM2 (tiny)", [])
    w._refresh_mask_draw_enabled()
    check("T2 no-image -> disabled", not w.mask_draw_action.isEnabled())

    # Edit Masks needs at least one editable mask present; it does NOT need a SAM
    # model. Vertex editing is model-free; the SAM "points re-run" sub-mode is
    # gated separately inside the edit dialog.
    w = mk_gating_window("seg", "DINO (SwinT)", "SAM2 (tiny)", ["a.jpg"], semiauto_masks=False)
    w._refresh_mask_draw_enabled()
    check("T2 edit action disabled with no editable masks", not w.semiauto_edit_action.isEnabled())
    w = mk_gating_window("seg", "DINO (SwinT)", "SAM2 (tiny)", ["a.jpg"], semiauto_masks=True)
    w._refresh_mask_draw_enabled()
    check("T2 edit action enabled with a mask present", w.semiauto_edit_action.isEnabled())
    w = mk_gating_window("seg", "YOLOE-vis", "(none)", ["a.jpg"], semiauto_masks=True)
    w._refresh_mask_draw_enabled()
    check("T2 edit action enabled with masks even without SAM (vertex editing)",
          w.semiauto_edit_action.isEnabled())

    # Active tool was semi-auto, then the model changes to a non-SAM one -> the
    # Draw button falls back to the box tool.
    w = mk_gating_window("seg", "DINO (SwinT)", "SAM2 (tiny)", ["a.jpg"])
    w._draw_tool = "semiauto"; w.draw_btn.setChecked(True)
    w.detector_choice = "YOLOE-vis"; w.segmenter_choice = "(none)"  # now invalid
    w._refresh_mask_draw_enabled()
    check("T2 semi-auto active + model goes non-SAM -> falls back to box",
          w._draw_tool == "box" and not w.draw_btn.isChecked(),
          f"tool={w._draw_tool} checked={w.draw_btn.isChecked()}")
t2()

# ══════════════════════════════════════════════════════════════════════════
# T3: AnnotationCanvas point capture + widget->image mapping
# ══════════════════════════════════════════════════════════════════════════
def mk_canvas():
    c = AnnotationCanvas()
    c.resize(100, 100)
    c._orig_w = 100; c._orig_h = 100
    c._zoom = 1.0; c._pan_x = 0.0; c._pan_y = 0.0
    return c

def press(c, x, y, button):
    ev = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonPress,
                           QtCore.QPointF(x, y), button, button,
                           QtCore.Qt.NoModifier)
    c.mousePressEvent(ev)

def move(c, x, y, button=QtCore.Qt.LeftButton):
    ev = QtGui.QMouseEvent(QtCore.QEvent.MouseMove,
                           QtCore.QPointF(x, y), QtCore.Qt.NoButton, button,
                           QtCore.Qt.NoModifier)
    c.mouseMoveEvent(ev)

def release(c, x, y, button=QtCore.Qt.LeftButton):
    ev = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonRelease,
                           QtCore.QPointF(x, y), button, button,
                           QtCore.Qt.NoModifier)
    c.mouseReleaseEvent(ev)

def t3():
    # Semi-auto: every left-click adds a connected FOREGROUND point; right-click
    # removes the nearest point. No background points anymore.
    c = mk_canvas()
    c.set_mask_draw_mode(True, kind="semiauto")
    fired = {"n": 0}
    c.mask_point_added.connect(lambda: fired.__setitem__("n", fired["n"] + 1))

    press(c, 30, 40, QtCore.Qt.LeftButton)
    press(c, 70, 60, QtCore.Qt.LeftButton)
    pts = c.get_mask_points_image_coords()
    check("T3 two foreground points captured", len(pts) == 2, f"pts={pts}")
    check("T3 all foreground (label 1)", all(lab == 1 for _p, lab in pts), f"{pts}")
    check("T3 coords correct", pts[0] == ((30.0, 40.0), 1) and pts[1] == ((70.0, 60.0), 1))
    check("T3 semi-auto clicks do NOT run SAM (no live preview)", fired["n"] == 0,
          f"n={fired['n']}")

    # right-click ON a point removes it
    press(c, 70, 60, QtCore.Qt.RightButton)
    check("T3 right-click removes nearest point",
          c.get_mask_points_image_coords() == [((30.0, 40.0), 1)])

    # out-of-image click is ignored
    before = len(c._mask_points)
    press(c, 150, 150, QtCore.Qt.LeftButton)
    check("T3 out-of-image click ignored", len(c._mask_points) == before)

    c.clear_mask_session()
    check("T3 clear empties points + preview",
          c._mask_points == [] and c.get_mask_preview() is None)

    c.set_mask_draw_mode(False)
    press(c, 20, 20, QtCore.Qt.LeftButton)
    check("T3 no capture when mode off", c._mask_points == [])

    # Auto draw: each click ADDS a refine point (outside the mask = positive,
    # inside = negative, Roboflow-style) AND runs SAM live. Points accumulate.
    c.set_mask_draw_mode(True, kind="autodraw")
    afired = {"n": 0}
    c.mask_point_added.connect(lambda: afired.__setitem__("n", afired["n"] + 1))
    press(c, 30, 30, QtCore.Qt.LeftButton)
    press(c, 60, 60, QtCore.Qt.LeftButton)
    check("T3 autodraw accumulates refine points",
          c.get_mask_points_image_coords() == [((30.0, 30.0), 1), ((60.0, 60.0), 1)],
          c.get_mask_points_image_coords())
    check("T3 autodraw runs SAM live per click", afired["n"] == 2, f"n={afired['n']}")
t3()

# ══════════════════════════════════════════════════════════════════════════
# T4: key handling: commit / cancel / backspace
# ══════════════════════════════════════════════════════════════════════════
def keypress(c, key):
    ev = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, key, QtCore.Qt.NoModifier)
    c.keyPressEvent(ev)

def t4():
    # Auto-draw key handling: Enter commits the live preview, Backspace re-runs,
    # Esc cancels.
    c = mk_canvas()
    c.set_mask_draw_mode(True, kind="autodraw")
    committed = {"n": 0}; added = {"n": 0}
    c.mask_commit_requested.connect(lambda: committed.__setitem__("n", committed["n"] + 1))
    c.mask_point_added.connect(lambda: added.__setitem__("n", added["n"] + 1))

    press(c, 30, 30, QtCore.Qt.LeftButton)
    c.set_mask_preview([[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]])

    keypress(c, QtCore.Qt.Key_Return)
    check("T4 autodraw Enter requests commit", committed["n"] == 1, f"n={committed['n']}")

    n_before = len(c._mask_points); added_before = added["n"]
    keypress(c, QtCore.Qt.Key_Backspace)
    check("T4 Backspace pops a point", len(c._mask_points) == n_before - 1)
    check("T4 autodraw Backspace re-runs preview", added["n"] == added_before + 1)

    press(c, 50, 50, QtCore.Qt.LeftButton)
    keypress(c, QtCore.Qt.Key_Escape)
    check("T4 Esc cancels session", c._mask_points == [] and c.get_mask_preview() is None)
t4()

# ══════════════════════════════════════════════════════════════════════════
# T21: semi-auto close-to-segment (Google-Draw curve style)
# ══════════════════════════════════════════════════════════════════════════
def t21():
    c = mk_canvas()
    c.set_mask_draw_mode(True, kind="semiauto")
    closed = {"n": 0}
    c.mask_close_requested.connect(lambda: closed.__setitem__("n", closed["n"] + 1))

    # Place 3 points; no close yet.
    press(c, 20, 20, QtCore.Qt.LeftButton)
    press(c, 80, 20, QtCore.Qt.LeftButton)
    press(c, 50, 80, QtCore.Qt.LeftButton)
    check("T21 three points placed", len(c._mask_points) == 3)
    check("T21 not closed before the close gesture", closed["n"] == 0)

    # Clicking the FIRST point (20,20) closes the outline.
    press(c, 21, 21, QtCore.Qt.LeftButton)
    check("T21 clicking the first point closes the outline", closed["n"] == 1, f"n={closed['n']}")
    check("T21 the closing click did NOT add a 4th point", len(c._mask_points) == 3)

    # Enter also closes (>=3 points).
    keypress(c, QtCore.Qt.Key_Return)
    check("T21 Enter closes too", closed["n"] == 2)

    # With < 3 points, neither Enter nor a first-point click closes.
    c.clear_mask_session()
    press(c, 30, 30, QtCore.Qt.LeftButton)
    press(c, 40, 40, QtCore.Qt.LeftButton)
    keypress(c, QtCore.Qt.Key_Return)
    check("T21 Enter with <3 points does not close", closed["n"] == 2)

    # _close_mask_object runs SAM (one object; nesting covered by T9) and commits.
    w = mk_window(); w.image_label = c
    w.detector_choice = "DINO (SwinT)"; w.segmenter_choice = "SAM2 (tiny)"
    w.images = ["img.jpg"]; w.current_image_index = 0
    w.base_cv2_image = np.zeros((100, 100, 3), dtype=np.uint8)
    w.live_boxes = []; w.live_box_sources = []
    cap = {}
    # SAM returns a result with no usable mask, so _close_mask_object exercises
    # its documented "fall back to the drawn outline" path (contour extraction
    # itself is covered by T9). masks=None routes through _sam_points_call's
    # existing None-guard instead of crashing in _mask_to_polys.
    class FR: masks = None
    def fake(source, **kw): cap["kw"] = kw; cap["labels"] = kw.get("labels"); return [FR()]
    w._get_model = lambda key: fake
    G["result_clean_polys"] = lambda r: [[[0.2, 0.2], [0.8, 0.2], [0.5, 0.8]]]
    persisted = {"n": 0}
    w._rebake_overlay = lambda: None
    w._persist_annotations = lambda silent=False: persisted.__setitem__("n", persisted["n"] + 1)
    c._mask_points = [[20, 20, 1], [80, 20, 1], [50, 80, 1]]
    w._close_mask_object()
    check("T21 close runs SAM as ONE foreground object", cap.get("labels") == [[1, 1, 1]], cap)
    semis = [a for a in c.annotations if a.get("semiauto")]
    check("T21 close commits a sticky semi-auto mask", len(semis) == 1 and persisted["n"] == 1)
    check("T21 session cleared after close", c._mask_points == [] and c.get_mask_preview() is None)
    G["result_clean_polys"] = lambda r: []
t21()

# ══════════════════════════════════════════════════════════════════════════
# T5: _commit_mask_object: append poly, align live_boxes, persist
# ══════════════════════════════════════════════════════════════════════════
def t5():
    w = mk_window()
    c = mk_canvas()
    w.image_label = c
    # Pre-existing seg state: 2 detector polys + 1 trailing prompt rect.
    c.annotations = [
        {"type": "poly", "data": [[0.0, 0.0], [0.1, 0.0], [0.1, 0.1]], "deleted": False, "source": "detector"},
        {"type": "poly", "data": [[0.2, 0.2], [0.3, 0.2], [0.3, 0.3]], "deleted": False, "source": "detector"},
        {"type": "rect", "data": [0.5, 0.5, 0.1, 0.1], "deleted": False, "source": "prompt"},
    ]
    w.live_boxes = [[0, 0, 10, 10], [20, 20, 30, 30]]      # aligned with the 2 polys
    w.live_box_sources = ["detector", "detector"]
    rebaked = {"n": 0}; persisted = {"n": 0}
    w._rebake_overlay = lambda: rebaked.__setitem__("n", rebaked["n"] + 1)
    w._persist_annotations = lambda silent=False: persisted.__setitem__("n", persisted["n"] + 1)

    # Two foreground points so the commit also stores normalized sam_points.
    c._mask_points = [[50, 50, 1], [60, 60, 1]]
    # A committed mask preview spanning x in [0.4,0.6], y in [0.4,0.7].
    c.set_mask_preview([[0.4, 0.4], [0.6, 0.4], [0.6, 0.7], [0.4, 0.7]])
    w._commit_mask_object()

    new = c.annotations[-1]  # appended at the end (sticky masks decoupled from live_boxes)
    check("T5 new semiauto poly appended",
          new["type"] == "poly" and new["source"] == "manual" and new.get("semiauto") is True,
          f"{new}")
    check("T5 normalized sam_points stored (all foreground)",
          new.get("sam_points") == [[0.5, 0.5, 1], [0.6, 0.6, 1]], f"{new.get('sam_points')}")
    check("T5 live_boxes UNCHANGED (mask decoupled)",
          w.live_boxes == [[0, 0, 10, 10], [20, 20, 30, 30]] and w.live_box_sources == ["detector", "detector"])
    check("T5 preview cleared after commit", c.get_mask_preview() is None)
    check("T5 rebake + persist called", rebaked["n"] == 1 and persisted["n"] == 1)

    # commit with no preview is a no-op
    before = len(c.annotations)
    w._commit_mask_object()
    check("T5 no-op when no preview", len(c.annotations) == before)
t5()

def mk_split_button_window(mode="seg"):
    """A window wired up like the real Draw-Boxes split button."""
    w = mk_window()
    w.current_mode = mode
    w.image_label = StubLabel()
    w._draw_tool = "box"
    w.draw_btn = mk_draw_button()
    w.edit_btn = mk_draw_button()
    w.resize_btn = mk_draw_button()
    w.multi_select_btn = QtWidgets.QPushButton(); w.multi_select_btn.setCheckable(True)
    w.mask_checkbox = QtWidgets.QCheckBox("Segmentation")
    w.draw_tool_box_action = mk_action(True)
    w.mask_draw_action = mk_action(False)
    w.draw_btn.toggled.connect(w._toggle_draw_btn)
    w.resize_btn.toggled.connect(w._toggle_resize_mode)
    return w

# ══════════════════════════════════════════════════════════════════════════
# T6: split button: tool select (arm), toggle on/off, mutual exclusion
# ══════════════════════════════════════════════════════════════════════════
def t6():
    w = mk_split_button_window("seg")

    # Default tool = box. Pressing the button toggles BOX drawing.
    w.draw_btn.setChecked(True)
    check("T6 box tool: press -> box-draw on",
          any(c == ("set_draw_mode", (True,)) for c in w.image_label.calls))
    check("T6 box tool label", w.draw_btn.text() == "Draw Boxes: ON", w.draw_btn.text())
    w.draw_btn.setChecked(False)

    # Pick Semi-Auto from the dropdown -> ARMS it: button switches mode but stays OFF.
    w._select_draw_tool("semiauto")
    check("T6 select semi-auto -> tool=semiauto", w._draw_tool == "semiauto")
    check("T6 select semi-auto -> button still OFF (armed)", not w.draw_btn.isChecked())
    check("T6 armed label is Semi-Auto OFF", w.draw_btn.text() == "Manual Masks: OFF", w.draw_btn.text())

    # Press the button -> semi-auto activates (orange/checked) + canvas mask mode.
    w.image_label.calls.clear()
    w.draw_btn.setChecked(True)
    check("T6 press -> semi-auto canvas mode on",
          any(c == ("set_mask_draw_mode", (True,)) for c in w.image_label.calls))
    check("T6 semi-auto ON label", w.draw_btn.text() == "Manual Masks: ON", w.draw_btn.text())

    # BUG FIX: off -> on must STAY semi-auto (not revert to Draw Boxes).
    w.draw_btn.setChecked(False)
    check("T6 semi-auto OFF stays semi-auto tool", w._draw_tool == "semiauto")
    check("T6 semi-auto OFF label", w.draw_btn.text() == "Manual Masks: OFF", w.draw_btn.text())
    w.draw_btn.setChecked(True)
    check("T6 semi-auto back ON (no revert to box)",
          w._draw_tool == "semiauto" and w.draw_btn.text() == "Manual Masks: ON")

    # Image Resize PARKS the active draw tool (preserved, canvas input gated)
    # rather than switching it off, so the draw button stays on and resumes
    # when resize turns back off.
    w.resize_btn.setChecked(True)
    check("T6 resize ON -> draw button stays on (parked)", w.draw_btn.isChecked())

    # Switch back to Draw Boxes via the dropdown.
    w._select_draw_tool("box")
    check("T6 switch back to box tool", w._draw_tool == "box"
          and w.draw_btn.text() == "Draw Boxes: OFF")
t6()

# ══════════════════════════════════════════════════════════════════════════
# T7: activating semi-auto from a non-seg view auto-switches to Segmentation
# ══════════════════════════════════════════════════════════════════════════
def t7():
    w = mk_split_button_window("bbox")
    w._select_draw_tool("semiauto")
    w.draw_btn.setChecked(True)  # activate -> should flip to seg view
    check("T7 activating from bbox view checks Segmentation", w.mask_checkbox.isChecked())

    w2 = mk_split_button_window("seg")
    w2.mask_checkbox.setChecked(False)
    w2._select_draw_tool("semiauto")
    w2.draw_btn.setChecked(True)
    check("T7 already in seg view -> no spurious checkbox flip", not w2.mask_checkbox.isChecked())
t7()

# ══════════════════════════════════════════════════════════════════════════
# T8: point->image coordinate mapping is correct under zoom/pan
# ══════════════════════════════════════════════════════════════════════════
def t8():
    c = mk_canvas()           # 100x100 widget, 100x100 image
    c._zoom = 2.0             # scale=2, off=(-50,-50): widget(10,10)->image(30,30)
    c.set_mask_draw_mode(True)
    press(c, 10, 10, QtCore.Qt.LeftButton)
    pts = c.get_mask_points_image_coords()
    check("T8 zoomed click maps to correct image coords",
          len(pts) == 1 and pts[0] == ((30.0, 30.0), 1), f"pts={pts}")

    # pan shifts the mapping by the pan amount / scale
    c.clear_mask_session()
    c._pan_x = 20.0           # off_x = -50 + 20 = -30 ; widget(10,10)-> (10+30)/2 = 20
    press(c, 10, 10, QtCore.Qt.LeftButton)
    pts = c.get_mask_points_image_coords()
    check("T8 panned click maps with pan offset",
          len(pts) == 1 and pts[0] == ((20.0, 30.0), 1), f"pts={pts}")
t8()

# ══════════════════════════════════════════════════════════════════════════
# T9: _on_mask_point_added builds the SAM call correctly (fake model)
# ══════════════════════════════════════════════════════════════════════════
def t9():
    w = mk_window()
    _orig_m2p = G["_mask_to_polys"]   # restore after the overrides below
    w.detector_choice = "DINO (SwinT)"; w.segmenter_choice = "SAM2 (tiny)"
    w.images = ["img.jpg"]; w.current_image_index = 0
    w.base_cv2_image = np.zeros((100, 100, 3), dtype=np.uint8)
    c = mk_canvas(); w.image_label = c

    cap = {}
    class FR:
        masks = object()
    def fake(source, **kw):
        cap["source"] = source; cap["kw"] = kw
        return [FR()]
    w._get_model = lambda key: fake

    # --- nesting invariant (the core bug fix) via _sam_points_call directly ---
    G["_mask_to_polys"] = lambda r: [[[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]]]
    w._sam_points_call("sam2_t", "src", [(30, 40), (70, 60)])
    check("T9 multi-point nested as ONE object (1,N,2)",
          cap["kw"]["points"] == [[[30.0, 40.0], [70.0, 60.0]]], cap["kw"])
    check("T9 labels nested (1,N) all foreground", cap["kw"]["labels"] == [[1, 1]])
    check("T9 quiet, no-save flags", cap["kw"]["verbose"] is False and cap["kw"]["save"] is False)
    w._sam_points_call("sam2_t", "src", [(50, 50)])
    check("T9 single point nested as (1,1,2)",
          cap["kw"]["points"] == [[[50.0, 50.0]]] and cap["kw"]["labels"] == [[1]], cap["kw"])

    # --- ROI crop: the model runs on a WINDOW around the points, not the whole
    #     image; points are crop-local; the polygon comes back image-normalized. ---
    cap.clear()
    G["_mask_to_polys"] = lambda r: [[[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]]
    c._mask_points = [[30, 40, 1], [70, 60, 1]]
    w._on_mask_point_added()
    src = cap["source"]
    check("T9 ROI: model receives a CROP (smaller than the 100x100 image)",
          hasattr(src, "shape") and src.shape[0] < 100 and src.shape[1] < 100,
          getattr(src, "shape", src))
    pp = cap["kw"]["points"][0]
    check("T9 ROI: points are crop-local (inside the crop, not raw image coords)",
          all(0 <= px < src.shape[1] and 0 <= py < src.shape[0] for px, py in pp), pp)
    prev = c.get_mask_preview()
    check("T9 ROI: preview comes back in image-normalized coords",
          prev is not None and all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in prev), prev)

    # --- grow-and-retry: an edge-touching mask triggers a bigger crop ---
    cap.clear()
    sizes = []
    def fake_grow(source, **kw):
        sizes.append(source.shape[:2]); return [FR()]
    w._get_model = lambda key: fake_grow
    seq = iter([[[0.0, 0.5], [0.4, 0.5], [0.2, 0.9]],     # touches LEFT crop edge -> grow
                [[0.3, 0.3], [0.6, 0.3], [0.45, 0.6]]])    # clean -> accept
    G["_mask_to_polys"] = lambda r: [next(seq)]       # list-of-polygons
    w._segment_region("img.jpg", [(40, 50), (50, 50)], max_grows=2)
    check("T9 grow-and-retry enlarged the crop on edge-touch",
          len(sizes) == 2 and (sizes[1][0] * sizes[1][1]) > (sizes[0][0] * sizes[0][1]), sizes)
    G["_mask_to_polys"] = _orig_m2p          # restore the real extractor
    G["result_clean_polys"] = lambda r: []   # restore stub
t9()

# ══════════════════════════════════════════════════════════════════════════
# T10: Edit Semi-Auto Segments: select a mask, edit points, apply
# ══════════════════════════════════════════════════════════════════════════
def t10():
    c = mk_canvas()
    # one committed semi-auto mask: square (0.2..0.8) with one fg point at centre
    c.annotations = [{
        "type": "poly",
        "data": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
        "deleted": False, "source": "manual", "semiauto": True,
        "sam_points": [[0.5, 0.5, 1]],
    }]
    check("T10 has_semiauto_masks true", c.has_semiauto_masks())

    c.set_semiauto_edit_mode(True)
    # click inside the mask -> selects it and loads its stored point
    press(c, 50, 50, QtCore.Qt.LeftButton)
    check("T10 click selects the mask", c.get_semiauto_selected_index() == 0)
    check("T10 stored point loaded to image coords",
          c._mask_points == [[50.0, 50.0, 1]], f"{c._mask_points}")

    # add a foreground point with a left-click on empty area inside the image
    press(c, 30, 30, QtCore.Qt.LeftButton)
    check("T10 left-click adds a foreground point",
          len(c._mask_points) == 2 and c._mask_points[1][2] == 1, f"{c._mask_points}")

    # right-click ON an existing point removes it
    press(c, 50, 50, QtCore.Qt.RightButton)
    check("T10 right-click on a point removes it", len(c._mask_points) == 1, f"{c._mask_points}")

    # apply: ManualWindow rewrites the annotation polygon + sam_points (sticky
    # masks are NOT in live_boxes, so live_boxes is left untouched)
    w = mk_window(); w.image_label = c
    w.live_boxes = []; w.live_box_sources = []
    rebaked = {"n": 0}; persisted = {"n": 0}
    w._rebake_overlay = lambda: rebaked.__setitem__("n", rebaked["n"] + 1)
    w._persist_annotations = lambda silent=False: persisted.__setitem__("n", persisted["n"] + 1)
    c._mask_points = [[40, 40, 1]]
    c.set_mask_preview([[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]])
    w._apply_semiauto_edit()
    check("T10 apply rewrites polygon",
          c.annotations[0]["data"] == [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]])
    check("T10 apply rewrites sam_points", c.annotations[0]["sam_points"] == [[0.4, 0.4, 1]],
          f"{c.annotations[0]['sam_points']}")
    check("T10 selection cleared after apply", c.get_semiauto_selected_index() is None)
    check("T10 apply persisted + rebaked", rebaked["n"] == 1 and persisted["n"] == 1)
t10()

# ══════════════════════════════════════════════════════════════════════════
# T11: polygon vertex editing: drag, add, delete, apply
# ══════════════════════════════════════════════════════════════════════════
def t11():
    c = mk_canvas()
    c.annotations = [{
        "type": "poly",
        "data": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
        "deleted": False, "source": "manual", "semiauto": True,
        "sam_points": [[0.5, 0.5, 1]],
    }]
    c.set_semiauto_edit_mode(True)
    press(c, 50, 50, QtCore.Qt.LeftButton)            # select (points target default)
    c.set_semiauto_edit_target("vertices")
    check("T11 target switched to vertices", c.get_semiauto_edit_target() == "vertices")

    # drag vertex 0 (0.2,0.2 -> widget 20,20) to widget (10,10) -> (0.1,0.1)
    press(c, 20, 20, QtCore.Qt.LeftButton)
    check("T11 vertex grabbed", c._vertex_drag_idx == 0)
    move(c, 10, 10)
    release(c, 10, 10)
    check("T11 vertex moved", c.annotations[0]["data"][0] == [0.1, 0.1],
          c.annotations[0]["data"][0])

    # add a vertex (left-click off any handle)
    n0 = len(c.annotations[0]["data"])
    press(c, 50, 18, QtCore.Qt.LeftButton)
    check("T11 vertex added", len(c.annotations[0]["data"]) == n0 + 1)

    # delete a vertex (right-click on the handle at 0.8,0.2 -> widget 80,20)
    n1 = len(c.annotations[0]["data"])
    press(c, 80, 20, QtCore.Qt.RightButton)
    check("T11 vertex removed", len(c.annotations[0]["data"]) == n1 - 1)

    # never drop below a triangle
    while len(c.annotations[0]["data"]) > 3:
        v = c._to_label(c.annotations[0]["data"])[0]
        press(c, int(v[0]), int(v[1]), QtCore.Qt.RightButton)
    n2 = len(c.annotations[0]["data"])
    v = c._to_label(c.annotations[0]["data"])[0]
    press(c, int(v[0]), int(v[1]), QtCore.Qt.RightButton)
    check("T11 won't delete below 3 vertices", len(c.annotations[0]["data"]) == n2 == 3)

    # apply (vertices target uses ann['data'] directly)
    w = mk_window(); w.image_label = c
    w.live_boxes = [[20, 20, 80, 80]]; w.live_box_sources = ["manual"]
    persisted = {"n": 0}
    w._rebake_overlay = lambda: None
    w._persist_annotations = lambda silent=False: persisted.__setitem__("n", persisted["n"] + 1)
    applied_data = [list(p) for p in c.annotations[0]["data"]]
    w._apply_semiauto_edit()
    check("T11 apply keeps the edited vertices", c.annotations[0]["data"] == applied_data)
    check("T11 apply deselects + persists",
          c.get_semiauto_selected_index() is None and persisted["n"] == 1)

    # Esc reverts in-progress vertex edits
    c.set_semiauto_edit_mode(True)
    press(c, 50, 50, QtCore.Qt.LeftButton)
    c.set_semiauto_edit_target("vertices")
    orig = [list(p) for p in c.annotations[0]["data"]]
    press(c, *[int(v) for v in c._to_label(c.annotations[0]["data"])[0]], button=QtCore.Qt.LeftButton)
    move(c, 5, 5); release(c, 5, 5)
    keypress(c, QtCore.Qt.Key_Escape)
    check("T11 Esc reverts vertex edits", c.annotations[0]["data"] == orig)
t11()

# ══════════════════════════════════════════════════════════════════════════
# T12: polygon simplification (Ramer-Douglas-Peucker, pure python)
# ══════════════════════════════════════════════════════════════════════════
def t12():
    # square with a near-collinear extra point on the top edge
    poly = [[0.0, 0.0], [0.5, 0.001], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    out = ManualWindow._simplify_poly(poly, 0.01)
    check("T12 drops near-collinear vertex", len(out) < len(poly) and len(out) >= 3,
          f"{out}")
    same = ManualWindow._simplify_poly(poly, 0.0)
    check("T12 eps=0 is a no-op", same == [list(p) for p in poly])
    tri = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
    check("T12 triangle unchanged", ManualWindow._simplify_poly(tri, 0.5) == tri)
t12()

# ══════════════════════════════════════════════════════════════════════════
# T13: class id flows through to the saved label files
# ══════════════════════════════════════════════════════════════════════════
def t13():
    import tempfile, os as _os
    d = tempfile.mkdtemp()
    w = mk_window(); w.output_folder = d
    anns = [{"type": "poly",
             "data": [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]],
             "deleted": False, "source": "manual", "semiauto": True, "cls": 3}]
    w._write_label_files(anns, "img.jpg")
    box_line = open(_os.path.join(d, "boxes", "img.txt")).read().split()[0]
    seg_line = open(_os.path.join(d, "segments", "img.txt")).read().split()[0]
    check("T13 box label uses class id", box_line == "3", box_line)
    check("T13 segment label uses class id", seg_line == "3", seg_line)
    # default class id stays 0 when unset (back-compat)
    anns2 = [{"type": "rect", "data": [0.5, 0.5, 0.2, 0.2], "deleted": False, "source": "manual"}]
    w._write_label_files(anns2, "img2.jpg")
    check("T13 default class id is 0",
          open(_os.path.join(d, "boxes", "img2.txt")).read().split()[0] == "0")
t13()

# ══════════════════════════════════════════════════════════════════════════
# T14: settings dialog getters
# ══════════════════════════════════════════════════════════════════════════
def t14():
    dlg = SemiAutoSettingsDialog(None, target="vertices", cls=5)
    check("T14 dialog reports target", dlg.target() == "vertices")
    check("T14 dialog reports class id", dlg.cls() == 5)
    check("T14 dialog default simplify eps 0", dlg.simplify_eps() == 0.0)
    dlg.simplify_spin.setValue(2.0)
    check("T14 simplify percent -> normalized eps", abs(dlg.simplify_eps() - 0.02) < 1e-9)
    dlg2 = SemiAutoSettingsDialog(None, target="points", cls=0)
    check("T14 points target round-trips", dlg2.target() == "points")
t14()

# ══════════════════════════════════════════════════════════════════════════
# T15: SAM availability prediction + "in use" detection
# ══════════════════════════════════════════════════════════════════════════
def t15():
    w = mk_window(); c = mk_canvas(); w.image_label = c
    w.detector_choice = "DINO (SwinT)"; w.segmenter_choice = "SAM2 (tiny)"
    check("T15 predict SAM for DINO+SAM2", w._sam_available_for("DINO (SwinT)", "SAM2 (tiny)"))
    check("T15 predict no-SAM for YOLOE-vis+(none)",
          not w._sam_available_for("YOLOE-vis", "(none)"))
    check("T15 predict SAM for SAM3 one-shot", w._sam_available_for("SAM3 (one-shot)", "(none)"))
    # committed choice unchanged by prediction
    check("T15 prediction leaves committed choice intact",
          (w.detector_choice, w.segmenter_choice) == ("DINO (SwinT)", "SAM2 (tiny)"))

    w._draw_tool = "box"
    check("T15 not in use by default", not w._semiauto_in_use())
    w._draw_tool = "semiauto"
    check("T15 in use when tool=semiauto", w._semiauto_in_use())
    w._draw_tool = "box"; c.set_semiauto_edit_mode(True)
    check("T15 in use when edit mode on", w._semiauto_in_use())
    c.set_semiauto_edit_mode(False)
t15()

# ══════════════════════════════════════════════════════════════════════════
# T16: model-change guard: Revert restores combo, Switch tears semi-auto down
# ══════════════════════════════════════════════════════════════════════════
def mk_guard_window(det, seg):
    w = mk_window(); c = mk_canvas(); w.image_label = c
    w.detector_choice = det; w.segmenter_choice = seg
    w._draw_tool = "box"
    w.draw_btn = mk_draw_button()
    w.draw_tool_box_action = mk_action(True)
    w.autodraw_action = mk_action(False)
    w.mask_draw_action = mk_action(False)
    w.semiauto_edit_action = mk_action(False)
    w.detector_combo = QtWidgets.QComboBox(); w.detector_combo.addItems(ManualWindow.DETECTORS)
    w.segmenter_combo = QtWidgets.QComboBox(); w.segmenter_combo.addItems(ManualWindow.SEGMENTERS)
    w.pipeline_combo = QtWidgets.QComboBox(); w.pipeline_combo.addItems(ManualWindow.PIPELINE_PRESETS)
    return w, c

def t16():
    # Revert: keep the SAM segmenter; combo + choice restored, semi-auto intact.
    w, c = mk_guard_window("YOLOE-vis", "SAM2 (tiny)")
    w._draw_tool = "semiauto"; w.mask_draw_action.setChecked(True); w.draw_tool_box_action.setChecked(False)
    w.segmenter_combo.blockSignals(True); w.segmenter_combo.setCurrentText("(none)"); w.segmenter_combo.blockSignals(False)
    w._ask_sam_loss = lambda unfinished: "revert"
    w._on_segmenter_changed("(none)")
    check("T16 revert keeps segmenter choice", w.segmenter_choice == "SAM2 (tiny)")
    check("T16 revert restores the combo text", w.segmenter_combo.currentText() == "SAM2 (tiny)")
    check("T16 revert leaves semi-auto active", w._draw_tool == "semiauto")

    # Switch: allow the change; semi-auto torn down (falls back to box tool).
    w, c = mk_guard_window("YOLOE-vis", "SAM2 (tiny)")
    w._draw_tool = "semiauto"; w.mask_draw_action.setChecked(True); w.draw_tool_box_action.setChecked(False)
    w._ask_sam_loss = lambda unfinished: "switch"
    w._on_segmenter_changed("(none)")
    check("T16 switch applies the change", w.segmenter_choice == "(none)")
    check("T16 switch deactivates semi-auto -> box tool", w._draw_tool == "box")

    # No prompt when the feature is not in use (must not even call _ask_sam_loss).
    w, c = mk_guard_window("YOLOE-vis", "SAM2 (tiny)")
    w._draw_tool = "box"
    def _boom(unfinished): raise AssertionError("should not prompt")
    w._ask_sam_loss = _boom
    w._on_segmenter_changed("(none)")
    check("T16 no prompt when feature unused", w.segmenter_choice == "(none)")
t16()

# ══════════════════════════════════════════════════════════════════════════
# T17: unfinished-mask guard on leaving the view / next image
# ══════════════════════════════════════════════════════════════════════════
def t17():
    w = mk_window(); c = mk_canvas(); w.image_label = c
    c.set_mask_draw_mode(True); c._mask_points = [[10, 10, 1]]
    check("T17 has_unfinished true with uncommitted points", c.has_unfinished_semiauto())

    # Cancel -> stay, points kept
    w._ask_discard_unfinished = lambda ctx: False
    check("T17 cancel returns False", not w._confirm_leave_unfinished("x"))
    check("T17 cancel keeps points", c._mask_points == [[10, 10, 1]])

    # Discard -> proceed, points cleared
    w._ask_discard_unfinished = lambda ctx: True
    check("T17 discard returns True", w._confirm_leave_unfinished("x"))
    check("T17 discard clears points", c._mask_points == [])

    # No prompt when nothing is unfinished
    c.set_mask_draw_mode(False)
    w._ask_discard_unfinished = lambda ctx: (_ for _ in ()).throw(AssertionError("no prompt"))
    check("T17 no prompt when finished", w._confirm_leave_unfinished("x"))
t17()

# ══════════════════════════════════════════════════════════════════════════
# T18: stale selection self-heals instead of crashing
# ══════════════════════════════════════════════════════════════════════════
def t18():
    c = mk_canvas()
    c.annotations = [{"type": "poly", "data": [[0.2, 0.2], [0.8, 0.2], [0.5, 0.8]],
                      "deleted": False, "source": "manual", "semiauto": True, "sam_points": []}]
    c.set_semiauto_edit_mode(True)
    press(c, 50, 40, QtCore.Qt.LeftButton)  # select
    check("T18 selected", c.get_semiauto_selected_index() == 0)
    # annotations replaced underneath (e.g. a regenerate)
    c.set_annotations(polys=[[[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]]])
    check("T18 selection invalidated on replace", c.get_semiauto_selected_index() is None)
    # a stale index must not crash the hit tests / accessor
    c._semiauto_sel_idx = 99
    check("T18 guarded accessor heals stale idx", c._semiauto_selected_ann() is None)
    check("T18 stale idx cleared", c.get_semiauto_selected_index() is None)
    check("T18 hit_vertex safe on stale idx", c._hit_vertex(QtCore.QPointF(50, 40)) is None)
t18()

# ══════════════════════════════════════════════════════════════════════════
# T19: sticky masks survive set_annotations (the persistence fix)
# ══════════════════════════════════════════════════════════════════════════
def t19():
    c = mk_canvas()
    sticky = {"type": "poly", "data": [[0.3, 0.3], [0.6, 0.3], [0.6, 0.6], [0.3, 0.6]],
              "deleted": False, "source": "manual", "semiauto": True,
              "sam_points": [[0.45, 0.45, 1]], "cls": 2}
    c.annotations = [sticky]
    # A segmenter re-run replaces annotations with fresh DETECTOR polys.
    c.set_annotations(polys=[[[0.0, 0.0], [0.1, 0.0], [0.1, 0.1]]], poly_sources=["detector"])
    semis = [a for a in c.annotations if a.get("semiauto")]
    check("T19 sticky mask survived the re-run", len(semis) == 1, f"{c.annotations}")
    check("T19 metadata intact (data/sam_points/cls)",
          semis[0]["data"] == sticky["data"] and semis[0]["sam_points"] == [[0.45, 0.45, 1]]
          and semis[0]["cls"] == 2)
    check("T19 detector poly present too", any(a.get("source") == "detector" for a in c.annotations))
    # Manual drawn-BOX polys (no semiauto flag) are NOT carried, so they regenerate.
    c.annotations = [{"type": "poly", "data": [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]],
                      "deleted": False, "source": "manual"}]
    c.set_annotations(polys=[[[0.5, 0.5], [0.6, 0.5], [0.6, 0.6]]], poly_sources=["detector"])
    check("T19 non-semiauto manual poly not carried",
          all(not (a.get("source") == "manual" and not a.get("semiauto")) for a in c.annotations))
t19()

# ══════════════════════════════════════════════════════════════════════════
# T20: Draw button label + hover tooltip per active tool
# ══════════════════════════════════════════════════════════════════════════
def t20():
    w = mk_window()
    w._draw_tool = "box"
    w.draw_btn = mk_draw_button()
    w.draw_tool_box_action = mk_action(True)
    w.autodraw_action = mk_action(False)
    w.mask_draw_action = mk_action(False)
    for tool, label, tip in (
        ("box", "Draw Boxes", G["TOOLTIP_BOX"]),
        ("autodraw", "Semi-Auto Points", G["TOOLTIP_AUTODRAW"]),
        ("semiauto", "Manual Masks", G["TOOLTIP_SEMIAUTO"]),
    ):
        w._select_draw_tool(tool)
        check(f"T20 {tool} label", w.draw_btn.text() == f"{label}: OFF", w.draw_btn.text())
        check(f"T20 {tool} tooltip is the hover instructions", w.draw_btn.toolTip() == tip)
    # the three tools are mutually exclusive in the menu
    w._select_draw_tool("autodraw")
    check("T20 exclusive: autodraw checked, others not",
          w.autodraw_action.isChecked() and not w.mask_draw_action.isChecked()
          and not w.draw_tool_box_action.isChecked())
t20()

# ══════════════════════════════════════════════════════════════════════════
# T22: Regenerate must not stack model output over a hand-drawn mask
# ══════════════════════════════════════════════════════════════════════════
def t22():
    c = mk_canvas()
    # a committed semi-auto mask covering roughly x,y in [0.3,0.6]
    c.annotations = [{
        "type": "poly", "data": [[0.3, 0.3], [0.6, 0.3], [0.6, 0.6], [0.3, 0.6]],
        "deleted": False, "source": "manual", "semiauto": True, "sam_points": [[0.45, 0.45, 1]],
    }]
    # A regenerate installs fresh DETECTOR polys: one overlapping the mask, one
    # elsewhere. The overlapping one must be dropped (manual wins); the other kept.
    overlapping = [[0.31, 0.31], [0.59, 0.31], [0.59, 0.59], [0.31, 0.59]]  # ~same region
    elsewhere   = [[0.05, 0.05], [0.15, 0.05], [0.15, 0.15], [0.05, 0.15]]
    c.set_annotations(polys=[overlapping, elsewhere], poly_sources=["detector", "detector"])

    semis = [a for a in c.annotations if a.get("semiauto")]
    det   = [a for a in c.annotations if a.get("source") == "detector"]
    check("T22 the hand-drawn mask is kept", len(semis) == 1)
    check("T22 overlapping detector poly dropped (no stacking)", len(det) == 1, f"{det}")
    check("T22 non-overlapping detector poly kept",
          det and det[0]["data"] == elsewhere, f"{det}")

    # No sticky mask -> nothing dropped (back-compat).
    c.annotations = []
    c.set_annotations(polys=[overlapping, elsewhere], poly_sources=["detector", "detector"])
    check("T22 no dedup when there are no hand-drawn masks",
          len([a for a in c.annotations if a.get("source") == "detector"]) == 2)

    # Drawing a mask over existing detector output removes the overlap IMMEDIATELY
    # (on commit), not just on the next Regenerate.
    c2 = mk_canvas()
    c2.annotations = [
        {"type": "poly", "data": list(overlapping), "deleted": False, "source": "detector"},
        {"type": "poly", "data": list(elsewhere), "deleted": False, "source": "detector"},
    ]
    w = mk_window(); w.image_label = c2; w.live_boxes = []; w.live_box_sources = []
    w._rebake_overlay = lambda: None
    w._persist_annotations = lambda silent=False: None
    c2._mask_points = [[45, 45, 1]]
    c2.set_mask_preview([[0.3, 0.3], [0.6, 0.3], [0.6, 0.6], [0.3, 0.6]])
    w._commit_mask_object()
    live = [a for a in c2.annotations if not a.get("deleted")]
    check("T22 commit soft-deletes the covered detector poly",
          sum(1 for a in live if a.get("source") == "detector") == 1
          and sum(1 for a in live if a.get("semiauto")) == 1, f"{live}")
    check("T22 the far detector poly is untouched",
          any(a.get("source") == "detector" and a["data"] == elsewhere for a in live))
t22()

# ══════════════════════════════════════════════════════════════════════════
# T23: mask IoU preserves clustered neighbours (boxes overlap, masks don't)
# ══════════════════════════════════════════════════════════════════════════
def t23():
    c = mk_canvas()
    # Hand-drawn mask = upper-left triangle of the box [0.2,0.2,0.6,0.6].
    mask = {"type": "poly", "data": [[0.2, 0.2], [0.6, 0.2], [0.2, 0.6]],
            "deleted": False, "source": "manual", "semiauto": True}
    # A NEIGHBOUR = lower-right triangle of the SAME box: bounding boxes overlap
    # almost completely, but the two masks share only the diagonal (≈0 area).
    neighbour = [[0.6, 0.6], [0.6, 0.2], [0.2, 0.6]]
    # A true DUPLICATE = essentially the same triangle as the mask.
    duplicate = [[0.21, 0.21], [0.59, 0.21], [0.21, 0.59]]

    # Sanity: bounding-box IoU would (wrongly) flag the neighbour as a dup.
    bb_mask = c._ann_bbox_norm_xyxy(mask)
    bb_neigh = c._ann_bbox_norm_xyxy({"type": "poly", "data": neighbour})
    check("T23 bbox IoU WOULD have flagged the neighbour (>0.6)",
          c._bbox_iou(bb_mask, bb_neigh) > 0.6, c._bbox_iou(bb_mask, bb_neigh))

    # Mask IoU keeps the neighbour, drops the duplicate.
    check("T23 neighbour is NOT a duplicate (mask IoU ~0)",
          not c._is_duplicate_of({"type": "poly", "data": neighbour}, mask))
    check("T23 true duplicate IS caught (mask IoU high)",
          c._is_duplicate_of({"type": "poly", "data": duplicate}, mask))

    # End-to-end through set_annotations: neighbour kept, duplicate dropped.
    c.annotations = [dict(mask)]
    c.set_annotations(polys=[neighbour, duplicate], poly_sources=["detector", "detector"])
    det = [a for a in c.annotations if a.get("source") == "detector"]
    check("T23 set_annotations keeps the clustered neighbour, drops the dup",
          len(det) == 1 and det[0]["data"] == neighbour, f"{det}")
t23()

# ══════════════════════════════════════════════════════════════════════════
# T24: switching models/tools/states stays robust for ALL SAM tools
# ══════════════════════════════════════════════════════════════════════════
def mk_full_draw_window(mode="seg"):
    w = mk_window()
    w.current_mode = mode
    w.image_label = StubLabel()
    w._draw_tool = "box"
    w.draw_btn = mk_draw_button()
    w.edit_btn = mk_draw_button()
    w.resize_btn = mk_draw_button()
    w.multi_select_btn = QtWidgets.QPushButton(); w.multi_select_btn.setCheckable(True)
    w.mask_checkbox = QtWidgets.QCheckBox("Segmentation")
    w.draw_tool_box_action = mk_action(True)
    w.autodraw_action = mk_action(False)
    w.mask_draw_action = mk_action(False)
    w.semiauto_edit_action = mk_action(False)
    w.draw_btn.toggled.connect(w._toggle_draw_btn)
    w.resize_btn.toggled.connect(w._toggle_resize_mode)
    w.semiauto_edit_action.toggled.connect(w._set_edit_masks)  # renamed from _toggle_semiauto_edit_mode
    return w

def t24():
    # "in use" + unfinished detection covers autodraw AND semiauto.
    for kind in ("autodraw", "semiauto"):
        w = mk_window(); c = mk_canvas(); w.image_label = c
        w._draw_tool = kind
        check(f"T24 {kind} counts as in-use", w._semiauto_in_use())
        c.set_mask_draw_mode(True, kind=kind)
        c._mask_points = [[10, 10, 1]]
        check(f"T24 {kind} single uncommitted point -> unfinished", c.has_unfinished_semiauto())

    # Model-change guard fires for autodraw too: Revert keeps it, Switch drops it.
    w, c = mk_guard_window("YOLOE-vis", "SAM2 (tiny)")
    w._draw_tool = "autodraw"; w.autodraw_action.setChecked(True); w.draw_tool_box_action.setChecked(False)
    w.segmenter_combo.blockSignals(True); w.segmenter_combo.setCurrentText("(none)"); w.segmenter_combo.blockSignals(False)
    w._ask_sam_loss = lambda u: "revert"
    w._on_segmenter_changed("(none)")
    check("T24 autodraw + revert keeps choice + tool", w.segmenter_choice == "SAM2 (tiny)" and w._draw_tool == "autodraw")
    w, c = mk_guard_window("YOLOE-vis", "SAM2 (tiny)")
    w._draw_tool = "autodraw"; w.autodraw_action.setChecked(True); w.draw_tool_box_action.setChecked(False)
    w._ask_sam_loss = lambda u: "switch"
    w._on_segmenter_changed("(none)")
    check("T24 autodraw + switch deactivates -> box", w.segmenter_choice == "(none)" and w._draw_tool == "box")

    # Tool-switch storm: cycle every tool + edit without exceptions; final state sane.
    w = mk_full_draw_window("seg")
    seq = ["box", "autodraw", "semiauto", "box", "semiauto", "autodraw", "box"]
    for t in seq:
        w._select_draw_tool(t)
        w.draw_btn.setChecked(True)
        w.draw_btn.setChecked(False)
    w.semiauto_edit_action.setChecked(True)   # enter edit mode
    w.semiauto_edit_action.setChecked(False)
    w.resize_btn.setChecked(True)             # resize knocks draw off
    check("T24 tool-switch storm leaves a sane state",
          w._draw_tool == "box" and not w.draw_btn.isChecked())
    # entering a SAM tool turned the box tool off etc. No crash got here = pass
    check("T24 storm completed without exception", True)
t24()

# ══════════════════════════════════════════════════════════════════════════
# T25: alerts use the app's consistent style; close-failure alerts (not print)
# ══════════════════════════════════════════════════════════════════════════
def t25():
    w = mk_window()
    box = w._styled_message("hello", "Title")
    ss = box.styleSheet()
    check("T25 styled alert: white background", "background-color: white" in ss, ss)
    check("T25 styled alert: black 24px text",
          "color: black" in ss and "font-size: 24px" in ss, ss)

    # When SAM yields no polygon, _close_mask_object now FALLS BACK to the drawn
    # outline so closing ALWAYS yields a mask (the old "dead-end alert + keep
    # points" behavior was removed). The alert now only fires if the outline
    # itself can't form a >=3-point polygon. With a valid 3-point outline here,
    # the close commits via the fallback and clears the session -- no alert.
    c = mk_canvas(); w.image_label = c
    w.detector_choice = "DINO (SwinT)"; w.segmenter_choice = "SAM2 (tiny)"
    w.images = ["img.jpg"]; w.current_image_index = 0
    w.base_cv2_image = np.zeros((100, 100, 3), dtype=np.uint8)
    w.live_boxes = []; w.live_box_sources = []
    w._rebake_overlay = lambda: None
    w._persist_annotations = lambda silent=False: None
    c._mask_points = [[20, 20, 1], [80, 20, 1], [50, 80, 1]]
    class FR:
        masks = object()
    w._get_model = lambda key: (lambda path, **kw: [FR()])
    _orig_m2p = G["_mask_to_polys"]
    G["_mask_to_polys"] = lambda r: []          # SAM yields no polygon
    shown = {"n": 0}
    class FakeBox:
        def exec_(self): shown["n"] += 1
    w._styled_message = lambda *a, **k: FakeBox()
    w._close_mask_object()
    check("T25 SAM-failure falls back to the outline (no dead-end alert)", shown["n"] == 0)
    semis = [a for a in c.annotations if a.get("semiauto")]
    check("T25 close still commits a mask via the outline fallback", len(semis) == 1, len(semis))
    check("T25 session cleared after fallback commit", c._mask_points == [])
    G["_mask_to_polys"] = _orig_m2p
t25()

# ══════════════════════════════════════════════════════════════════════════
# T26: SpatialGrid: correctness + it only returns local candidates
# ══════════════════════════════════════════════════════════════════════════
def t26():
    g = SpatialGrid(cells=10)
    g.insert((0.0, 0.0, 0.1, 0.1), "A")     # top-left
    g.insert((0.85, 0.85, 0.95, 0.95), "B")  # bottom-right
    g.insert((0.4, 0.4, 0.6, 0.6), "C")      # centre
    # A query in the top-left returns A, never the bottom-right B.
    tl = g.query_bbox((0.0, 0.0, 0.05, 0.05))
    check("T26 top-left query returns A", "A" in tl)
    check("T26 top-left query excludes far-away B", "B" not in tl, tl)
    # A point query in the centre hits C only.
    cen = g.query_point(0.5, 0.5)
    check("T26 centre point -> C only", cen == ["C"], cen)
    # An overlapping query returns the overlapped item.
    check("T26 overlap query returns B", "B" in g.query_bbox((0.9, 0.9, 1.0, 1.0)))
    # Every inserted item is reachable by its own bbox (no false negatives).
    for bb, name in [((0.0, 0.0, 0.1, 0.1), "A"), ((0.85, 0.85, 0.95, 0.95), "B"),
                     ((0.4, 0.4, 0.6, 0.6), "C")]:
        check(f"T26 self-query finds {name}", name in g.query_bbox(bb))

    # Grid-pruned dedup == brute-force dedup (same result, fewer comparisons).
    c = mk_canvas()
    import random; random.seed(1)
    masks = []
    for _ in range(40):
        x = random.uniform(0, 0.9); y = random.uniform(0, 0.9); s = 0.05
        masks.append({"type": "poly",
                      "data": [[x, y], [x + s, y], [x + s, y + s], [x, y + s]],
                      "deleted": False, "source": "manual", "semiauto": True})
    c.annotations = list(masks)
    # New detector polys: half exact-overlap a mask (dup), half in open space.
    newp = []
    for m in masks[:20]:
        d = m["data"]; newp.append([[p[0] + 0.001, p[1] + 0.001] for p in d])  # dup
    for k in range(20):
        newp.append([[0.95, 0.0 + k * 0.002], [0.96, 0.0 + k * 0.002],
                     [0.96, 0.001 + k * 0.002]])  # tiny, in empty right edge
    c.set_annotations(polys=newp, poly_sources=["detector"] * len(newp))
    det = [a for a in c.annotations if a.get("source") == "detector"]
    check("T26 grid dedup drops the 20 duplicates", len(det) == 20, len(det))
    check("T26 grid dedup keeps the 20 non-overlapping", all(not a.get("semiauto") for a in det))
t26()

# ══════════════════════════════════════════════════════════════════════════
# T27: prompt Text/Boxes radio gating is consistent (DINO box greyed, etc.)
# ══════════════════════════════════════════════════════════════════════════
def t27():
    cases = [
        ("DINO (SwinT)",          True,  False),  # text-grounded -> box greyed
        ("DINO (SwinB)",          True,  False),
        ("YOLOE-vis",             False, True),   # visual prompts -> text greyed
        ("YOLOE-seg (one-shot)",  True,  True),   # both
        ("SAM3 (one-shot)",       True,  True),   # both
    ]
    for det, text_ok, box_ok in cases:
        w = mk_window()
        w.detector_choice = det; w.prompt_mode = "text"
        w.prompt_mode_text_btn = QtWidgets.QRadioButton()
        w.prompt_mode_boxes_btn = QtWidgets.QRadioButton()
        w._apply_prompt_radio_gating()
        check(f"T27 {det}: text radio enabled={text_ok}",
              w.prompt_mode_text_btn.isEnabled() == text_ok)
        check(f"T27 {det}: box radio enabled={box_ok}",
              w.prompt_mode_boxes_btn.isEnabled() == box_ok)
t27()

# ══════════════════════════════════════════════════════════════════════════
# T28: Manual Masks clips SAM to the drawn outline (no leaf-bleed past it)
# ══════════════════════════════════════════════════════════════════════════
def t28():
    outline = [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]]
    # SAM over-reached to the whole image -> clipped back to the outline.
    clip = ManualWindow._clip_poly_to_outline(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], outline)
    xs = [p[0] for p in clip]; ys = [p[1] for p in clip]
    check("T28 over-reaching SAM clipped to the outline bounds",
          min(xs) >= 0.3 - 1e-6 and max(xs) <= 0.7 + 1e-6
          and min(ys) >= 0.3 - 1e-6 and max(ys) <= 0.7 + 1e-6, clip)
    # SAM fully inside the outline is preserved.
    inside = [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]
    cl2 = ManualWindow._clip_poly_to_outline(inside, outline)
    check("T28 SAM inside the outline is kept",
          max(p[0] for p in cl2) <= 0.6 + 1e-6 and min(p[0] for p in cl2) >= 0.4 - 1e-6)
    # SAM entirely outside the outline -> nothing inside -> None.
    check("T28 SAM entirely outside -> None",
          ManualWindow._clip_poly_to_outline([[0.8, 0.8], [0.9, 0.8], [0.9, 0.9]], outline) is None)

    # End-to-end close: a SAM mask that exceeds the outline is committed clipped.
    w = mk_window(); c = mk_canvas(); w.image_label = c
    w.detector_choice = "DINO (SwinT)"; w.segmenter_choice = "SAM2 (tiny)"
    w.images = ["img.jpg"]; w.current_image_index = 0
    w.base_cv2_image = np.zeros((100, 100, 3), dtype=np.uint8)
    w.live_boxes = []; w.live_box_sources = []
    w._rebake_overlay = lambda: None
    w._persist_annotations = lambda silent=False: None
    class FR: masks = object()
    w._get_model = lambda key: (lambda src, **kw: [FR()])
    # crop-normalized poly that doesn't touch crop edges but, translated, exceeds
    # the outline -> must be clipped to the 0.3..0.7 square on commit.
    _orig_m2p = G["_mask_to_polys"]
    G["_mask_to_polys"] = lambda r: [[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]]
    c._mask_points = [[30, 30, 1], [70, 30, 1], [70, 70, 1], [30, 70, 1]]
    w._close_mask_object()
    semis = [a for a in c.annotations if a.get("semiauto")]
    check("T28 close commits exactly one clipped mask", len(semis) == 1, len(semis))
    if semis:
        xs = [p[0] for p in semis[0]["data"]]; ys = [p[1] for p in semis[0]["data"]]
        check("T28 committed mask stays within the outline (no leaf-bleed)",
              max(xs) <= 0.7 + 1e-3 and min(xs) >= 0.3 - 1e-3
              and max(ys) <= 0.7 + 1e-3 and min(ys) >= 0.3 - 1e-3,
              (min(xs), max(xs), min(ys), max(ys)))
    G["_mask_to_polys"] = _orig_m2p
t28()

# ══════════════════════════════════════════════════════════════════════════
# T29: side-by-side <-> swap moves titles, folder buttons, images, names
# ══════════════════════════════════════════════════════════════════════════
def t29():
    w = SideBySideWindow(None, None)
    # default: Ground Truth on the left, Synthetic on the right (as before)
    check("T29 default GT-left / synth-right", (w._left, w._right) == ("gt", "synth"))
    check("T29 default left title is GT", w.left_title.text() == "Ground Truth")
    check("T29 default left folder btn is GT", w.left_folder_btn.text() == "Open Ground Truth Folder")

    # load a pair so names/images map through the slots
    w.synth_images = ["s/synthA.png"]; w.gt_images = ["g/gtA.png"]
    w._build_pairs(); w._show_current()
    check("T29 left name shows GT file", w.left_name.text() == "gtA.png", w.left_name.text())
    check("T29 right name shows synth file", w.right_name.text() == "synthA.png", w.right_name.text())

    # edit the left (GT) title; it must follow GT across a swap
    w.left_title.setText("Reference"); w.titles[w._left] = "Reference"

    w._swap_sides()
    check("T29 swap toggles mapping", (w._left, w._right) == ("synth", "gt"))
    check("T29 swap: synth title now on left", w.left_title.text() == "Synthetic Images")
    check("T29 swap: edited GT title moved right", w.right_title.text() == "Reference")
    check("T29 swap: folder buttons swapped",
          w.left_folder_btn.text() == "Open Synthetic Folder"
          and w.right_folder_btn.text() == "Open Ground Truth Folder")
    check("T29 swap: filenames swapped",
          w.left_name.text() == "synthA.png" and w.right_name.text() == "gtA.png")

    # reversible
    w._swap_sides()
    check("T29 swap is reversible", (w._left, w._right) == ("gt", "synth")
          and w.left_name.text() == "gtA.png")

    # bulletproof: swapping with NO folders/pairs loaded must not crash
    w2 = SideBySideWindow(None, None)
    w2._swap_sides(); w2._swap_sides()
    check("T29 swap with no pairs is safe", (w2._left, w2._right) == ("gt", "synth"))
t29()

# ══════════════════════════════════════════════════════════════════════════
# T30: box-exemplar carry capability matrix (_detector_uses_box_exemplars).
#       Locks the rule: carry boxes forward only for detectors that RE-DETECT
#       from example boxes -- YOLOE-vis always; YOLOE-seg / SAM3 only in Boxes
#       mode; DINO never (it carries the typed text prompt instead).
# ══════════════════════════════════════════════════════════════════════════
def t30():
    # (detector, segmenter, text-mode result, boxes-mode result)
    cases = [
        ("DINO (SwinT)",         "SAM2 (tiny)", False, False),  # text-only model
        ("DINO (SwinB)",         "SAM2 (tiny)", False, False),
        ("YOLOE-vis",            "(none)",      True,  True),    # always box-driven
        ("YOLOE-seg (one-shot)", "(none)",      False, True),    # boxes mode only
        ("SAM3 (one-shot)",      "(none)",      False, True),    # boxes mode only
    ]
    for det, seg, text_res, box_res in cases:
        w = mk_window()
        w.detector_choice, w.segmenter_choice = det, seg
        w.prompt_mode = "text"
        check(f"T30 {det}: box-carry in Text mode = {text_res}",
              w._detector_uses_box_exemplars() == text_res,
              w._detector_uses_box_exemplars())
        w.prompt_mode = "boxes"
        check(f"T30 {det}: box-carry in Boxes mode = {box_res}",
              w._detector_uses_box_exemplars() == box_res,
              w._detector_uses_box_exemplars())
t30()

# ══════════════════════════════════════════════════════════════════════════
# T31: Auto Annotate Remaining availability matrix (_auto_annotate_available).
#       DINO needs text; YOLOE-vis needs a (drawn or carried) box; YOLOE-seg /
#       SAM3 accept either. No images -> always unavailable.
# ══════════════════════════════════════════════════════════════════════════
def t31():
    class _PromptLabel:
        """Minimal image_label exposing only the prompt-box getter."""
        def __init__(self, boxes): self._boxes = boxes
        def get_prompt_boxes_in_image_coords(self): return list(self._boxes)
    def mk(det, seg, mode, text, boxes, images=True, anchor=None):
        w = mk_window()
        w.detector_choice, w.segmenter_choice = det, seg
        w.prompt_mode = mode
        w.images = ["a.jpg"] if images else []
        w._carry_anchor = anchor
        w.image_label = _PromptLabel([[1, 2, 3, 4]] if boxes else [])
        w.prompt_entry = QtWidgets.QLineEdit("berry" if text else "")
        return w
    # No images -> never available, whatever the inputs.
    check("T31 no images -> unavailable",
          mk("SAM3 (one-shot)", "(none)", "boxes", True, True, images=False)
          ._auto_annotate_available() is False)
    # DINO: gated on text only.
    check("T31 DINO + text -> available",
          mk("DINO (SwinT)", "SAM2 (tiny)", "text", True, False)._auto_annotate_available())
    check("T31 DINO no text -> unavailable",
          mk("DINO (SwinT)", "SAM2 (tiny)", "text", False, False)._auto_annotate_available() is False)
    check("T31 DINO box-but-no-text -> unavailable",
          mk("DINO (SwinT)", "SAM2 (tiny)", "text", False, True)._auto_annotate_available() is False)
    # YOLOE-vis: gated on boxes only.
    check("T31 YOLOE-vis + drawn box -> available",
          mk("YOLOE-vis", "(none)", "boxes", False, True)._auto_annotate_available())
    check("T31 YOLOE-vis + carry anchor -> available",
          mk("YOLOE-vis", "(none)", "boxes", False, False, anchor=[[0.5, 0.5, 0.2, 0.2]])
          ._auto_annotate_available())
    check("T31 YOLOE-vis no box -> unavailable",
          mk("YOLOE-vis", "(none)", "boxes", False, False)._auto_annotate_available() is False)
    check("T31 YOLOE-vis text-but-no-box -> unavailable",
          mk("YOLOE-vis", "(none)", "boxes", True, False)._auto_annotate_available() is False)
    # YOLOE-seg / SAM3: either box or text is enough.
    for det in ("YOLOE-seg (one-shot)", "SAM3 (one-shot)"):
        check(f"T31 {det} + text only -> available",
              mk(det, "(none)", "text", True, False)._auto_annotate_available())
        check(f"T31 {det} + box only -> available",
              mk(det, "(none)", "boxes", False, True)._auto_annotate_available())
        check(f"T31 {det} neither -> unavailable",
              mk(det, "(none)", "boxes", False, False)._auto_annotate_available() is False)
t31()

# ══════════════════════════════════════════════════════════════════════════
# T32: carry toggle is enabled only for box-capable detectors (DINO greyed).
# ══════════════════════════════════════════════════════════════════════════
def t32():
    cases = [
        ("DINO (SwinT)",         "SAM2 (tiny)", False),
        ("DINO (SwinB)",         "SAM2 (tiny)", False),
        ("YOLOE-vis",            "(none)",      True),
        ("YOLOE-seg (one-shot)", "(none)",      True),
        ("SAM3 (one-shot)",      "(none)",      True),
    ]
    for det, seg, enabled in cases:
        w = mk_window()
        w.detector_choice, w.segmenter_choice = det, seg
        w.carry_forward_checkbox = QtWidgets.QPushButton()
        w.carry_forward_checkbox.setCheckable(True)
        w._refresh_carry_checkbox_enabled()
        check(f"T32 {det}: carry toggle enabled={enabled}",
              w.carry_forward_checkbox.isEnabled() == enabled,
              w.carry_forward_checkbox.isEnabled())
t32()

# ══════════════════════════════════════════════════════════════════════════
# T33: set_annotations returns a survivor mask and keeps live_boxes aligned.
#       A detector poly that duplicates a sticky hand-drawn mask is dropped;
#       kept_new marks it so the caller can shrink its parallel arrays. This is
#       the index-alignment fix behind the "purple masks vanish" report.
# ══════════════════════════════════════════════════════════════════════════
def t33():
    c = mk_canvas()
    # One sticky hand-drawn mask (semiauto) the user authored.
    sticky = {"type": "poly",
              "data": [[0.20, 0.20], [0.30, 0.20], [0.30, 0.30], [0.20, 0.30]],
              "deleted": False, "source": "manual", "semiauto": True}
    c.annotations = [dict(sticky)]
    # Two fresh detector polys: index 0 duplicates the sticky mask (must drop),
    # index 1 is in open space (must survive).
    dup    = [[0.201, 0.201], [0.301, 0.201], [0.301, 0.301], [0.201, 0.301]]
    unique = [[0.70, 0.70], [0.80, 0.70], [0.80, 0.80], [0.70, 0.80]]
    kept_new = c.set_annotations(polys=[dup, unique],
                                 poly_sources=["detector", "detector"])
    check("T33 set_annotations returns a survivor mask", kept_new is not None, kept_new)
    check("T33 dup poly dropped, unique kept", kept_new == [False, True], kept_new)
    det = [a for a in c.annotations if a.get("source") == "detector"]
    check("T33 only the non-duplicate detector poly remains", len(det) == 1, len(det))
    # Caller-side lockstep shrink (mirrors display_masks_with_borders): a parallel
    # live_boxes built from the ORIGINAL poly order, filtered by kept_new, stays
    # aligned with the surviving detector annotations.
    live_boxes = [[20, 20, 30, 30], [70, 70, 80, 80]]
    aligned = [b for b, k in zip(live_boxes, kept_new) if k]
    check("T33 live_boxes shrinks to match surviving anns", aligned == [[70, 70, 80, 80]], aligned)
t33()

# ══════════════════════════════════════════════════════════════════════════
# T34: budgeted model cache evicts the LRU model; unset budget = unbounded.
# ══════════════════════════════════════════════════════════════════════════
def t34():
    # Unset budget -> legacy unbounded behavior: nothing is evicted.
    os.environ.pop("AUTOANNOTATE_MODEL_BUDGET_GB", None)
    w = mk_window()
    w._model_cache = {}; w._model_lru = {}; w._model_lru_tick = 0
    w._get_model("sam3"); w._get_model("dino_swint"); w._get_model("yoloe_vis")
    check("T34 no budget -> all three resident",
          set(w._model_cache) == {"sam3", "dino_swint", "yoloe_vis"}, set(w._model_cache))
    # Small budget (1.0GB): SAM3 (3.3) loads alone, then loading DINO evicts it.
    os.environ["AUTOANNOTATE_MODEL_BUDGET_GB"] = "1.0"
    try:
        w2 = mk_window()
        w2._model_cache = {}; w2._model_lru = {}; w2._model_lru_tick = 0
        w2._get_model("sam3")
        check("T34 SAM3 loads even over budget (cannot evict the requested key)",
              "sam3" in w2._model_cache)
        w2._get_model("dino_swint")
        check("T34 loading DINO evicts the LRU SAM3", "sam3" not in w2._model_cache, set(w2._model_cache))
        check("T34 DINO is now resident", "dino_swint" in w2._model_cache)
    finally:
        os.environ.pop("AUTOANNOTATE_MODEL_BUDGET_GB", None)
t34()

# ══════════════════════════════════════════════════════════════════════════
# T35: two-stage batch chunk size reads AUTOANNOTATE_BATCH_CHUNK (default 8).
# ══════════════════════════════════════════════════════════════════════════
def t35():
    w = mk_window()
    os.environ.pop("AUTOANNOTATE_BATCH_CHUNK", None)
    check("T35 default chunk = 8", w._batch_chunk_size() == 8, w._batch_chunk_size())
    for val, exp in [("4", 4), ("1", 1), ("0", 8), ("-3", 8), ("notanint", 8)]:
        os.environ["AUTOANNOTATE_BATCH_CHUNK"] = val
        check(f"T35 chunk env {val!r} -> {exp}", w._batch_chunk_size() == exp,
              w._batch_chunk_size())
    os.environ.pop("AUTOANNOTATE_BATCH_CHUNK", None)
t35()

# ══════════════════════════════════════════════════════════════════════════
# T36: reject list is size-aware: deleting a big cluster box no longer wipes
#       the smaller model masks nested inside it, but a same-object re-detection
#       of the deleted box is still suppressed.
# ══════════════════════════════════════════════════════════════════════════
def t36():
    class _L:
        _orig_w = 100; _orig_h = 100
    w = mk_window()
    w.image_label = _L()
    # User deleted a big cluster mask occupying the 40..60 px square.
    w._rejected_boxes = [[0.40, 0.40, 0.60, 0.60]]
    small_berry      = [45, 45, 50, 50]   # nested inside the deleted cluster
    big_redetection  = [41, 41, 61, 61]   # ~the same cluster box, re-detected
    far_away         = [5, 5, 12, 12]     # unrelated detection elsewhere
    kept, _ = w._drop_rejected([small_berry, big_redetection, far_away], None)
    check("T36 nested small berry is KEPT (not a re-detection)",
          small_berry in kept, kept)
    check("T36 same-object re-detection is dropped",
          big_redetection not in kept, kept)
    check("T36 unrelated far-away detection is kept", far_away in kept, kept)
    # Polys shrink in lockstep with the kept boxes.
    kb, kp = w._drop_rejected([small_berry, big_redetection], [["pA"], ["pB"]])
    check("T36 polys stay aligned with kept boxes",
          kb == [small_berry] and kp == [["pA"]], (kb, kp))
t36()

# ══════════════════════════════════════════════════════════════════════════
# T37: switching models offloads the ones the new pipeline no longer uses,
#       keeping only the active detector + segmenter (+ interactive SAM).
# ══════════════════════════════════════════════════════════════════════════
def t37():
    import os as _os
    _os.environ.pop("AUTOANNOTATE_KEEP_MODELS_WARM", None)
    def mk(det, seg, cached):
        w = mk_window()
        w._busy = False
        w.detector_choice, w.segmenter_choice = det, seg
        w._model_cache = {k: object() for k in cached}
        w._model_lru = {k: 1 for k in cached}
        return w
    # DINO+SAM2: keep only dino_swint + sam2_t; drop the rest.
    w = mk("DINO (SwinT)", "SAM2 (tiny)",
           ["dino_swint", "sam2_t", "sam3", "yoloe_vis", "dino_swinb"])
    w._offload_unused_models()
    check("T37 DINO+SAM2 keeps only its two models",
          set(w._model_cache) == {"dino_swint", "sam2_t"}, set(w._model_cache))
    # YOLOE-vis one-shot (segmenter none): keep only yoloe_vis.
    w = mk("YOLOE-vis", "(none)", ["dino_swint", "sam2_t", "yoloe_vis"])
    w._offload_unused_models()
    check("T37 YOLOE-vis keeps only yoloe_vis",
          set(w._model_cache) == {"yoloe_vis"}, set(w._model_cache))
    # SAM3 one-shot: keep sam3_det + the interactive sam3 it uses for semi-auto.
    w = mk("SAM3 (one-shot)", "(none)", ["sam3_det", "sam3", "dino_swint"])
    w._offload_unused_models()
    check("T37 SAM3 keeps sam3_det + interactive sam3",
          set(w._model_cache) == {"sam3_det", "sam3"}, set(w._model_cache))
    # KEEP_MODELS_WARM=1 disables offload entirely.
    _os.environ["AUTOANNOTATE_KEEP_MODELS_WARM"] = "1"
    try:
        w = mk("YOLOE-vis", "(none)", ["dino_swint", "sam2_t", "yoloe_vis"])
        w._offload_unused_models()
        check("T37 KEEP_MODELS_WARM keeps everything",
              set(w._model_cache) == {"dino_swint", "sam2_t", "yoloe_vis"}, set(w._model_cache))
    finally:
        _os.environ.pop("AUTOANNOTATE_KEEP_MODELS_WARM", None)
    # Busy guard: never offload mid-inference.
    w = mk("YOLOE-vis", "(none)", ["dino_swint", "yoloe_vis"])
    w._busy = True
    w._offload_unused_models()
    check("T37 busy guard skips offload",
          set(w._model_cache) == {"dino_swint", "yoloe_vis"}, set(w._model_cache))
t37()

# ══════════════════════════════════════════════════════════════════════════
# T38: _imread_cached decodes each path once and returns independent copies.
# ══════════════════════════════════════════════════════════════════════════
def t38():
    w = mk_window()
    calls = {"n": 0}
    orig = G.get("imread_unicode")
    def counting(p):
        calls["n"] += 1
        return np.zeros((8, 8, 3), dtype=np.uint8)
    G["imread_unicode"] = counting
    try:
        a = w._imread_cached("a.jpg")
        b = w._imread_cached("a.jpg")
        check("T38 same path decodes once", calls["n"] == 1, calls["n"])
        check("T38 returns distinct copies", a is not b)
        a[0, 0, 0] = 255
        c = w._imread_cached("a.jpg")
        check("T38 caller mutation does not corrupt cache", int(c[0, 0, 0]) == 0, int(c[0, 0, 0]))
        w._imread_cached("b.jpg")
        check("T38 different path re-decodes", calls["n"] == 2, calls["n"])
        G["imread_unicode"] = lambda p: None
        check("T38 unreadable -> None", w._imread_cached("z.jpg") is None)
        check("T38 unreadable None is cached (not retried)", w._imread_cached("z.jpg") is None)
    finally:
        if orig is not None:
            G["imread_unicode"] = orig

t38()

# ══════════════════════════════════════════════════════════════════════════
# T39: AUTOANNOTATE_RELEASE_EVERY gates the per-image memory release cadence.
# ══════════════════════════════════════════════════════════════════════════
def t39():
    import os as _os
    w = mk_window()
    _os.environ.pop("AUTOANNOTATE_RELEASE_EVERY", None)
    check("T39 default release-every = 1", w._release_every() == 1, w._release_every())
    for v, exp in [("3", 3), ("1", 1), ("0", 1), ("-2", 1), ("notanint", 1)]:
        _os.environ["AUTOANNOTATE_RELEASE_EVERY"] = v
        check(f"T39 env {v!r} -> {exp}", w._release_every() == exp, w._release_every())
    # Cadence must not raise (torch undefined in this harness -> caught); force
    # always runs. Just exercise the paths.
    _os.environ["AUTOANNOTATE_RELEASE_EVERY"] = "3"
    try:
        w._release_tick = 0
        for _ in range(4):
            w._release_inference_memory()
        w._release_inference_memory(force=True)
        check("T39 cadence + force run without error", True)
    finally:
        _os.environ.pop("AUTOANNOTATE_RELEASE_EVERY", None)

t39()

# ══════════════════════════════════════════════════════════════════════════
# T40: lazy VLM: generate_prompts(None model) loads on demand and degrades
#       gracefully (returns []) when the VLM can't load, instead of crashing.
# ══════════════════════════════════════════════════════════════════════════
def t40():
    gp = G["generate_prompts"]
    # Heavy VLM deps are absent in this harness, so ensure_llm() fails inside its
    # try/except and returns (None, None); generate_prompts must return [] before
    # ever touching the image. (Locks the "lazy + safe" contract.)
    G["_llm_cache"] = None
    out = gp("nonexistent.jpg", "berry", None, None)
    check("T40 generate_prompts(None) -> [] (lazy load degrades gracefully)", out == [], out)
    check("T40 failed load is cached as (None, None)", G["_llm_cache"] == (None, None), G.get("_llm_cache"))

t40()

# ══════════════════════════════════════════════════════════════════════════
# T41: _retry_factor parses/clamps AUTOANNOTATE_RETRY_FACTOR to (0,1).
# ══════════════════════════════════════════════════════════════════════════
def t41():
    import os as _os
    w = mk_window()
    _os.environ.pop("AUTOANNOTATE_RETRY_FACTOR", None)
    check("T41 default factor = 0.5", w._retry_factor() == 0.5, w._retry_factor())
    for v, exp in [("0.3", 0.3), ("0.5", 0.5), ("1.0", 0.5), ("0", 0.5), ("-1", 0.5), ("x", 0.5)]:
        _os.environ["AUTOANNOTATE_RETRY_FACTOR"] = v
        check(f"T41 factor {v!r} -> {exp}", w._retry_factor() == exp, w._retry_factor())
    _os.environ.pop("AUTOANNOTATE_RETRY_FACTOR", None)

t41()

# ══════════════════════════════════════════════════════════════════════════
# T42: _detect_with_retry: ONE bounded retry at a lower threshold on empty.
# ══════════════════════════════════════════════════════════════════════════
def t42():
    import os as _os
    _os.environ.pop("AUTOANNOTATE_RETRY_FACTOR", None)
    w = mk_window()
    # First call empty, retry returns a box.
    seq = [([], None), ([[1, 2, 3, 4]], "yres")]
    calls = []
    def fake(image_path, prompt, det, mask, carried, ref=None):
        calls.append(det)
        return seq[len(calls) - 1]
    w._run_detector = fake
    boxes, yres, used, retried = w._detect_with_retry("a.jpg", "p", 0.30, 0.50, [], None)
    check("T42 retried once on empty (2 detector calls)", retried is True and len(calls) == 2, (retried, calls))
    check("T42 retry threshold is lower", used < 0.30, used)
    check("T42 retry returned the box", boxes == [[1, 2, 3, 4]], boxes)
    # Non-empty first -> no retry (exactly one detector call).
    calls.clear()
    w._run_detector = lambda *a, **k: (calls.append(a[3]) or [[9, 9, 9, 9]], "y") if True else None
    def fake2(image_path, prompt, det, mask, carried, ref=None):
        calls.append(det); return ([[9, 9, 9, 9]], "y")
    w._run_detector = fake2
    boxes, yres, used, retried = w._detect_with_retry("a.jpg", "p", 0.30, 0.50, [], None)
    check("T42 no retry when first is non-empty", retried is False and len(calls) == 1, (retried, calls))
    check("T42 used threshold unchanged when not retried", used == 0.30, used)

t42()

# ══════════════════════════════════════════════════════════════════════════
# T43: _finalize_review writes boxes/segments subfolders + CSV, copies images.
# ══════════════════════════════════════════════════════════════════════════
def t43():
    import os as _os, tempfile, csv as _csv
    w = mk_window()
    d = tempfile.mkdtemp()
    img1 = _os.path.join(d, "e1.jpg"); open(img1, "wb").write(b"x")
    img2 = _os.path.join(d, "f2.jpg"); open(img2, "wb").write(b"y")
    out = _os.path.join(d, "out"); _os.makedirs(out)
    review = [
        {"image": img1, "stage": "boxes", "status": "empty", "reason": "0 dets",
         "detector": "DINO (SwinT)", "prompt": "berry", "orig": "0.30", "retry": "0.15"},
        {"image": img2, "stage": "segments", "status": "failed", "reason": "boom",
         "detector": "SAM3 (one-shot)", "prompt": "", "orig": "0.50", "retry": ""},
    ]
    rd = w._finalize_review(review, out)
    check("T43 review dir created", bool(rd) and _os.path.isdir(rd), rd)
    check("T43 empty image copied to boxes/", _os.path.exists(_os.path.join(rd, "boxes", "e1.jpg")))
    check("T43 failed image copied to segments/", _os.path.exists(_os.path.join(rd, "segments", "f2.jpg")))
    rep_path = _os.path.join(rd, "review_report.csv")
    check("T43 review_report.csv written", _os.path.exists(rep_path))
    rows = list(_csv.reader(open(rep_path)))
    check("T43 report = header + 2 rows", len(rows) == 3, len(rows))
    check("T43 header has stage+status+reason",
          rows[0][1:4] == ["stage", "status", "reason"], rows[0])
    check("T43 empty review -> None (no folder/file spam)", w._finalize_review([], out) is None)

t43()

# ══════════════════════════════════════════════════════════════════════════
# T44: _review folder is idempotent: each run clears the previous run's copies;
#       a clean run removes the folder entirely. Guards never delete non-_review.
# ══════════════════════════════════════════════════════════════════════════
def t44():
    import os as _os, tempfile
    w = mk_window()
    d = tempfile.mkdtemp(); out = _os.path.join(d, "out"); _os.makedirs(out)
    # Review folder is now scoped per model: <output>/_review/<model_tag>/.
    tag = w._model_tag()
    rd = _os.path.join(out, "_review", tag)
    a = _os.path.join(d, "a.jpg"); open(a, "wb").write(b"x")
    b = _os.path.join(d, "b.jpg"); open(b, "wb").write(b"y")
    mk_entry = lambda p: {"image": p, "stage": "boxes", "status": "empty",
                          "reason": "r", "detector": "D", "prompt": "p",
                          "orig": "0.30", "retry": "0.15"}
    w._finalize_review([mk_entry(a)], out)
    check("T44 run1 copied a.jpg", _os.path.exists(_os.path.join(rd, "boxes", "a.jpg")))
    # Second run with a DIFFERENT empty image: stale a.jpg must be gone.
    w._finalize_review([mk_entry(b)], out)
    check("T44 run2 cleared stale a.jpg", not _os.path.exists(_os.path.join(rd, "boxes", "a.jpg")))
    check("T44 run2 has the new b.jpg", _os.path.exists(_os.path.join(rd, "boxes", "b.jpg")))
    # Clean run (no problems): this model's review folder is removed, and the
    # now-empty _review parent is dropped too so a clean run leaves no clutter.
    check("T44 clean run returns None", w._finalize_review([], out) is None)
    check("T44 clean run removed model review dir", not _os.path.exists(rd))
    check("T44 clean run removed empty _review parent",
          not _os.path.exists(_os.path.join(out, "_review")))
    # Guard: never deletes a folder that isn't the model's review subfolder.
    safe = _os.path.join(out, "boxes"); _os.makedirs(safe, exist_ok=True)
    open(_os.path.join(safe, "keep.txt"), "wb").write(b"z")
    w._reset_review_dir(out, tag)
    check("T44 guard leaves output/boxes untouched",
          _os.path.exists(_os.path.join(safe, "keep.txt")))

t44()

# ══════════════════════════════════════════════════════════════════════════
# T45: AUTOANNOTATE_MAX_AREA_FRAC parses/clamps to (0,1]; default 0.5.
# ══════════════════════════════════════════════════════════════════════════
def t45():
    import os as _os
    w = mk_window()
    _os.environ.pop("AUTOANNOTATE_MAX_AREA_FRAC", None)
    check("T45 default max-area-frac = 0.5", w._max_area_frac() == 0.5, w._max_area_frac())
    for v, exp in [("0.9", 0.9), ("0.4", 0.4), ("1.0", 1.0), ("0", 0.5), ("1.5", 0.5),
                   ("-0.2", 0.5), ("nope", 0.5)]:
        _os.environ["AUTOANNOTATE_MAX_AREA_FRAC"] = v
        check(f"T45 frac {v!r} -> {exp}", w._max_area_frac() == exp, w._max_area_frac())
    _os.environ.pop("AUTOANNOTATE_MAX_AREA_FRAC", None)

t45()

# ══════════════════════════════════════════════════════════════════════════
# T46: Auto Annotate Remaining start index. "Use First Image as Prompt"
# (carry) ON includes the CURRENT (prompt) image so the output folder gets N
# files, not N-1; OFF keeps the legacy "remaining only" start (current + 1).
# ══════════════════════════════════════════════════════════════════════════
def t46():
    w = mk_window()
    w.images = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    w.current_image_index = 1
    w.carry_forward_checkbox = QtWidgets.QCheckBox()

    w.carry_forward_checkbox.setChecked(True)
    check("T46 carry ON starts AT current index", w._batch_start_index() == 1,
          w._batch_start_index())
    check("T46 carry ON includes prompt image (N - current targets)",
          len(w.images[w._batch_start_index():]) == 3,
          len(w.images[w._batch_start_index():]))

    w.carry_forward_checkbox.setChecked(False)
    check("T46 carry OFF starts at NEXT index", w._batch_start_index() == 2,
          w._batch_start_index())
    check("T46 carry OFF excludes current image (N-1 from current)",
          len(w.images[w._batch_start_index():]) == 2,
          len(w.images[w._batch_start_index():]))

    delattr(w, "carry_forward_checkbox")
    check("T46 no carry checkbox -> legacy next-index", w._batch_start_index() == 2,
          w._batch_start_index())

t46()

# ══════════════════════════════════════════════════════════════════════════
# T47: _clear_segment_file overwrites the segments label with an EMPTY file
# (keeps boxes/ and segments/ in lockstep when an image yields no mask, so a
# stale mask from a prior interactive save can't linger).
# ══════════════════════════════════════════════════════════════════════════
def t47():
    import tempfile, os as _os
    w = mk_window()
    d = tempfile.mkdtemp()
    # Pre-seed a stale segments file, as a prior interactive save would leave.
    stale = _os.path.join(d, "img.txt")
    with open(stale, "w") as f:
        f.write("0 0.1 0.1 0.2 0.2 0.3 0.3\n")
    w._clear_segment_file("/some/path/img.jpg", d)
    check("T47 segments file still present after clear", _os.path.exists(stale))
    check("T47 stale mask cleared to empty", _os.path.getsize(stale) == 0,
          _os.path.getsize(stale))
    # Creates the dir + empty file when absent, named by the image stem.
    d2 = _os.path.join(d, "sub")
    w._clear_segment_file("/x/y/foo.png", d2)
    check("T47 creates empty segments file in a new dir",
          _os.path.exists(_os.path.join(d2, "foo.txt")))

# ══════════════════════════════════════════════════════════════════════════
# T48: pipeline DISPATCH CONTRACT. For every supported detector x segmenter x
# prompt-mode, _run_detector must route into a REAL model call, never the
# silent `return [], None` fall-through. This is the exact safety net that was
# missing when YOLOE-seg + a separate segmenter (two-stage) silently returned 0
# boxes: the detector helpers are stubbed to raise a sentinel the instant they
# are reached, so "no sentinel raised" == the fall-through bug.
# ══════════════════════════════════════════════════════════════════════════
def t48():
    import types as _types
    class _Dispatched(Exception):
        def __init__(self, who):
            super().__init__(who); self.who = who
    def _raiser(name):
        def _f(*a, **k):
            raise _Dispatched(name)
        return _f

    # Inject detector helpers into the cell-4 globals (referenced as globals
    # inside _run_detector); save + restore so other tests are unaffected.
    helpers = ["run_dino_from_model", "run_yoloe_vis", "run_yoloe_text", "run_sam3_text"]
    saved = {h: G.get(h) for h in helpers}
    for h in helpers:
        G[h] = _raiser(h)

    w = mk_window()
    w.output_folder = "/tmp"
    w._get_model = lambda key: object()
    w.image_label = _types.SimpleNamespace(_orig_w=0, _orig_h=0,
                                           annotations=[], get_active_annotations=lambda: [])
    w._run_sam3_boxes_partitioned = _raiser("sam3_boxes_partitioned")
    w._run_sam3_crop_composite = _raiser("sam3_crop_composite")

    BOX = [[10.0, 10.0, 60.0, 60.0]]
    # detector_choice, segmenter_choice, prompt_mode, prompt_boxes, expected target
    matrix = [
        ("DINO (SwinT)",        "SAM2",               "text",  None, "run_dino_from_model"),
        ("DINO (SwinB)",        "SAM2",               "text",  None, "run_dino_from_model"),
        ("YOLOE-vis",           "(none)",             "boxes", BOX,  "run_yoloe_vis"),
        ("YOLOE-vis",           "SAM3 (interactive)", "boxes", BOX,  "run_yoloe_vis"),
        ("YOLOE-seg",           "(none)",             "text",  None, "run_yoloe_text"),
        ("YOLOE-seg",           "(none)",             "boxes", BOX,  "run_yoloe_vis"),
        ("YOLOE-seg",           "SAM2",               "text",  None, "run_yoloe_text"),   # two-stage (regression)
        ("YOLOE-seg",           "SAM3 (interactive)", "boxes", BOX,  "run_yoloe_vis"),    # two-stage (regression)
        ("SAM3 (interactive)",  "(none)",             "text",  None, "run_sam3_text"),
        ("SAM3 (interactive)",  "(none)",             "boxes", BOX,  "sam3_boxes_partitioned"),
    ]
    try:
        for det, seg, mode, boxes, expected in matrix:
            w.detector_choice = det
            w.segmenter_choice = seg
            w.prompt_mode = mode
            w._oneshot_polys_aligned = None
            who = None
            try:
                w._run_detector("/tmp/x.jpg", "berry", 0.2, 0.2, boxes, ref=None)
            except _Dispatched as d:
                who = d.who
            label = f"T48 {det} + {seg} [{mode}]"
            check(f"{label} dispatches (no silent fall-through)", who is not None,
                  "returned without calling any detector")
            check(f"{label} -> {expected}", who == expected, f"got {who!r}")
    finally:
        for h, v in saved.items():
            if v is None:
                G.pop(h, None)
            else:
                G[h] = v

t48()

# ══════════════════════════════════════════════════════════════════════════
# T49: parse_prompt_classes splits comma-separated prompts into ordered class
# names (ids in saved labels follow this order).
# ══════════════════════════════════════════════════════════════════════════
def t49():
    p = G["parse_prompt_classes"]
    check("T49 empty -> []", p("") == [])
    check("T49 None -> []", p(None) == [])
    check("T49 single", p("blueberry") == ["blueberry"])
    check("T49 two classes", p("blueberry, leaf") == ["blueberry", "leaf"])
    check("T49 messy commas/space", p(" a ,, b ") == ["a", "b"])

t49()

# ══════════════════════════════════════════════════════════════════════════
# T50: label writers accept per-item classes and default to class 0 exactly
# as before (single-class back-compat).
# ══════════════════════════════════════════════════════════════════════════
def t50():
    import tempfile, os as _os
    from autoannotate.pipeline.labels import save_boxes_yolo as _sb, \
        save_polys_yolo as _sp, save_classes_txt as _sc
    d = tempfile.mkdtemp()
    img = _os.path.join(d, "img.jpg")  # cv2.imread is stubbed to 100x100
    _sb([[10, 10, 30, 30], [40, 40, 60, 60]], img, d)
    first_cols = [l.split()[0] for l in open(_os.path.join(d, "img.txt"))]
    check("T50 boxes default all class 0", first_cols == ["0", "0"], first_cols)
    _sb([[10, 10, 30, 30], [40, 40, 60, 60]], img, d, classes=[0, 1])
    first_cols = [l.split()[0] for l in open(_os.path.join(d, "img.txt"))]
    check("T50 boxes classes column written", first_cols == ["0", "1"], first_cols)
    tri = [[0.1, 0.1], [0.3, 0.1], [0.2, 0.3]]
    tri2 = [[0.5, 0.5], [0.7, 0.5], [0.6, 0.7]]
    _sp([tri, None, tri2], d, img, classes=[2, 0, 1])
    first_cols = [l.split()[0] for l in open(_os.path.join(d, "img.txt"))]
    check("T50 polys skip degenerate but keep class alignment",
          first_cols == ["2", "1"], first_cols)
    _sc(["blueberry", "leaf"], d)
    names = open(_os.path.join(d, "classes.txt")).read().splitlines()
    check("T50 classes.txt one name per line", names == ["blueberry", "leaf"], names)

t50()

# ══════════════════════════════════════════════════════════════════════════
# T51: _nms_dedup keeps its 2-tuple return without classes (back-compat),
# shrinks the class list in lockstep, and by default only ever suppresses a box
# with another box of the SAME class -- two classes claiming one object is a
# real disagreement and both rows are written.
# ══════════════════════════════════════════════════════════════════════════
def t51():
    w = mk_window()
    boxes = [[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]]  # first two dup
    out = w._nms_dedup(boxes)
    check("T51 no classes -> 2-tuple", len(out) == 2 and out[1] is None, out)
    check("T51 dup dropped", len(out[0]) == 2, out[0])

    out = w._nms_dedup(boxes, None, classes=[0, 1, 2])
    check("T51 with classes -> 3-tuple", len(out) == 3, len(out))
    check("T51 cross-class overlap kept", out[2] == [0, 1, 2], out[2])
    check("T51 cross-class boxes kept", len(out[0]) == 3, out[0])

    # Same class on the overlapping pair -> the duplicate IS dropped.
    out = w._nms_dedup(boxes, None, classes=[0, 0, 2])
    check("T51 same-class dup dropped", out[2] == [0, 2], out[2])
    check("T51 same-class boxes shrink", len(out[0]) == 2, out[0])

    # cross_class=True restores the old suppress-everything behavior.
    out = w._nms_dedup(boxes, None, classes=[0, 1, 2], cross_class=True)
    check("T51 cross_class=True suppresses across classes", out[2] == [0, 2], out[2])

    # polys shrink alongside classes, and only for same-class dups.
    polys = [[[0, 0]], [[1, 1]], [[2, 2]]]
    out = w._nms_dedup(boxes, polys, classes=[3, 3, 3])
    check("T51 polys shrink in lockstep", out[1] == [[[0, 0]], [[2, 2]]], out[1])

t51()

# ══════════════════════════════════════════════════════════════════════════
# T52: suppress_negative_hits drops every negative detection and any positive
# overlapping one; empty negatives are a no-op.
# ══════════════════════════════════════════════════════════════════════════
def t52():
    from autoannotate.pipeline.postfilter import suppress_negative_hits as f
    boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [80, 80, 90, 90]]
    cls   = [0, 1, 0, 1]   # n_pos=1 -> class 1 = negative
    polys = ["p0", "p1", "p2", "p3"]
    b, c, p = f(boxes, cls, polys, n_pos=1)
    check("T52 negatives removed", all(v < 1 for v in c), c)
    check("T52 overlapping positive dropped, distant kept",
          b == [[50, 50, 60, 60]], b)
    check("T52 polys aligned", p == ["p2"], p)
    b, c, p = f([[0, 0, 10, 10]], [0], None, n_pos=1)
    check("T52 no negatives -> no-op", b == [[0, 0, 10, 10]] and p is None, (b, p))
    b, c, p = f([], [], None, n_pos=1)
    check("T52 empty input -> empty", b == [] and c == [], (b, c))

t52()

# ══════════════════════════════════════════════════════════════════════════
# T53: _batch_targets. Recycle OFF == the old forward slice; ON appends the
# earlier images AT THE END (order preserved), interacting correctly with the
# carry checkbox via _batch_start_index.
# ══════════════════════════════════════════════════════════════════════════
def t53():
    w = mk_window()
    w.images = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    w.current_image_index = 2
    w.carry_forward_checkbox = QtWidgets.QCheckBox()
    w.recycle_checkbox = QtWidgets.QCheckBox()

    w.carry_forward_checkbox.setChecked(False)
    w.recycle_checkbox.setChecked(False)
    check("T53 recycle OFF == old slice", w._batch_targets() == ["d.jpg"],
          w._batch_targets())
    w.recycle_checkbox.setChecked(True)
    check("T53 recycle ON appends earlier at the end",
          w._batch_targets() == ["d.jpg", "a.jpg", "b.jpg", "c.jpg"],
          w._batch_targets())
    w.carry_forward_checkbox.setChecked(True)
    check("T53 recycle ON + carry ON: current not duplicated",
          w._batch_targets() == ["c.jpg", "d.jpg", "a.jpg", "b.jpg"],
          w._batch_targets())
    delattr(w, "recycle_checkbox")
    check("T53 no recycle widget -> old slice (headless-safe)",
          w._batch_targets() == ["c.jpg", "d.jpg"], w._batch_targets())

t53()

# ══════════════════════════════════════════════════════════════════════════
# T54: _parse_saved_labels round-trips boxes + segments with class columns and
# skips the box line that duplicates a polygon (label files carry a box per
# polygon; loading both would double every mask).
# ══════════════════════════════════════════════════════════════════════════
def t54():
    import tempfile, os as _os
    f = G["_parse_saved_labels"]
    d = tempfile.mkdtemp()
    seg_path = _os.path.join(d, "seg.txt")
    box_path = _os.path.join(d, "box.txt")
    with open(seg_path, "w") as fh:
        fh.write("1 0.1 0.1 0.3 0.1 0.2 0.3\n")     # triangle, class 1
        fh.write("0 0.5\n")                          # malformed -> skipped
    with open(box_path, "w") as fh:
        fh.write("1 0.2 0.2 0.2 0.2\n")              # duplicates the poly bbox
        fh.write("0 0.7 0.7 0.1 0.1\n")              # distinct box, class 0
    rects, rect_cls, polys, poly_cls = f(box_path, seg_path)
    check("T54 poly loaded with class", len(polys) == 1 and poly_cls == [1],
          (len(polys), poly_cls))
    check("T54 duplicate box of poly skipped", rects == [[0.7, 0.7, 0.1, 0.1]], rects)
    check("T54 rect class kept", rect_cls == [0], rect_cls)
    rects, rect_cls, polys, poly_cls = f(_os.path.join(d, "none.txt"), None)
    check("T54 missing files -> all empty",
          rects == [] and polys == [] and rect_cls == [] and poly_cls == [],
          (rects, polys))

t54()

# ══════════════════════════════════════════════════════════════════════════
# T55: set_annotations stores per-annotation class ids only when passed;
# omitted -> no 'cls' key (readers fall back to 0 exactly as before).
# ══════════════════════════════════════════════════════════════════════════
def t55():
    c = AnnotationCanvas()
    tri = [[0.1, 0.1], [0.3, 0.1], [0.2, 0.3]]
    c.set_annotations(polys=[tri], rects=[[0.6, 0.6, 0.2, 0.2]],
                      poly_cls=[2], rect_cls=[1])
    anns = c.get_active_annotations()
    check("T55 poly cls stored", anns[0].get('cls') == 2, anns[0])
    check("T55 rect cls stored", anns[1].get('cls') == 1, anns[1])
    c.set_annotations(polys=[tri], rects=[[0.6, 0.6, 0.2, 0.2]])
    anns = c.get_active_annotations()
    check("T55 omitted -> no cls key",
          all('cls' not in a for a in anns), anns)

t55()

# ══════════════════════════════════════════════════════════════════════════
# T56: _run_detector class side channel. Single-class prompt leaves
# _det_classes_aligned None (guaranteed no-behavior-change fast path);
# multi-class reads per-detection ids; a negative prompt runs ONE pass over
# pos+neg and suppresses negatives and overlapping positives.
# ══════════════════════════════════════════════════════════════════════════
def t56():
    import types as _types

    class _Boxes:
        def __init__(self, xyxy, cls=None):
            self.xyxy = _types.SimpleNamespace(tolist=lambda: list(xyxy))
            if cls is not None:
                self.cls = _types.SimpleNamespace(tolist=lambda: list(cls))

    class _Result:
        def __init__(self, xyxy, cls=None):
            self.boxes = _Boxes(xyxy, cls)
            self.masks = object()
            self.orig_shape = (100, 100)

    tri_a = [[0.1, 0.1], [0.3, 0.1], [0.2, 0.3]]
    tri_b = [[0.5, 0.5], [0.7, 0.5], [0.6, 0.7]]

    saved_yt = G.get("run_yoloe_text")
    saved_rcp = G.get("result_clean_polys")
    captured = {}

    def fake_yoloe_text(model, image_path, prompt, **k):
        captured["prompt"] = prompt
        return None, [captured["result"]]

    G["run_yoloe_text"] = fake_yoloe_text
    G["result_clean_polys"] = lambda r: [tri_a, tri_b]

    w = mk_window()
    w.output_folder = "/tmp"
    w.detector_choice = "YOLOE-seg (one-shot)"
    w.segmenter_choice = "(none)"
    w.prompt_mode = "text"
    w._get_model = lambda key: object()
    w.image_label = _types.SimpleNamespace(_orig_w=100, _orig_h=100,
                                           annotations=[],
                                           get_active_annotations=lambda: [])
    try:
        # Single class: side channel must stay None.
        captured["result"] = _Result([[10, 10, 30, 30], [50, 50, 70, 70]], cls=[0, 0])
        boxes, _ = w._run_detector("/tmp/x.jpg", "berry", 0.2, 0.2, None)
        check("T56 single class -> 2 boxes", len(boxes) == 2, boxes)
        check("T56 single class -> classes None", w._det_classes_aligned is None,
              w._det_classes_aligned)

        # Multi class: per-detection ids ride the side channel.
        captured["result"] = _Result([[10, 10, 30, 30], [50, 50, 70, 70]], cls=[0, 1])
        boxes, _ = w._run_detector("/tmp/x.jpg", "berry, leaf", 0.2, 0.2, None)
        check("T56 multi class -> classes aligned",
              w._det_classes_aligned == [0, 1], w._det_classes_aligned)

        # Negative prompt: one pass over pos+neg, negatives suppressed.
        w.neg_prompt_entry = _types.SimpleNamespace(text=lambda: "leaf")
        captured["result"] = _Result([[10, 10, 30, 30], [50, 50, 70, 70]], cls=[0, 1])
        boxes, _ = w._run_detector("/tmp/x.jpg", "berry", 0.2, 0.2, None)
        check("T56 negative: combined one-pass prompt",
              captured["prompt"] == "berry, leaf", captured["prompt"])
        check("T56 negative hit dropped, positive kept",
              len(boxes) == 1 and w._det_classes_aligned == [0],
              (boxes, w._det_classes_aligned))

        # Negative overlapping the positive suppresses the positive too.
        captured["result"] = _Result([[10, 10, 30, 30], [12, 12, 32, 32]], cls=[0, 1])
        boxes, _ = w._run_detector("/tmp/x.jpg", "berry", 0.2, 0.2, None)
        check("T56 overlapping positive suppressed", len(boxes) == 0, boxes)
    finally:
        if saved_yt is not None:
            G["run_yoloe_text"] = saved_yt
        if saved_rcp is not None:
            G["result_clean_polys"] = saved_rcp

t56()

# ══════════════════════════════════════════════════════════════════════════
# T57: SAM3 box exemplars are searched ONE CLASS PER PASS. ultralytics forces
# nc=1 whenever bboxes are passed, so a single flat call blends two classes
# into one concept and returns one class -- the bug that made a two-class run
# write class 1 on every row. One distinct class must still cost exactly one
# pass (the hot path), and each pass's detections carry its class.
# ══════════════════════════════════════════════════════════════════════════
def t57():
    import types as _types

    w = mk_window()
    w.prompt_mode = "boxes"
    w.active_class = 0
    calls = []

    def fake_partitioned(image_path, exemplars, conf, text, supplementary_exemplars=None):
        calls.append({"exemplars": list(exemplars),
                      "anchors": list(supplementary_exemplars or [])})
        # One detection per exemplar, offset so nothing overlaps across passes.
        n = len(exemplars) or 1
        base = len(calls) * 100
        boxes = [[base + i * 10, 0, base + i * 10 + 5, 5] for i in range(n)]
        polys = [[[0.1, 0.1], [0.2, 0.1], [0.15, 0.2]] for _ in range(n)]
        return boxes, polys, "results-sentinel"

    w._run_sam3_boxes_partitioned = fake_partitioned

    # Single class -> exactly one pass, results object passed straight through.
    calls.clear()
    boxes, polys, cls, results = w._run_sam3_boxes_multiclass(
        "/tmp/x.jpg", [[0, 0, 5, 5], [9, 9, 14, 14]], [2, 2], 0.2, "")
    check("T57 single class -> one pass", len(calls) == 1, len(calls))
    check("T57 single class -> tagged with that class", cls == [2, 2], cls)
    check("T57 single class -> results passed through",
          results == "results-sentinel", results)

    # Two classes -> two passes, each seeded with only its own exemplars.
    calls.clear()
    boxes, polys, cls, results = w._run_sam3_boxes_multiclass(
        "/tmp/x.jpg",
        [[0, 0, 5, 5], [9, 9, 14, 14], [20, 20, 25, 25]], [0, 1, 0], 0.2, "",
        prior_anchors_by_cls={0: [[30, 30, 35, 35]], 1: [[40, 40, 45, 45]]})
    check("T57 two classes -> two passes", len(calls) == 2, len(calls))
    check("T57 pass 0 gets only class-0 exemplars",
          calls[0]["exemplars"] == [[0, 0, 5, 5], [20, 20, 25, 25]], calls[0]["exemplars"])
    check("T57 pass 1 gets only class-1 exemplars",
          calls[1]["exemplars"] == [[9, 9, 14, 14]], calls[1]["exemplars"])
    check("T57 pass 0 anchored on class-0 priors only",
          calls[0]["anchors"] == [[30, 30, 35, 35]], calls[0]["anchors"])
    check("T57 pass 1 anchored on class-1 priors only",
          calls[1]["anchors"] == [[40, 40, 45, 45]], calls[1]["anchors"])
    check("T57 detections carry their pass's class", cls == [0, 0, 1], cls)
    check("T57 boxes and classes aligned", len(boxes) == len(cls) == 3, (boxes, cls))
    check("T57 polys and classes aligned", len(polys) == len(cls), (polys, cls))
    check("T57 multi-pass returns no single results object", results is None, results)

    # No drawn exemplars: the prior anchors alone drive one pass per class.
    calls.clear()
    _b, _p, cls, _r = w._run_sam3_boxes_multiclass(
        "/tmp/x.jpg", [], [], 0.2, "",
        prior_anchors_by_cls={0: [[1, 1, 2, 2]], 3: [[5, 5, 6, 6]]})
    check("T57 anchors alone -> one pass per anchor class", len(calls) == 2, len(calls))
    check("T57 anchor-only classes tagged", sorted(set(cls)) == [0, 3], cls)

t57()

# ══════════════════════════════════════════════════════════════════════════
# T58: the SAM3 box branch of _run_detector routes through the multi-class
# helper and publishes per-detection classes on the side channel.
# ══════════════════════════════════════════════════════════════════════════
def t58():
    import types as _types

    w = mk_window()
    w.output_folder = "/tmp"
    w.detector_choice = "SAM3 (one-shot)"
    w.segmenter_choice = "(none)"
    w.prompt_mode = "boxes"
    w.active_class = 1
    prompt_boxes = [[0, 0, 5, 5], [50, 50, 55, 55]]
    w.image_label = _types.SimpleNamespace(
        _orig_w=100, _orig_h=100, annotations=[],
        get_active_annotations=lambda: [{}],
        get_prompt_boxes_with_cls_in_image_coords=lambda: (prompt_boxes, [0, 1]))
    w._boxes_from_seg_polys = lambda polys, path, fallback=None: list(fallback or [])

    seen = {}

    def fake_multiclass(image_path, exemplars, exemplar_cls, conf, text,
                        prior_anchors_by_cls=None):
        seen["cls"] = list(exemplar_cls)
        seen["anchors"] = dict(prior_anchors_by_cls or {})
        return ([[1, 1, 2, 2], [3, 3, 4, 4]],
                [[[0.1, 0.1]], [[0.2, 0.2]]], [0, 1], None)

    w._run_sam3_boxes_multiclass = fake_multiclass
    boxes, _ = w._run_detector("/tmp/x.jpg", "", 0.2, 0.2, prompt_boxes)
    check("T58 exemplar classes read from the drawn boxes", seen["cls"] == [0, 1], seen)
    check("T58 side channel carries per-detection classes",
          w._det_classes_aligned == [0, 1], w._det_classes_aligned)
    check("T58 boxes returned", len(boxes) == 2, boxes)
    check("T58 polys aligned with boxes",
          len(w._oneshot_polys_aligned) == 2, w._oneshot_polys_aligned)

    # All class 0 -> the None single-class fast path, unchanged behavior.
    w._run_sam3_boxes_multiclass = lambda *a, **k: (
        [[1, 1, 2, 2]], [[[0.1, 0.1]]], [0], None)
    w._run_detector("/tmp/x.jpg", "", 0.2, 0.2, prompt_boxes)
    check("T58 all-class-0 -> classes None", w._det_classes_aligned is None,
          w._det_classes_aligned)

t58()

# ══════════════════════════════════════════════════════════════════════════
# T59: _prior_anchors_by_class groups prior detector output by class so a
# class-1 search is never anchored on class-0 objects.
# ══════════════════════════════════════════════════════════════════════════
def t59():
    import types as _types

    w = mk_window()
    anns = [
        {"type": "rect", "data": [0.1, 0.1, 0.1, 0.1], "source": "detector", "cls": 0, "deleted": False},
        {"type": "rect", "data": [0.5, 0.5, 0.1, 0.1], "source": "detector", "cls": 1, "deleted": False},
        {"type": "poly", "data": [[0.7, 0.7], [0.9, 0.7], [0.8, 0.9]], "source": "restored", "cls": 1, "deleted": False},
        {"type": "rect", "data": [0.2, 0.2, 0.1, 0.1], "source": "manual", "cls": 0, "deleted": False},
        {"type": "rect", "data": [0.3, 0.3, 0.1, 0.1], "source": "detector", "cls": 0, "deleted": True},
        {"type": "rect", "data": [0.4, 0.4, 0.1, 0.1], "source": "prompt", "cls": 1, "deleted": False},
    ]
    w.image_label = _types.SimpleNamespace(_orig_w=100, _orig_h=100, annotations=anns)
    got = w._prior_anchors_by_class()
    check("T59 grouped by class", sorted(got) == [0, 1], sorted(got))
    check("T59 one class-0 anchor", len(got[0]) == 1, got[0])
    check("T59 class-1 takes detector + restored", len(got[1]) == 2, got[1])
    check("T59 manual/deleted/prompt anns excluded",
          all(len(v) <= 2 for v in got.values()), got)

    w.image_label = _types.SimpleNamespace(_orig_w=0, _orig_h=0, annotations=anns)
    check("T59 no image size -> empty", w._prior_anchors_by_class() == {},
          w._prior_anchors_by_class())

t59()

# ══════════════════════════════════════════════════════════════════════════
# T60: carry-forward carries the class. _collect_box_prompt_crops records a
# class per crop; _ref_cls_array turns it into YOLOE's visual-prompt cls array;
# _run_sam3_crop_composite composites one class per pass.
# ══════════════════════════════════════════════════════════════════════════
def t60():
    import types as _types
    import numpy as _np

    w = mk_window()
    w.images = ["/tmp/x.jpg"]
    w.current_image_index = 0
    # Second box is degenerate (sub-2px) and must drop its class with it.
    boxes = [[0, 0, 10, 10], [1, 1, 2, 2], [20, 20, 40, 40]]
    w.image_label = _types.SimpleNamespace(
        get_prompt_boxes_with_cls_in_image_coords=lambda: (boxes, [0, 1, 2]))
    w._imread_cached = lambda p: _np.zeros((100, 100, 3), dtype=_np.uint8)
    ref = w._collect_box_prompt_crops()
    check("T60 degenerate crop dropped", len(ref["crops"]) == 2, len(ref["crops"]))
    check("T60 cls aligned with surviving crops", ref["cls"] == [0, 2], ref["cls"])
    check("T60 boxes aligned with cls",
          len(ref["boxes_xyxy"]) == len(ref["cls"]), ref)

    arr = w._ref_cls_array(ref)
    check("T60 ref cls array matches", list(arr) == [0, 2], list(arr))
    check("T60 legacy bundle without cls -> zeros",
          list(w._ref_cls_array({"boxes_xyxy": [[0, 0, 1, 1], [2, 2, 3, 3]]})) == [0, 0],
          list(w._ref_cls_array({"boxes_xyxy": [[0, 0, 1, 1], [2, 2, 3, 3]]})))

    # Crop-composite: one pass per class, each seeing only its own crops.
    passes = []

    def fake_single(image_path, sub_ref, conf):
        passes.append(list(sub_ref["cls"]) if "cls" in sub_ref else None)
        n = len(sub_ref["crops"])
        base = len(passes) * 100
        # Spread the boxes so nothing overlaps: same-class dedup is real and
        # would otherwise collapse identical detections and mask the count.
        return ([[base + i * 10, 0, base + i * 10 + 5, 5] for i in range(n)],
                [[[0.1, 0.1], [0.2, 0.1], [0.15, 0.2]] for _ in range(n)],
                "res")

    w._run_sam3_crop_composite_single = fake_single
    multi = {"image_path": "/tmp/x.jpg", "crops": ["a", "b", "c"],
             "boxes_xyxy": [[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]],
             "cls": [0, 1, 0]}
    b, p, c, r = w._run_sam3_crop_composite("/tmp/x.jpg", multi, 0.2)
    check("T60 crop-composite one pass per class", len(passes) == 2, len(passes))
    check("T60 crop-composite classes tagged", c == [0, 0, 1], c)
    check("T60 crop-composite multi-pass results None", r is None, r)
    check("T60 crop-composite boxes aligned", len(b) == len(c) == len(p), (b, c, p))

    # Single class stays a single pass and returns the raw results object.
    passes.clear()
    single = {"image_path": "/tmp/x.jpg", "crops": ["a", "b"],
              "boxes_xyxy": [[0, 0, 1, 1], [2, 2, 3, 3]], "cls": [3, 3]}
    b, p, c, r = w._run_sam3_crop_composite("/tmp/x.jpg", single, 0.2)
    check("T60 single-class carry -> one pass", len(passes) == 1, len(passes))
    check("T60 single-class carry tagged", c == [3, 3], c)
    check("T60 single-class carry keeps results", r == "res", r)

    # The negative-box matcher consumes the SAME 4-tuple. It silently skipped
    # suppression for a whole batch run when the composite grew its cls return
    # and this caller kept unpacking 3 (ValueError swallowed by the guard).
    w._run_sam3_crop_composite = lambda *a, **k: ([[1, 2, 3, 4]], [None], [0], None)
    got = w._detect_neg_matches("/tmp/x.jpg", single, 0.2, "sam3_det")
    check("T60 neg-box matcher unpacks the 4-tuple composite return",
          got == [[1, 2, 3, 4]], got)

t60()

# ══════════════════════════════════════════════════════════════════════════
# T61: the carry ANCHOR (the frozen exemplar set reused on later images) keeps
# a class per box, and _prompt_box_classes reads it instead of flattening every
# carried box onto the active class.
# ══════════════════════════════════════════════════════════════════════════
def t61():
    import types as _types

    w = mk_window()
    boxes = [[0, 0, 10, 10], [20, 20, 30, 30]]
    w.image_label = _types.SimpleNamespace(
        _orig_w=100, _orig_h=100,
        get_prompt_boxes_with_cls_in_image_coords=lambda: (boxes, [0, 2]))
    w._refresh_and_get_carry_anchor()
    check("T61 anchor freezes classes", w._carry_anchor_cls == [0, 2], w._carry_anchor_cls)
    check("T61 anchor cls list matches anchor length",
          len(w._carry_anchor_cls_list()) == len(w._carry_anchor), w._carry_anchor)

    # Later image: nothing drawn, so classes must come from the anchor.
    w.active_class = 1
    w.image_label = _types.SimpleNamespace(
        _orig_w=100, _orig_h=100,
        get_prompt_boxes_with_cls_in_image_coords=lambda: ([], []))
    carried = w._carry_anchor_boxes_img()
    check("T61 carried boxes recovered", len(carried) == 2, carried)
    check("T61 carried classes survive, not flattened to active",
          w._prompt_box_classes(carried) == [0, 2], w._prompt_box_classes(carried))

    # An anchor with no recorded classes falls back to all-zeros, not a crash.
    w._carry_anchor_cls = []
    check("T61 legacy anchor -> zeros", w._carry_anchor_cls_list() == [0, 0],
          w._carry_anchor_cls_list())

t61()

# ══════════════════════════════════════════════════════════════════════════
# T62: classes.txt must not shrink. _max_box_class_used is bounded by the
# configured class count, not by whichever boxes happen to be drawn right now
# (advancing to an undrawn image used to rewrite classes.txt down to one line).
# ══════════════════════════════════════════════════════════════════════════
def t62():
    import types as _types

    w = mk_window()
    w.prompt_mode = "boxes"
    w.active_class = 0
    w.box_class_names = ["berry", "leaf", "stem"]
    w.image_label = _types.SimpleNamespace(
        get_prompt_boxes_with_cls_in_image_coords=lambda: ([], []))
    check("T62 no drawn boxes -> count from the registry",
          w._max_box_class_used() == 2, w._max_box_class_used())
    check("T62 classes.txt keeps every configured name",
          w._class_names_for_run("") == ["berry", "leaf", "stem"],
          w._class_names_for_run(""))

    # Text mode still names classes from the prompt terms.
    w.prompt_mode = "text"
    check("T62 text mode names from prompt",
          w._class_names_for_run("berry, leaf") == ["berry", "leaf"],
          w._class_names_for_run("berry, leaf"))
    check("T62 empty prompt -> object fallback",
          w._class_names_for_run("") == ["object"], w._class_names_for_run(""))

    # A window built without the registry (headless) still gets a valid name.
    w2 = mk_window()
    w2.prompt_mode = "boxes"
    w2.active_class = 0
    w2.image_label = _types.SimpleNamespace(
        get_prompt_boxes_with_cls_in_image_coords=lambda: ([], []))
    check("T62 missing registry -> default single class",
          w2._class_names_for_run("") == ["class_0"], w2._class_names_for_run(""))

t62()

# ══════════════════════════════════════════════════════════════════════════
# T63: box class names live in the in-process session store: they survive a
# window teardown (back to the main menu and in again) but never an app
# restart, and nothing is written to ~/.autoannotate.
# ══════════════════════════════════════════════════════════════════════════
def t63():
    from autoannotate.gui import session_state
    from autoannotate.palette import MAX_BOX_CLASSES, class_color_rgb

    # Box prompts offer class ids 0..4. A deliberate product limit: each extra
    # class costs one more SAM3 pass per image.
    check("T63 box classes capped at 5", MAX_BOX_CLASSES == 5, MAX_BOX_CLASSES)
    cols = [class_color_rgb(i) for i in range(MAX_BOX_CLASSES)]
    check("T63 every class inside the cap has its own color",
          len(set(cols)) == MAX_BOX_CLASSES, cols)

    session_state.reset()
    try:
        w = mk_window()
        check("T63 fresh session starts from one unnamed class",
              w._load_box_class_names() == ["class_0"], w._load_box_class_names())

        w._save_box_class_names(["berry", "leaf"])
        check("T63 names stored in the session, not on disk",
              session_state.STATE["box_class_names"] == ["berry", "leaf"],
              session_state.STATE["box_class_names"])

        # A second window in the same session (main menu round trip) sees them.
        w2 = mk_window()
        check("T63 names survive a window teardown",
              w2._load_box_class_names() == ["berry", "leaf"],
              w2._load_box_class_names())

        # A store touched by a build with a higher cap must clamp on load, not
        # resurrect classes the dialog can no longer show.
        w2._save_box_class_names([f"n{i}" for i in range(9)])
        check("T63 over-long stored list clamps to the cap",
              len(w2._load_box_class_names()) == MAX_BOX_CLASSES,
              w2._load_box_class_names())

        # A malformed store falls back to the default rather than crashing.
        session_state.STATE["box_class_names"] = ["", "  "]
        check("T63 malformed store -> default",
              w2._load_box_class_names() == ["class_0"], w2._load_box_class_names())

        # An app restart = a fresh store: back to one unnamed class.
        session_state.reset()
        check("T63 app restart resets to one unnamed class",
              mk_window()._load_box_class_names() == ["class_0"],
              session_state.STATE["box_class_names"])

        # Nothing box-class-related touches the disk anymore.
        src = open(os.path.join(_REPO_ROOT, "autoannotate", "gui", "manual_window.py"),
                   encoding="utf-8").read()
        check("T63 box_classes.json is gone from the code",
              "box_classes.json" not in src and "BOX_CLASSES_FILE" not in src,
              "manual_window still references the settings file")
    finally:
        session_state.reset()

t63()

# ══════════════════════════════════════════════════════════════════════════
# T64: the saved review images color each shape by class, and fall back to
# class-0 magenta when no classes are supplied (single-class output unchanged).
# ══════════════════════════════════════════════════════════════════════════
def t64():
    import types as _types
    import numpy as _np
    from autoannotate.palette import class_color_bgr as _bgr

    w = mk_window()
    drawn = []
    saved_cv2 = G.get("cv2")

    class _FakeCv2:
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 0
        IMWRITE_JPEG_QUALITY = 1

        @staticmethod
        def rectangle(img, p1, p2, color, thickness):
            drawn.append(("rect", color))

        @staticmethod
        def polylines(img, pts, closed, color, thickness):
            drawn.append(("poly", color))

    G["cv2"] = _FakeCv2
    try:
        w._imread_cached = lambda p: _np.zeros((100, 100, 3), dtype=_np.uint8)

        drawn.clear()
        w._render_overlay_image("/tmp/x.jpg",
                                boxes=[[0, 0, 5, 5], [6, 6, 9, 9]],
                                box_classes=[0, 1])
        check("T64 boxes colored per class",
              drawn == [("rect", _bgr(0)), ("rect", _bgr(1))], drawn)

        drawn.clear()
        tri = [[0.1, 0.1], [0.3, 0.1], [0.2, 0.3]]
        w._render_overlay_image("/tmp/x.jpg", polys=[tri, tri], poly_classes=[2, 0])
        check("T64 polys colored per class",
              drawn == [("poly", _bgr(2)), ("poly", _bgr(0))], drawn)

        drawn.clear()
        w._render_overlay_image("/tmp/x.jpg", boxes=[[0, 0, 5, 5]], polys=[tri])
        check("T64 no classes -> class 0 magenta",
              drawn == [("poly", (255, 0, 255)), ("rect", (255, 0, 255))], drawn)

        # A class list shorter than the shape list must not raise.
        drawn.clear()
        w._render_overlay_image("/tmp/x.jpg",
                                boxes=[[0, 0, 5, 5], [6, 6, 9, 9]], box_classes=[3])
        check("T64 short class list pads with class 0",
              drawn == [("rect", _bgr(3)), ("rect", _bgr(0))], drawn)
    finally:
        if saved_cv2 is not None:
            G["cv2"] = saved_cv2

t64()

# ══════════════════════════════════════════════════════════════════════════
# T65: _save_split_overlays threads the class list into both renders, and
# save_class_legend_image writes a standalone key BESIDE the review folders
# rather than painting over any labelled image.
# ══════════════════════════════════════════════════════════════════════════
def t65():
    import tempfile as _tf
    import numpy as _np
    from autoannotate.pipeline.overlay import save_class_legend_image

    w = mk_window()
    seen = []
    w._render_overlay_image = lambda path, boxes=None, polys=None, box_classes=None, poly_classes=None: (
        seen.append({"boxes": boxes, "polys": polys,
                     "box_classes": box_classes, "poly_classes": poly_classes}) or "img")
    w._save_annotated_image = lambda path, img, kind: seen[-1].update({"kind": kind})

    tri = [[0.1, 0.1], [0.3, 0.1], [0.2, 0.3]]
    w._save_split_overlays("/tmp/x.jpg", [[0, 0, 5, 5]], [tri], classes=[1])
    check("T65 two renders (boxes + masks)", len(seen) == 2, len(seen))
    check("T65 boxes render gets box_classes",
          seen[0]["box_classes"] == [1] and seen[0]["kind"] == "boxes", seen[0])
    check("T65 masks render gets poly_classes",
          seen[1]["poly_classes"] == [1] and seen[1]["kind"] == "masks", seen[1])

    # No polys -> no masks render, as before.
    seen.clear()
    w._save_split_overlays("/tmp/x.jpg", [[0, 0, 5, 5]], [], classes=[0])
    check("T65 no polys -> boxes render only", len(seen) == 1, len(seen))

    # The legend is a real PNG, written where it cannot be mistaken for data.
    with _tf.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "annotated_SAM3", "class_legend.png")
        # cv2 is stubbed at module import, so drive the real one if present.
        try:
            import cv2 as _real_cv2
        except Exception:
            _real_cv2 = None
        if _real_cv2 is not None and hasattr(_real_cv2, "imwrite"):
            got = save_class_legend_image(["berry", "leaf"], out)
            check("T65 legend written", got == out and os.path.exists(out), got)
            check("T65 legend sits beside boxes/ and masks/, not inside",
                  os.path.basename(os.path.dirname(out)) == "annotated_SAM3", out)
        check("T65 no names -> no legend",
              save_class_legend_image([], out) is None, "empty names")

t65()

# ══════════════════════════════════════════════════════════════════════════
# T66: a two-stage segmenter run keeps the class on every mask. save_masks is
# handed the box classes, which segment_with_boxes guarantees are index-aligned
# with the masks it returns.
# ══════════════════════════════════════════════════════════════════════════
def t66():
    import types as _types
    import numpy as _np

    w = mk_window()
    w.output_folder = "/tmp"
    w.images = ["/tmp/x.jpg"]
    w.current_image_index = 0
    captured = {}
    saved_sm = G.get("save_masks")
    saved_ow = G.get("overlay_with_borders")
    saved_am = G.get("adjust_masks")
    try:
        G["save_masks"] = lambda res, d, p, classes=None: captured.update({"classes": classes})
        G["adjust_masks"] = lambda res: [object(), object()]
        G["overlay_with_borders"] = lambda img, m, color, thickness=2: img

        boxes = [[0, 0, 5, 5], [6, 6, 9, 9]]
        w.image_label = _types.SimpleNamespace(
            get_boxes_with_cls_in_image_coords=lambda: (boxes, [0, 2]))
        w._run_segmenter = lambda p, b: ["results"]
        w._imread_cached = lambda p: _np.zeros((10, 10, 3), dtype=_np.uint8)
        w.show_result_image = lambda img: None
        w._write_class_key = lambda names, folder=None: captured.update({"names": names})
        w._positive_prompt_text = lambda: "berry, stem"
        w.prompt_mode = "text"

        w.run_with_manual_boxes()
        check("T66 manual-box segments keep their classes",
              captured["classes"] == [0, 2], captured.get("classes"))
        check("T66 manual-box run writes the class key",
              captured["names"] == ["berry", "stem"], captured.get("names"))

        # All class 0 -> the None fast path, byte-identical to the old output.
        captured.clear()
        w.image_label = _types.SimpleNamespace(
            get_boxes_with_cls_in_image_coords=lambda: (boxes, [0, 0]))
        w.run_with_manual_boxes()
        check("T66 all class 0 -> classes None", captured["classes"] is None,
              captured.get("classes"))
    finally:
        for k, v in (("save_masks", saved_sm), ("overlay_with_borders", saved_ow),
                     ("adjust_masks", saved_am)):
            if v is not None:
                G[k] = v

t66()

# ══════════════════════════════════════════════════════════════════════════
# T67: Auto Annotate Remaining reports how long the run took. format_duration
# reads in the units a user plans with; the batch method must not shadow the
# module-level `time` with a local import placed after the clock starts.
# ══════════════════════════════════════════════════════════════════════════
def t67():
    import inspect
    format_duration = G["format_duration"]

    for secs, want in [(0, "0.0s"), (3.14, "3.1s"), (59.9, "59.9s"),
                       (60, "1m 00s"), (187, "3m 07s"),
                       (3600, "1h 00m 00s"), (4350, "1h 12m 30s")]:
        check(f"T67 format_duration({secs}) == {want}",
              format_duration(secs) == want, format_duration(secs))
    check("T67 negative elapsed clamped", format_duration(-5) == "0.0s",
          format_duration(-5))

    src = inspect.getsource(ManualWindow.auto_annotate_remaining)
    check("T67 clock started before the loop",
          "run_started = time.perf_counter()" in src, "missing run_started")
    check("T67 no local 'import time' shadowing the clock",
          "import time" not in src, "a local import time raises UnboundLocalError")
    check("T67 elapsed reported in the summary", "Time taken:" in src, "missing")

t67()

# ══════════════════════════════════════════════════════════════════════════
# T68: the side-by-side zoom/pan view. Zoom is inert until Image Resize is
# armed, clamps to [1, 8], never strands a panned image off-screen, and its
# whole state round-trips so the middle-arrow swap can carry it.
# ══════════════════════════════════════════════════════════════════════════
def t68():
    from autoannotate.gui.zoompan import ZoomPanImageView, MIN_ZOOM, MAX_ZOOM

    v = ZoomPanImageView()
    v.resize(400, 300)
    pm = QtGui.QPixmap(800, 600)
    pm.fill(QtGui.QColor("red"))
    v.set_pixmap(pm)
    check("T68 starts at fit", v.view_state()["zoom"] == 1.0, v.view_state())
    check("T68 fit scale is min(w/iw, h/ih)",
          abs(v._get_scale_offset()[0] - 0.5) < 1e-9, v._get_scale_offset())

    # Zoom toward the widget centre keeps the centre fixed -> no pan.
    v.set_resize_mode(True)
    v._zoom_at(2.0, 200, 150)
    st = v.view_state()
    check("T68 zoom applied", abs(st["zoom"] - 2.0) < 1e-9, st)
    check("T68 centre zoom does not pan",
          abs(st["pan_x"]) < 1e-9 and abs(st["pan_y"]) < 1e-9, st)

    # Zoom toward a corner must pan to keep that corner under the cursor.
    v.reset_view()
    v._zoom_at(2.0, 0, 0)
    check("T68 corner zoom pans", v.view_state()["pan_x"] != 0.0, v.view_state())

    v.reset_view()
    check("T68 reset returns to fit",
          v.view_state()["zoom"] == 1.0 and v.view_state()["pan_x"] == 0.0, v.view_state())

    v._zoom = 0.01; v._clamp_view()
    check("T68 cannot zoom out past fit", v._zoom == MIN_ZOOM, v._zoom)
    v._zoom = 999; v._clamp_view()
    check("T68 zoom capped", v._zoom == MAX_ZOOM, v._zoom)
    v._zoom = 1.0; v._pan_x = 500; v._pan_y = -400; v._clamp_view()
    check("T68 pan forced to zero at fit", (v._pan_x, v._pan_y) == (0.0, 0.0),
          (v._pan_x, v._pan_y))
    v._zoom = 2.0; v._pan_x = 10 ** 9; v._clamp_view()
    check("T68 pan clamped, image never stranded", v._pan_x <= 400, v._pan_x)

    # Full state round-trip: what _swap_sides relies on.
    v.set_resize_mode(True)
    state = v.view_state()
    other = ZoomPanImageView(); other.resize(400, 300); other.set_pixmap(pm)
    other.apply_view_state(state)
    check("T68 view state round-trips", other.view_state() == state,
          (other.view_state(), state))
    check("T68 partial state keeps current values",
          (other.apply_view_state({"zoom": 3.0}) or other.view_state()["resize_mode"]) is True,
          other.view_state())

    # No image -> no crash, no transform.
    empty = ZoomPanImageView(); empty.resize(100, 100)
    empty.set_resize_mode(True)
    empty._zoom_at(2.0, 10, 10)
    check("T68 no image -> zoom is a no-op", empty.view_state()["zoom"] == 1.0,
          empty.view_state())

t68()

# ══════════════════════════════════════════════════════════════════════════
# T69: the middle-arrow swap carries each side's zoom / pan with its image,
# and the per-pane Image Resize buttons re-sync to the state that moved
# under them.
# ══════════════════════════════════════════════════════════════════════════
def t69():
    import tempfile as _tf
    from autoannotate.gui.side_by_side import SideBySideWindow

    w = SideBySideWindow.__new__(SideBySideWindow)
    QtWidgets.QWidget.__init__(w)
    w.model = w.processor = None
    w._synth_pixmap = w._gt_pixmap = None
    w.synth_images = w.gt_images = []
    w.pairs = []
    w.current_index = 0
    w.titles = {"synth": "Synthetic Images", "gt": "Ground Truth"}
    w._left, w._right = "gt", "synth"
    w.view_states = {"synth": None, "gt": None}
    w.init_ui()
    w.resize(800, 600)

    with _tf.TemporaryDirectory() as tmp:
        # Real files on disk: _show_current reloads pixmaps from the pair paths,
        # and a missing path legitimately clears the view.
        paths = []
        for i, color in enumerate(("blue", "green", "red", "yellow")):
            pm = QtGui.QPixmap(80, 60)
            pm.fill(QtGui.QColor(color))
            p = os.path.join(tmp, f"img_{i}.png")
            pm.save(p)
            paths.append(p)
        w.synth_images = [paths[0], paths[1]]
        w.gt_images = [paths[2], paths[3]]
        w.pairs = [(paths[0], paths[2]), (paths[1], paths[3])]
        w.current_index = 0
        w._show_current()
        check("T69 both panes have an image",
              w.left_view.has_image() and w.right_view.has_image(), "no image loaded")

        # Zoom the LEFT pane, which is currently the 'gt' side.
        w.left_resize_btn.setChecked(True)
        w.left_view._zoom_at(3.0, 5, 5)
        gt_state = w.left_view.view_state()
        check("T69 left pane zoomed", gt_state["zoom"] > 1.0, gt_state)

        w._swap_sides()
        check("T69 logical sides swapped", (w._left, w._right) == ("synth", "gt"),
              (w._left, w._right))
        check("T69 gt view state followed gt to the right pane",
              w.right_view.view_state() == gt_state,
              (w.right_view.view_state(), gt_state))
        check("T69 synth pane still at fit",
              w.left_view.view_state()["zoom"] == 1.0, w.left_view.view_state())
        check("T69 right Image Resize button re-checked",
              w.right_resize_btn.isChecked(), w.right_resize_btn.text())
        check("T69 right button text follows",
              w.right_resize_btn.text() == "Image Resize: ON", w.right_resize_btn.text())
        check("T69 left button unchecked",
              not w.left_resize_btn.isChecked(), w.left_resize_btn.text())

        # Swapping back restores the original placement.
        w._swap_sides()
        check("T69 swap back restores gt to the left",
              w.left_view.view_state() == gt_state, w.left_view.view_state())

        # Next must not throw away the zoom the user set: stepping through pairs
        # at a zoom is the whole point of zooming.
        z = w.left_view.view_state()["zoom"]
        w.show_next()
        check("T69 zoom survives Next",
              abs(w.left_view.view_state()["zoom"] - z) < 1e-9, w.left_view.view_state())
        check("T69 Next advanced the pair", w.current_index == 1, w.current_index)

        # A missing pair member clears that pane back to the placeholder.
        w.pairs = [(paths[0], None)]
        w.current_index = 0
        w._show_current()
        gt_pane = w.left_view if w._left == "gt" else w.right_view
        check("T69 unmatched side shows no image", not gt_pane.has_image(), "still has image")

        # A new folder starts fresh.
        w._reset_views()
        check("T69 new folder resets both panes to fit",
              w.left_view.view_state()["zoom"] == 1.0
              and w.right_view.view_state()["zoom"] == 1.0,
              (w.left_view.view_state(), w.right_view.view_state()))

    w.close()

t69()

# ══════════════════════════════════════════════════════════════════════════
# T70: image I/O is unicode-safe, and NOTHING calls cv2.imread/imwrite directly.
# On Windows those go through the ANSI API and fail silently on any path outside
# the active code page -- including the default temp dir under a non-ASCII
# username, which the SAM3 crop-composite carry writes to.
# ══════════════════════════════════════════════════════════════════════════
def t70():
    import re as _re

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg = os.path.join(repo, "autoannotate")
    offenders = []
    for root, dirs, files in os.walk(pkg):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in files:
            if not fn.endswith(".py") or fn == "imageio.py":
                continue
            path = os.path.join(root, fn)
            with open(path, encoding="utf-8") as fh:
                for i, line in enumerate(fh, 1):
                    if _re.search(r"\bcv2\.(imread|imwrite)\s*\(", line):
                        offenders.append(f"{os.path.relpath(path, repo)}:{i}")
    check("T70 no raw cv2.imread/imwrite outside imageio.py",
          not offenders, offenders)

    # The real helpers (not the stubs) must round-trip a non-ASCII path. cv2 is
    # stubbed in THIS process, so the probe runs in a clean interpreter against
    # the genuine module. Skipped when real cv2 is not installed.
    import subprocess as _sp
    probe = (
        "import os, sys, tempfile, numpy as np\n"
        f"sys.path.insert(0, {repo!r})\n"
        "from autoannotate.imageio import imread_unicode, imwrite_unicode\n"
        "with tempfile.TemporaryDirectory() as t:\n"
        "    p = os.path.join(t, 'b\\u00e4i\\u5b57', 'x.png')\n"
        "    w = imwrite_unicode(p, np.zeros((4,4,3), dtype=np.uint8))\n"
        "    r = imread_unicode(p) is not None\n"
        "    bare = imwrite_unicode(os.path.join(t,'noext'), np.zeros((4,4,3), dtype=np.uint8))\n"
        "    nofile = imread_unicode(os.path.join(t,'missing.png'))\n"
        "print(int(w), int(r), int(bare), int(nofile is None))\n"
    )
    res = _sp.run([sys.executable, "-c", probe], capture_output=True, text=True)
    out = (res.stdout or "").strip().splitlines()
    got = out[-1] if out else ""
    if res.returncode != 0 and "ModuleNotFoundError" in (res.stderr or ""):
        print("SKIP - T70 unicode round-trip (real cv2 not installed)")
        return
    check("T70 non-ASCII write+read round-trips", got.startswith("1 1"),
          got or (res.stderr or "")[-300:])
    check("T70 extension-less write refused, missing file -> None",
          got.endswith("0 1"), got or (res.stderr or "")[-300:])

t70()

# ══════════════════════════════════════════════════════════════════════════
# T71: every third-party package the code imports is declared in BOTH install
# files. Catches the classic "added a dependency, forgot requirements" break,
# which shows up only on the other machine.
# ══════════════════════════════════════════════════════════════════════════
def t71():
    import ast as _ast
    import re as _re

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg = os.path.join(repo, "autoannotate")

    # import name -> the distribution that provides it
    DIST = {"PIL": "pillow", "cv2": "opencv-python", "dotenv": "python-dotenv",
            "huggingface_hub": "huggingface_hub", "PyQt5": "pyqt5",
            "torch": "torch", "numpy": "numpy", "shapely": "shapely",
            "ultralytics": "ultralytics", "transformers": "transformers",
            "diffusers": "diffusers"}
    # Vendored (installed from source, deliberately absent from requirements).
    VENDORED = {"groundingdino"}
    stdlib = set(sys.stdlib_module_names)

    imported = set()
    for root, dirs, files in os.walk(pkg):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            with open(os.path.join(root, fn), encoding="utf-8") as fh:
                tree = _ast.parse(fh.read(), fn)
            for n in _ast.walk(tree):
                if isinstance(n, _ast.Import):
                    imported.update(a.name.split(".")[0] for a in n.names)
                elif isinstance(n, _ast.ImportFrom) and n.level == 0 and n.module:
                    imported.add(n.module.split(".")[0])
    third = {m for m in imported
             if m not in stdlib and m != "autoannotate" and m not in VENDORED}

    unmapped = sorted(m for m in third if m not in DIST)
    check("T71 every third-party import has a known distribution", not unmapped,
          unmapped)

    def declared(path):
        names = set()
        for line in open(path, encoding="utf-8"):
            line = line.split("#")[0].strip()
            if not line or line.startswith("-"):
                continue
            m = _re.match(r"^([A-Za-z0-9_.\-]+)", line)
            if m:
                names.add(m.group(1).lower().replace("_", "-"))
        return names

    for fname in ("requirements.txt", "requirements-windows.txt", "requirements-macos.lock"):
        path = os.path.join(repo, fname)
        if not os.path.exists(path):
            check(f"T71 {fname} exists", False, "missing")
            continue
        have = declared(path)
        missing = sorted(DIST[m] for m in third
                         if m in DIST and DIST[m].lower().replace("_", "-") not in have)
        check(f"T71 {fname} declares every imported package", not missing, missing)

t71()

# ══════════════════════════════════════════════════════════════════════════
# T72: classes.txt stays a plain YOLO name list (importers read the WHOLE line
# as the class name, so a trailing colour would rename every class). The colour
# key lives in the sibling class_colors.txt.
# ══════════════════════════════════════════════════════════════════════════
def t72():
    import tempfile as _tf

    save_classes_txt = G["save_classes_txt"]
    save_class_colors_txt = G["save_class_colors_txt"]
    from autoannotate.palette import class_color_image_rgb, rgb_to_hex

    with _tf.TemporaryDirectory() as d:
        save_classes_txt(["berry", "leaf"], d)
        lines = open(os.path.join(d, "classes.txt")).read().splitlines()
        check("T72 classes.txt is one bare name per line", lines == ["berry", "leaf"], lines)
        check("T72 classes.txt has no colour on any line",
              all("#" not in ln for ln in lines), lines)
        check("T72 classes.txt line index is the class id",
              lines.index("leaf") == 1, lines)

        path = save_class_colors_txt(["berry", "leaf"], d)
        check("T72 class_colors.txt written beside classes.txt",
              path == os.path.join(d, "class_colors.txt") and os.path.exists(path), path)
        text = open(path, encoding="utf-8").read()
        body = [ln for ln in text.splitlines() if ln and not ln.lstrip().startswith("#")]
        check("T72 one colour row per class", len(body) == 2, body)
        check("T72 row carries id, name, colour, hex and rgb",
              body[0].split() == ["0", "berry", "magenta", "#FF00FF", "255,0,255"],
              body[0].split())
        check("T72 class 1 row correct",
              body[1].split() == ["1", "leaf", "orange", "#FF8C00", "255,140,0"],
              body[1].split())

        # The hex must be the colour actually drawn into the review images, not
        # the canvas magenta. Those two differ for class 0 alone.
        check("T72 hex matches the saved-image colour",
              rgb_to_hex(class_color_image_rgb(0)) == "#FF00FF",
              rgb_to_hex(class_color_image_rgb(0)))

        # The file is JUST the table: a single commented column-header line
        # plus one row per class. The old explanatory comment block is gone on
        # purpose (user request) and must not creep back.
        all_lines = [ln for ln in text.splitlines() if ln.strip()]
        check("T72 file is only the header line plus the rows",
              len(all_lines) == 3, all_lines)
        check("T72 header line names the columns",
              all_lines[0].split() == ["#", "id", "name", "colour", "hex", "rgb"],
              all_lines[0].split())

        check("T72 empty names -> no file", save_class_colors_txt([], d) is None, "wrote one")

t72()

# ══════════════════════════════════════════════════════════════════════════
# T73: side-by-side viewer basics after the tint removal. The view transform
# round-trips across a swap, old saved states that still carry a dark_tint
# flag are tolerated, no tint machinery remains in the VIEWER (the annotation
# window keeps its geometry-based tint on purpose), and the in-GUI baked box
# overlay keeps taking RGB class colours (PIL draws in RGB space).
# ══════════════════════════════════════════════════════════════════════════
def t73():
    import re as _re
    from autoannotate.gui.zoompan import ZoomPanImageView

    v = ZoomPanImageView()
    v.resize(400, 300)
    pm = QtGui.QPixmap(40, 30)
    pm.fill(QtGui.QColor("blue"))
    v.set_pixmap(pm)
    v._zoom = 2.0
    v._pan_x = 5.0
    v.set_resize_mode(True)
    state = v.view_state()
    check("T73 view_state carries zoom/pan/resize only",
          sorted(state) == ["pan_x", "pan_y", "resize_mode", "zoom"], state)

    v2 = ZoomPanImageView()
    v2.resize(400, 300)
    v2.set_pixmap(pm)
    v2.apply_view_state(state)
    check("T73 view state round-trips across widgets",
          v2._zoom == 2.0 and v2._resize_mode, v2.view_state())
    # A state saved by the OLD build still carries dark_tint; it must be
    # ignored, not crash the restore.
    v2.apply_view_state({"zoom": 1.5, "dark_tint": True})
    check("T73 stale dark_tint key tolerated", v2._zoom == 1.5, v2._zoom)
    check("T73 no-image widget takes set_pixmap(None)",
          ZoomPanImageView().set_pixmap(None) is None, "raised")

    # The tint feature is REMOVED from the side-by-side viewer (user request:
    # burned-in annotations made pixel recovery unreliable on leafy photos).
    for mod in ("zoompan", "side_by_side"):
        src_mod = open(os.path.join(_REPO_ROOT, "autoannotate", "gui", mod + ".py"),
                       encoding="utf-8").read()
        check(f"T73 {mod}.py carries no tint machinery",
              ("set_dark_tint" not in src_mod and "_tint_pixmap" not in src_mod
               and "Darken Tint" not in src_mod), mod)
    # ...while the ANNOTATION window keeps its tint: it has real geometry.
    canvas_src = open(os.path.join(_REPO_ROOT, "autoannotate", "gui", "canvas.py"),
                      encoding="utf-8").read()
    check("T73 annotation canvas keeps its geometry-based tint",
          "set_dark_tint" in canvas_src, "canvas tint removed")

    # The in-GUI baked box overlay draws through PIL in RGB space; feeding it
    # the BGR form flips class 1 orange into blue on screen. Guard the call
    # sites the way T70 guards raw cv2.imread.
    mw_src = open(os.path.join(_REPO_ROOT, "autoannotate", "gui", "manual_window.py"),
                  encoding="utf-8").read()
    bgr_fed = 0
    for hit in _re.finditer(r"draw_boxes_on_image\(", mw_src):
        colors_block = mw_src[max(0, hit.start() - 400):hit.start()]
        colors_block = colors_block.split("colors = [")[-1]
        if "class_color_bgr" in colors_block:
            bgr_fed += 1
    check("T73 baked GUI boxes take RGB class colours (PIL draws in RGB)",
          bgr_fed == 0, f"{bgr_fed} call sites pass class_color_bgr")

t73()

# ══════════════════════════════════════════════════════════════════════════
# T74: per-class thresholds. With 2+ classes each class overrides the three
# sliders (det conf / seg conf / max area). Single-pass detectors run at the
# LOOSEST class value and re-filter each detection against its own class;
# SAM3 box passes take each class's own confidence directly. A moved global
# slider gates every run until applied to all classes or reverted. One class
# = all of it degrades to the plain slider values.
# ══════════════════════════════════════════════════════════════════════════
def t74():
    import tempfile as _tf
    from PIL import Image as _PILImage
    from autoannotate.gui import session_state

    session_state.reset()
    try:
        w = mk_window()

        # Single class: inactive, every helper degrades to the plain value.
        check("T74 single class -> per-class inactive", not w._per_class_active(),
              w._active_class_ids())
        check("T74 det floor == global with one class",
              w._det_thresh_floor(0.5) == 0.5, w._det_thresh_floor(0.5))
        check("T74 headless settings refresh is a no-op",
              w._refresh_class_settings_ui() is None, "raised")

        # Two box classes with different overrides.
        w.prompt_mode = "boxes"
        w.box_class_names = ["berry", "leaf"]
        w._save_box_class_names(w.box_class_names)
        session_state.STATE["class_settings"] = {
            0: {"det": 0.30, "max_area": 0.10},
            1: {"det": 0.60, "max_area": 0.90},
        }
        check("T74 two classes -> active", w._per_class_active(),
              w._active_class_ids())
        check("T74 per-class det honored",
              w._class_det_thresh(0, 0.5) == 0.30 and w._class_det_thresh(1, 0.5) == 0.60,
              (w._class_det_thresh(0, 0.5), w._class_det_thresh(1, 0.5)))
        check("T74 unset key falls back to the global value",
              w._class_seg_thresh(0, 0.3) == 0.3, w._class_seg_thresh(0, 0.3))
        check("T74 det floor is the loosest class",
              w._det_thresh_floor(0.5) == 0.30, w._det_thresh_floor(0.5))
        check("T74 max-area loosest is the biggest cap",
              abs(w._max_area_frac_loosest() - 0.90) < 1e-9,
              w._max_area_frac_loosest())

        # SAM3 multi-class: each class's pass runs at ITS confidence.
        seen = []
        def fake_part(image_path, ex, conf, text_prompt, supplementary_exemplars=None):
            seen.append(round(conf, 4))
            return ([[len(seen) * 100, 0, len(seen) * 100 + 5, 5]],
                    [[[0.1, 0.1], [0.2, 0.1], [0.15, 0.2]]], "r")
        w._run_sam3_boxes_partitioned = fake_part
        w._run_sam3_boxes_multiclass("/tmp/x.jpg", [[0, 0, 1, 1], [2, 2, 3, 3]],
                                     [0, 1], 0.5, "")
        check("T74 SAM3 passes use per-class confidences", seen == [0.30, 0.60], seen)

        # Exact per-class area cut: same-size boxes live or die by their class.
        with _tf.TemporaryDirectory() as tmp:
            img_path = os.path.join(tmp, "a.png")
            _PILImage.new("RGB", (100, 100)).save(img_path)
            boxes = [[0, 0, 20, 20],    # cls 0: 4%  < 10% cap -> kept
                     [0, 0, 40, 40],    # cls 0: 16% >= 10% cap -> dropped
                     [10, 10, 50, 50]]  # cls 1: 16% < 90% cap -> kept
            kb, kp, kc = w._cut_boxes_by_class_area(
                img_path, boxes, [None, None, None], [0, 0, 1])
            check("T74 per-class area cut keeps by each class's cap",
                  kb == [[0, 0, 20, 20], [10, 10, 50, 50]] and kc == [0, 1],
                  (kb, kc))
            check("T74 area cut keeps polys aligned", len(kp) == len(kb), (kb, kp))

        # Gate flow: a moved global slider blocks runs until reverted/applied.
        check("T74 clean state does not gate", not w._global_sliders_blocked(),
              getattr(w, "_global_dirty", None))
        w.global_apply_row = QtWidgets.QWidget()
        w._on_global_slider_moved()
        check("T74 moved global slider gates the run", w._global_sliders_blocked(),
              getattr(w, "_global_dirty", None))
        w._revert_global_sliders()
        check("T74 revert clears the gate", not w._global_sliders_blocked(),
              getattr(w, "_global_dirty", None))

        # Back to one class: the gate can never engage.
        w.box_class_names = ["class_0"]
        session_state.STATE["box_class_names"] = None
        w._on_global_slider_moved()
        check("T74 single class never gates", not w._global_sliders_blocked(),
              getattr(w, "_global_dirty", None))
    finally:
        session_state.reset()

t74()

# ══════════════════════════════════════════════════════════════════════════
# T75: input schemes + drawing on the zoomed view. Scroll pans (per the
# Trackpad/Mouse scheme) and Ctrl/Cmd+scroll zooms, so the left button draws
# and edits even while Image Resize is on; every coordinate routes through
# the zoom/pan transform, so a zoomed draw lands where it looks.
# ══════════════════════════════════════════════════════════════════════════
def t75():
    import sys as _sys
    from autoannotate.gui import session_state
    from autoannotate.gui.session_state import classify_wheel, input_scheme

    session_state.reset()
    try:
        expected = "trackpad" if _sys.platform == "darwin" else "mouse"
        check("T75 platform default scheme", input_scheme() == expected,
              (input_scheme(), _sys.platform))
        session_state.STATE["input_scheme"] = "mouse"
        check("T75 explicit scheme wins", input_scheme() == "mouse", input_scheme())

        check("T75 plain scroll pans", classify_wheel("mouse", 0, -40) == ("pan", 0, -40),
              classify_wheel("mouse", 0, -40))
        check("T75 ctrl+scroll zooms in both schemes",
              classify_wheel("mouse", 0, 30, ctrl=True) == ("zoom", 30)
              and classify_wheel("trackpad", 0, 30, ctrl=True) == ("zoom", 30),
              "mismatch")
        check("T75 mouse shift+wheel pans sideways",
              classify_wheel("mouse", 0, 25, shift=True) == ("pan", 25, 0),
              classify_wheel("mouse", 0, 25, shift=True))
        check("T75 trackpad supplies both axes itself",
              classify_wheel("trackpad", 12, -7, shift=True) == ("pan", 12, -7),
              classify_wheel("trackpad", 12, -7, shift=True))
        check("T75 no delta -> no action", classify_wheel("mouse", 0, 0) is None,
              classify_wheel("mouse", 0, 0))

        # Drawing while zoomed: the widget-to-image mapping honors zoom + pan,
        # and a left press in resize mode starts a DRAW drag, not a pan.
        c = AnnotationCanvas()
        c.resize(400, 300)
        c._orig_w, c._orig_h = 400, 300
        c.set_resize_mode(True)
        c._zoom = 2.0
        c._pan_x = -100.0
        c._pan_y = -50.0
        # scale = min(400/400, 300/300) * 2 = 2; offsets: (400-800)/2-100=-300,
        # (300-600)/2-50=-200. Widget (100, 100) -> image (200, 150).
        p = c._widget_point_to_image(QtCore.QPoint(100, 100))
        check("T75 zoomed draw lands where it looks",
              p is not None and abs(p[0] - 200) < 1e-6 and abs(p[1] - 150) < 1e-6, p)

        c.draw_mode = True
        ev = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonPress,
                               QtCore.QPointF(100, 100), QtCore.Qt.LeftButton,
                               QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
        c.mousePressEvent(ev)
        check("T75 left press in resize mode starts a draw drag",
              c._drag_start is not None, c._drag_start)

        # The old left-drag pan is gone from the canvas (pan lives on the
        # wheel now); the side-by-side panes keep it, having no drawing.
        canvas_src = open(os.path.join(_REPO_ROOT, "autoannotate", "gui", "canvas.py"),
                          encoding="utf-8").read()
        check("T75 canvas has no left-drag pan left", "_pan_last" not in canvas_src,
              "found _pan_last")
        zp_src = open(os.path.join(_REPO_ROOT, "autoannotate", "gui", "zoompan.py"),
                      encoding="utf-8").read()
        check("T75 side-by-side keeps its left-drag pan", "_pan_last" in zp_src,
              "missing _pan_last")
    finally:
        session_state.reset()

t75()

# ── summary ────────────────────────────────────────────────────────────────
passed = sum(1 for _, ok, _ in _results if ok)
total = len(_results)
print(f"\n==== {passed}/{total} checks passed ====")
sys.exit(0 if passed == total else 1)
