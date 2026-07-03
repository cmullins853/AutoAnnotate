"""Shared button/toggle/slider styling and the color legend for the GUI."""
from PyQt5 import QtGui, QtWidgets

# Shared button styling, one consistent look across every screen.
# Resting color carries meaning; hover brightens, press darkens, and the
# disabled / "already-pressed" state is a flat grey everywhere.
#   GREEN  safe anytime          BLUE   pick / configure (instant)
#   ORANGE takes a little time   RED    longest / batch operations
#   PURPLE synthetic images      GREY   neutral nav / mode toggles
BTN_GREEN  = "#1a7a1a"
BTN_BLUE   = "#4f82ff"
BTN_ORANGE = "#ff8c00"
BTN_RED    = "#c00000"
BTN_PURPLE = "#5a3aa0"
BTN_GREY   = "#555555"
BTN_DISABLED_BG = "#3a3a3a"
BTN_DISABLED_FG = "#888888"
BTN_GAP = 8          # standard gap (px) between buttons in a row/column

# Toggle "ON" accents (resting is grey via toggle_qss).
TGL_DRAW_ON  = "#ff8c00"
TGL_EDIT_ON  = "#8b30d0"
TGL_MULTI_ON = "#0a7a8a"

# Domain-agnostic Stable Diffusion defaults (edit per scene in the popup).
_SD_DEFAULT_PROMPT = ("a natural outdoor scene, photorealistic, sharp focus, "
                      "natural daylight, depth of field, high detail")
_SD_DEFAULT_NEG = ("text, letters, numbers, watermark, signature, illustration, "
                   "cartoon, drawing, painting, render, cgi, diagram, chart, "
                   "blurry, low quality, low resolution, jpeg artifacts, distorted, "
                   "deformed, disfigured, extra limbs, bad anatomy, duplicate, "
                   "cropped, out of frame, oversaturated, grainy, noise")

# Hover instructions for the SAM drawing/editing tools (shown on the Draw button
# and the edit menu item; replaces the old on-canvas banner).
TOOLTIP_BOX      = ("Draw rectangles. For box-capable detectors they become yellow "
                    "prompt boxes; otherwise they save as green manual annotations.")
TOOLTIP_AUTODRAW = ("Semi-Automatic Point Segmentation. Left-click an object and SAM "
                    "masks it for you. Click again outside the mask to grow it or pull in "
                    "a neighbour; click inside the mask to cut a piece out (a red negative "
                    "point). Right-click or Backspace removes your last point. Press Enter "
                    "to keep the mask, or Esc to start over. Needs a SAM2 or SAM3 "
                    "segmenter (or SAM3 one-shot), and it switches to the Segmentation "
                    "view for you.")
TOOLTIP_SEMIAUTO = ("Manually Draw Masks, like a curve tool. Left-click points around "
                    "the object one after another; the outline grows but nothing is "
                    "masked yet. The first point is amber: click it again (or "
                    "double-click, or press Enter) to close the outline. SAM then fills "
                    "in the shape, kept strictly inside what you drew. Right-click or "
                    "Backspace removes a point, Esc cancels. Needs a SAM2 or SAM3 "
                    "segmenter (or SAM3 one-shot), and it switches to the Segmentation "
                    "view for you.")
TOOLTIP_SEMIAUTO_EDIT = ("Edit any committed mask, model-generated or hand-drawn. "
                         "Click a mask to select it. "
                         "In Points mode, left-click adds a SAM point, right-click "
                         "removes one, and dragging a point re-runs SAM live. In Vertices "
                         "mode, drag a vertex to move it, click the outline to add a "
                         "vertex, and right-click a vertex to remove it. The X badge or "
                         "Delete removes the whole mask. Press S for settings (Points vs "
                         "Vertices, class id, simplify). Enter saves your changes, Esc "
                         "deselects.")


def _shade(hex_color, factor):
    """Lighten (factor>1) or darken (factor<1) a hex color via its HSV value."""
    c = QtGui.QColor(hex_color)
    h, s, v, a = c.getHsv()
    v = max(0, min(255, int(v * factor)))
    c.setHsv(h, s, v, a)
    return c.name()


def btn_qss(base, font_px=None, radius=6):
    """Consistent QPushButton stylesheet: rounded, hover-brighten,
    press-darken, flat-grey when disabled (the 'already pressed' state)."""
    fs = f"font-size: {font_px}px; " if font_px else ""
    return (
        f"QPushButton {{ background-color: {base}; color: white; {fs}"
        f"border: none; border-radius: {radius}px; padding: 6px 14px; }}"
        f"QPushButton:hover {{ background-color: {_shade(base, 1.18)}; }}"
        f"QPushButton:pressed {{ background-color: {BTN_DISABLED_BG}; }}"
        f"QPushButton:disabled {{ background-color: {BTN_DISABLED_BG}; "
        f"color: {BTN_DISABLED_FG}; }}"
    )


def toggle_qss(on_color, font_px=None, off_color=BTN_GREY, radius=6):
    """Checkable QPushButton: grey when OFF, accent when ON (:checked).
    Handlers only flip the label text; the color follows the check state."""
    fs = f"font-size: {font_px}px; " if font_px else ""
    return (
        f"QPushButton {{ background-color: {off_color}; color: white; {fs}"
        f"border: none; border-radius: {radius}px; padding: 6px 14px; }}"
        f"QPushButton:hover {{ background-color: {_shade(off_color, 1.18)}; }}"
        f"QPushButton:checked {{ background-color: {on_color}; }}"
        f"QPushButton:checked:hover {{ background-color: {_shade(on_color, 1.12)}; }}"
        f"QPushButton:disabled {{ background-color: {BTN_DISABLED_BG}; "
        f"color: {BTN_DISABLED_FG}; }}"
    )


def tool_toggle_qss(on_color, font_px, off_color=BTN_GREY, radius=6):
    """Same idea for the Edit-Boxes QToolButton (carries a dropdown menu)."""
    return (
        f"QToolButton {{ background-color: {off_color}; color: white; "
        f"font-size: {font_px}px; border: none; border-radius: {radius}px; "
        f"padding: 4px 10px; padding-right: 32px; }} "
        f"QToolButton:hover {{ background-color: {_shade(off_color, 1.18)}; }} "
        f"QToolButton:checked {{ background-color: {on_color}; }} "
        f"QToolButton:disabled {{ background-color: {BTN_DISABLED_BG}; color: {BTN_DISABLED_FG}; }} "
        f"QToolButton::menu-button {{ background: transparent; "
        f"border-left: 1px solid #999; width: 24px; }} "
        f"QToolButton::menu-arrow {{ width: 12px; height: 12px; }}"
    )


def slider_qss():
    """Consistent horizontal QSlider look (light groove fill + round handle).
    Forces QSS rendering -- the native macOS slider drag-bubble can render
    duplicated or off-position when the widget is height-pinned."""
    return (
        "QSlider::groove:horizontal { border: 1px solid #555; height: 6px; "
        "background: #2a2a2a; border-radius: 3px; }"
        "QSlider::sub-page:horizontal { background: #cfcfcf; border: 1px solid "
        "#cfcfcf; height: 6px; border-radius: 3px; }"
        "QSlider::handle:horizontal { background: #f0f0f0; border: 1px solid "
        "#888; width: 14px; margin: -6px 0; border-radius: 8px; }"
        "QSlider::handle:horizontal:hover { background: #ffffff; }"
        "QSlider::handle:horizontal:pressed { background: #ffffff; }"
    )


def lock_during(button, fn):
    """One-shot / nav buttons: grey the button out for the duration of its
    action (disable -> repaint so the grey 'pressed' state shows -> run ->
    wake back up when it finishes). Prevents rapid double-presses."""
    was_enabled = button.isEnabled()
    button.setEnabled(False)
    QtWidgets.QApplication.processEvents()
    try:
        return fn()
    finally:
        if was_enabled:
            button.setEnabled(True)
