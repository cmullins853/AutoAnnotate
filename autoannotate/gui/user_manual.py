"""In-app user manual: an accordion of topics, reachable from the main menu and
from the annotation window.

Everything the app knew about itself used to live in tooltips, which you can only
find by hovering the exact control you already understand. That is fine as a
reminder and useless as an introduction, and it left the one combination that
matters most (the switches Auto Annotate Remaining needs before carried boxes do
anything) written down nowhere at all.

The prose here is deliberately harvested from those tooltips rather than written
fresh: they are accurate, already in the project's voice, and staying close to
them keeps the manual and the controls saying the same thing.
"""
from PyQt5 import QtCore, QtWidgets

from ..config import AUTOANNOTATE_DEBUG
from .style import BTN_BLUE, BTN_GAP, BTN_GREY, btn_qss


def _debug_window_state(tag, widget=None):
    """Dump the window/focus picture, under AUTOANNOTATE_DEBUG only.

    Here because the manual has now been reported three ways (opening as a macOS
    window tab, flickering, and switching the display away to an empty Space) and
    two fixes were made on inference rather than measurement. Any further report
    should be answered with this output, not another hypothesis. Prints every
    top-level widget, because a stray one is what a Space switch usually means.
    """
    if not AUTOANNOTATE_DEBUG:
        return
    try:
        app = QtWidgets.QApplication.instance()
        tops = app.topLevelWidgets() if app else []
        print(f"[manual] {tag}")
        for w in tops:
            print(f"[manual]   top-level {type(w).__name__}: "
                  f"visible={w.isVisible()} active={w.isActiveWindow()} "
                  f"fullscreen={w.isFullScreen()} native={w.testAttribute(QtCore.Qt.WA_NativeWindow)}")
        act = app.activeWindow() if app else None
        foc = app.focusWidget() if app else None
        print(f"[manual]   activeWindow={type(act).__name__ if act else None} "
              f"focusWidget={type(foc).__name__ if foc else None}")
        if widget is not None:
            print(f"[manual]   overlay isWindow={widget.isWindow()} "
                  f"visible={widget.isVisible()} geom={widget.geometry()}")
    except Exception as e:
        print(f"[manual] state dump failed: {e}")

# Collapsed / expanded markers, matching the three collapsible sections already
# in the annotation window (Prompts, Sliders, Synthetic Images).
_ARROW_CLOSED = "▸"
_ARROW_OPEN   = "▾"


def _p(*paragraphs):
    """Join paragraphs with a blank line. Section bodies are rich text rendered
    by a QLabel, so paragraph breaks are markup rather than newlines."""
    return "<br><br>".join(paragraphs)


def _steps(*items):
    """A numbered list. QLabel's rich text subset supports <ol>, but not its
    margins, so the spacing is carried by the surrounding paragraphs."""
    return "<ol>" + "".join(f"<li>{i}</li>" for i in items) + "</ol>"


def _bullets(*items):
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


# (title, html body). Plain data so the headless suite can assert on the wording
# without building a widget.
MANUAL_SECTIONS = [

    ("Getting started (do these in order)", _p(
        "The manual annotation window runs one image at a time: you set up a "
        "prompt, run the model, correct what it got wrong, then move on. Auto "
        "Annotate Remaining repeats your setup across the rest of the folder.",
        _steps(
            "<b>Select Image Folder</b> loads a folder of .png / .jpg / .jpeg "
            "images. The counter above the image shows where you are.",
            "<b>Select Output Folder</b> chooses where labels, segments and "
            "annotated review images are written. You can skip this; Auto "
            "Annotate Remaining will ask for it when you press it.",
            "Pick a <b>Detector</b>. DINO is text-prompted, YOLOE-vis is "
            "box-prompted, YOLOE-seg and SAM3 accept either. Greyed-out entries "
            "are missing their weights on disk.",
            "Pick a <b>Segmenter</b> if the detector needs one. DINO always "
            "does. One-shot detectors (YOLOE-seg, SAM3) make their own masks and "
            "should be left on <b>(none)</b>.",
            "Choose <b>Text</b> or <b>Boxes</b> prompt mode. The radios grey "
            "themselves out to whatever the detector supports.",
            "Type a prompt, or draw a prompt box on the image.",
            "Tick <b>Bounding Box</b> or <b>Segmentation</b> to choose what you "
            "are looking at and what gets saved.",
            "Press <b>Regenerate</b> (or Enter) to run the model on this image.",
            "Fix anything wrong with the edit tools, then <b>Save &amp; "
            "Confirm</b>, or press <b>Next IMG</b> which saves and advances.",
        ),
        "Once one image is set up the way you want it, <b>Auto Annotate "
        "Remaining</b> applies the same setup to the rest of the folder.")),

    ("Text prompts", _p(
        "Type what you want to detect. One concept per field: put 'person' in "
        "one field and 'car' in the next by pressing <b>+ Add prompt</b>. Commas "
        "inside a single field also add classes, so a field and a comma do the "
        "same job; the fields just keep things readable.",
        "Class ids in the saved label files follow the order the classes appear, "
        "reading fields top to bottom and left to right within a field. Hover "
        "the <b>i</b> badge in the Prompts panel to see which outline colour "
        "each class draws in.",
        "<b>Negative classes</b> are things to rule out, for example 'leaf' "
        "while you are detecting 'blueberry'. They are found in the same pass "
        "and any detection overlapping them is dropped. They are optional and "
        "never stop the model from running.",
        "You can add up to five positive and five negative fields; the Add "
        "buttons grey out at the cap. Everything you type stays as you move "
        "between images and is reused by Auto Annotate Remaining. Collapsing the "
        "<b>Prompts</b> header only hides the fields, it does not clear them.")),

    ("Box annotation", _p(
        "Turn on <b>Draw Boxes</b> and drag on the image. Which kind of box you "
        "get depends on the mode and the detector, and the colour tells you "
        "which:",
        _bullets(
            "<b>Yellow, prompt</b>. Input to the model, in Boxes mode with a "
            "box-capable detector. Never saved as a label.",
            "<b>Green, manual</b>. Your own annotation. Saved, and it always "
            "wins over a model detection covering the same object.",
            "<b>Red, negative</b>. Turn on <b>Draw Negative Box</b> first. Its "
            "appearance is suppressed across every image in the folder, so "
            "look-alikes get dropped. Never saved.",
            "<b>Magenta</b>. Model output. <b>Cyan</b>. Selected or preview.",
        ),
        "This distinction is the one that catches people out: a box you draw in "
        "<b>Text</b> mode is a green manual annotation, not a yellow prompt, and "
        "carrying prompts forward will not see it. Switch to Boxes mode and the "
        "boxes you already drew are re-tagged for you.",
        "<b>Classes…</b> sets how many kinds of box you draw and names each "
        "one. Names are saved into class_colors.txt and class_legend.png beside "
        "your labels. Note that SAM3 searches for one class at a time, so every "
        "extra class costs another SAM3 pass per image.",
        "The Draw dropdown also holds the two SAM-assisted tools, "
        "<b>Semi-Automatic Point Segmentation</b> and <b>Manually Draw "
        "Masks</b>. Both need a SAM2 or SAM3 segmenter and both switch you to "
        "the Segmentation view.")),

    ("Use First Image as Prompt", _p(
        "This carries the boxes you drew on the current image to every other "
        "image in the run, so one set of examples drives the whole folder. It is "
        "the setting most likely to hand you a folder of empty labels if one of "
        "its requirements is missed, so check all five:",
        _steps(
            "The <b>Detector</b> is YOLOE-vis, YOLOE-seg or SAM3. DINO is "
            "text-only; its typed prompt already applies everywhere, so the "
            "toggle greys itself out.",
            "The prompt mode is <b>Boxes</b>. YOLOE-seg and SAM3 in Text mode "
            "ignore carried boxes completely, and the toggle still reads ON "
            "while doing nothing.",
            "For SAM3 one-shot, the <b>Segmenter</b> is <b>(none)</b>. Any other "
            "segmenter with SAM3 selected returns nothing at all, on every "
            "image, including a plain Regenerate.",
            "At least one <b>yellow</b> prompt box is drawn on this image. Green "
            "manual boxes do not count, and neither do masks.",
            "<b>Use First Image as Prompt</b> itself is ON. With YOLOE-vis and "
            "this off, the boxes you drew are never handed to the batch and "
            "every image comes back empty.",
        ),
        "The app now refuses to start a run that cannot produce anything and "
        "tells you which of these to change, so you should not lose a folder to "
        "it. The list is here so you can get it right the first time.",
        "One more thing worth knowing: with the toggle <b>ON</b> the current "
        "image is included in the run and its labels are overwritten, so the "
        "output has one file per image. With it <b>OFF</b> the run starts at the "
        "next image, on the assumption you already annotated this one by hand.")),

    ("Auto Annotate Remaining", _p(
        "Runs the current setup (detector, segmenter, prompt, thresholds, "
        "carried boxes) over the rest of the folder, with a progress dialog you "
        "can cancel. It <b>always overwrites</b> existing label files, so "
        "re-running after a threshold change does not need you to empty the "
        "output folder first.",
        "<b>Include Earlier Images</b> also processes the images before this "
        "one, appended after the remaining ones, so nothing in the folder is "
        "skipped.",
        "<b>Review Side by Side (post)</b> opens the comparison viewer as soon as the "
        "run finishes, with the originals on one side and this run's annotated "
        "images on the other. A two-stage pipeline saves both bounding boxes and "
        "segmentation, so it asks which you want to look at; a run that saved "
        "only one kind opens that one without asking. Closing the viewer, by "
        "Back, Esc or the window's close button, returns you to the main menu: "
        "the folder you just ran is finished.",
        "When it finishes you get a summary: how many were processed, how long "
        "it took, anything that failed, and anything that came back empty. "
        "Images needing a second look are copied into a <b>_review</b> folder "
        "alongside a review_report.csv listing why.")),

    ("Editing annotations", _p(
        "<b>Edit Boxes</b> is the default edit tool: drag a body to move it, "
        "drag a corner handle to resize, click the red X badge to delete. Undo "
        "and Redo live in the same dropdown.",
        "<b>Edit Masks</b> edits any committed mask, model-generated or "
        "hand-drawn. Click a mask to select it. In <b>Vertices</b> mode drag a "
        "vertex to move it, click the outline to add one, right-click a vertex "
        "to remove it. In <b>Points</b> mode left-click adds a SAM point, "
        "right-click removes one, and dragging a point re-runs SAM live. Press "
        "<b>S</b> for that mask's settings, <b>Enter</b> to keep your changes, "
        "<b>Esc</b> to back out. A mask you have edited is marked so a later "
        "Regenerate will not overwrite it.",
        "<b>Select Multiple</b> drags a rectangle and selects everything it "
        "touches. Each selected item shows a red X, or press Delete to remove "
        "them all at once. Esc dismisses the marquee. Turning it on switches "
        "Edit Boxes on for you.",
        "<b>Image Resize</b> zooms and pans while you keep drawing and editing "
        "with the left button. Its dropdown picks the input scheme: on a "
        "trackpad two-finger scroll pans and pinch zooms; with a mouse the wheel "
        "zooms and right-drag pans. <b>Original Size</b> puts it back. "
        "<b>Darken Tint</b> dims everything except your detections so they stand "
        "out, which affects the view only and never the saved images.",
        "<b>Save &amp; Confirm</b> writes what is on screen, including your "
        "edits, as the final labels for this image without re-running the model. "
        "<b>Next IMG</b> saves and advances. <b>Previous IMG</b> reloads the "
        "earlier image's saved annotations from disk exactly as you left them, "
        "so trimmed and edited results survive.")),

    ("Synthetic images (Stable Diffusion)", _p(
        "Expand <b>Synthetic Images (Diffusion)</b> on the right. This "
        "regenerates the background of an image while preserving everything you "
        "annotated, to grow a training set from a small number of real photos. "
        "It needs at least one annotated region to preserve, so run or draw "
        "something first.",
        "<b>Edit Prompts</b> opens a popup for the prompt (what the regenerated "
        "background should look like) and the negative prompt (what to keep "
        "out). Either can load a .txt file, appending to or replacing what is "
        "there.",
        "<b>Diffusion Strength</b> controls how far it departs from the "
        "original. Low keeps the real scene and looks most realistic; high "
        "repaints from the prompt and varies more, at the risk of unnatural "
        "fills. <b>0.1 to 0.2</b> is the range that worked best in testing, and "
        "0.20 is the default. Raise it only if the variations come out too "
        "similar to each other, and check the results: the fills start looking "
        "synthetic well before the slider runs out.",
        "<b>Generate Variation</b> does the current image and shows a "
        "side-by-side preview with Regenerate / Cancel / Save before anything is "
        "written. <b>Variations for Folder</b> does every image that already has "
        "a saved label file, then opens a viewer you can page through to delete "
        "the bad ones. Results land in a <b>synthetic images</b> folder with "
        "matching labels and segments.")),

    ("Keyboard and mouse", _p(
        "<b>Anywhere</b>",
        _bullets(
            "<b>Enter</b> runs the model on this image, the same as Regenerate.",
            "<b>Esc</b> clears the current selection and the marquee.",
            "<b>Delete</b> or <b>Backspace</b> removes everything selected, or "
            "the most recent hand-drawn box if nothing is selected.",
            "<b>Shift</b> and click adds or removes one item from a multi-selection.",
            "<b>Right-click</b> a box deletes it.",
        ),
        "<b>While drawing a mask</b>",
        _bullets(
            "<b>Enter</b> keeps the mask. In Manual Masks it closes the outline "
            "first, which you can also do by clicking the amber first point or "
            "double-clicking.",
            "<b>Esc</b> throws the whole in-progress mask away.",
            "<b>Delete</b>, <b>Backspace</b> or <b>right-click</b> removes the "
            "last point you placed.",
            "In Semi-Automatic Points, clicking <i>outside</i> the mask grows "
            "it, clicking <i>inside</i> cuts a piece out.",
        ),
        "<b>While editing a mask</b>",
        _bullets(
            "<b>Enter</b> applies your changes, <b>Esc</b> reverts them.",
            "<b>S</b> opens that mask's settings: Points or Vertices, class id, "
            "and simplify.",
            "<b>Delete</b> removes the last point, or the whole mask if there "
            "are no points to remove.",
        ),
        "Undo and Redo are in the Edit dropdown; there is no Ctrl+Z binding.")),
]


class UserManualOverlay(QtWidgets.QWidget):
    """The manual, drawn INSIDE the window that opened it.

    Deliberately not a QDialog, and the reason is worth keeping written down.
    Both windows that open this call showFullScreen(), and macOS defaults
    AppleWindowTabbingMode to "fullscreen", meaning "prefer tabs when opening
    windows in fullscreen". A QDialog is an ordinary top-level window, so the OS
    was merging the manual into the parent as a native window TAB, complete with
    the animation of it being pulled across. Positioning it differently and
    changing its modality both failed, because the thing macOS was reacting to
    was its being a separate window at all.

    A child widget has no NSWindow behind it, so there is nothing for any
    platform to tab, animate or misplace. That is what makes this immune rather
    than merely better behaved: the fix is structural, not a mitigation.

    Presented as a dimmed scrim over the whole window with a centred card, so
    the window stays visible around the edges and the manual reads as a layer on
    the page you were already on.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # (toggle button, body panel) per section, in display order. Kept for the
        # headless suite, which drives the toggles directly.
        self.sections = []
        # A stylesheet background is ignored on a QWidget SUBCLASS unless this is
        # set; Qt only auto-enables it for plain QWidget instances. Without it
        # the scrim paints nothing and the canvas shows straight through.
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        # StrongFocus so showing the overlay takes focus OFF the canvas. The
        # canvas grabs focus in three places and answers Delete by soft-deleting
        # an annotation, which the user would not see happen behind the manual.
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.init_ui()
        self.hide()

    def init_ui(self):
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        screen_h, screen_w = screen.height(), screen.width()
        font  = max(13, screen_h // 58)
        btn_h = max(40, screen_h // 16)
        # Scrim. Dark enough to push the window back, light enough to still read
        # it, so it is obvious you are on top of the same page.
        self.setStyleSheet("UserManualOverlay { background-color: rgba(0, 0, 0, 170); }")

        scrim = QtWidgets.QVBoxLayout(self)
        scrim.setContentsMargins(0, 0, 0, 0)
        scrim.addStretch()

        # The card carries everything the dialog used to. Centred by the
        # stretches around it, so a window resize re-centres it for free and only
        # the overlay itself ever needs explicit geometry.
        # Every widget below is given its parent AT CONSTRUCTION. A widget
        # built with no parent is a top-level widget until a layout adopts it,
        # and on macOS a top-level widget is a native window waiting to happen.
        # Adopting them a moment later is not the same as never being parentless.
        self.card = QtWidgets.QFrame(self)
        self.card.setObjectName("manualCard")
        self.card.setStyleSheet(
            "QFrame#manualCard { background-color: #3a3a3a; border-radius: 10px; }"
            "QLabel { color: white; }")
        self.card.setMaximumWidth(min(900, int(screen_w * 0.6)))
        self.card.setMaximumHeight(min(760, int(screen_h * 0.8)))
        scrim.addWidget(self.card, 0, QtCore.Qt.AlignHCenter)
        scrim.addStretch()

        outer = QtWidgets.QVBoxLayout(self.card)
        outer.setSpacing(BTN_GAP)
        outer.setContentsMargins(BTN_GAP * 2, BTN_GAP * 2, BTN_GAP * 2, BTN_GAP * 2)

        title_row = QtWidgets.QHBoxLayout()
        heading = QtWidgets.QLabel("User Manual", self.card)
        heading.setStyleSheet(
            f"color: white; font-size: {int(font * 1.3)}px; font-weight: bold;")
        title_row.addWidget(heading)
        title_row.addStretch()
        self.close_btn = QtWidgets.QPushButton("Close", self.card)
        self.close_btn.setStyleSheet(btn_qss(BTN_GREY, font))
        self.close_btn.setFixedHeight(int(btn_h * 0.8))
        self.close_btn.clicked.connect(self.close_overlay)
        title_row.addWidget(self.close_btn)
        outer.addLayout(title_row)

        intro = QtWidgets.QLabel(
            "Pick a topic. Start with the first one if you have not used this "
            "window before. Press Esc, or click outside this panel, to close.",
            self.card)
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: #cccccc; font-size: {font}px;")
        outer.addWidget(intro)

        # The scroll area and its viewport sit outside the card stylesheet's
        # reach, so they keep the platform's light default and leave the white
        # body text unreadable. Paint them to match the card.
        host = QtWidgets.QWidget(self.card)
        host.setStyleSheet("background-color: #3a3a3a;")
        body = QtWidgets.QVBoxLayout(host)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(BTN_GAP // 2)

        for idx, (title, html) in enumerate(MANUAL_SECTIONS):
            # First section open so the manual never opens looking empty; the
            # rest closed so every topic is reachable without scrolling.
            self._add_section(body, title, html, font, btn_h, expanded=(idx == 0))
        body.addStretch()

        scroll = QtWidgets.QScrollArea(self.card)
        scroll.setWidgetResizable(True)
        scroll.setWidget(host)
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollBar:vertical { background: #3a3a3a; width: 12px; margin: 0; }"
            "QScrollBar::handle:vertical { background: #6a6a6a; border-radius: 5px;"
            " min-height: 24px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
            "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical"
            " { background: transparent; }")
        outer.addWidget(scroll, 1)

    def _add_section(self, layout, title, html, font, btn_h, expanded=False):
        """One collapsible topic: a checkable header button and a body panel
        directly beneath it, the same construction the annotation window uses for
        Prompts, Sliders and Synthetic Images."""
        parent_w = layout.parentWidget()
        header = QtWidgets.QPushButton(
            f"{_ARROW_OPEN if expanded else _ARROW_CLOSED}  {title}", parent_w)
        header.setCheckable(True)
        header.setChecked(expanded)
        # Left-aligned, unlike the collapsibles in the annotation window. Those
        # are one-offs in a control column; eight of them stacked need their
        # arrows in a straight line down the edge to read as one list.
        header.setStyleSheet(
            btn_qss(BTN_BLUE, font)
            + "QPushButton { text-align: left; padding-left: 14px; }")
        header.setFixedHeight(int(btn_h * 0.8))
        header.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                             QtWidgets.QSizePolicy.Fixed)

        panel = QtWidgets.QWidget(parent_w)
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(BTN_GAP, BTN_GAP // 2, BTN_GAP, BTN_GAP)
        text = QtWidgets.QLabel(html, panel)
        text.setWordWrap(True)
        text.setTextFormat(QtCore.Qt.RichText)
        text.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        # Selectable so anyone can copy a step out into a note or a bug report.
        text.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        text.setStyleSheet(f"color: white; font-size: {font}px;")
        panel_layout.addWidget(text)
        panel.setVisible(expanded)

        header.toggled.connect(
            lambda on, h=header, p=panel, t=title: self._toggle_section(h, p, t, on))
        layout.addWidget(header)
        layout.addWidget(panel)
        self.sections.append((header, panel))

    @staticmethod
    def _toggle_section(header, panel, title, on):
        panel.setVisible(on)
        header.setText(f"{_ARROW_OPEN if on else _ARROW_CLOSED}  {title}")

    # Showing and hiding
    def show_over(self):
        """Cover the host window and take focus.

        An explicit method rather than a showEvent override: the geometry has to
        be right BEFORE the widget becomes visible, and showEvent fires after the
        first paint, which is what produced a visible jump last time."""
        host = self.parentWidget()
        if host is None:
            return
        self.setGeometry(host.rect())
        self.show()
        self.raise_()
        self.setFocus(QtCore.Qt.OtherFocusReason)
        _debug_window_state("after show_over", self)

    def close_overlay(self):
        """Hide and hand focus back to the window underneath."""
        self.hide()
        host = self.parentWidget()
        if host is not None:
            host.setFocus(QtCore.Qt.OtherFocusReason)

    def mousePressEvent(self, event):
        # Click the dimmed area to dismiss; clicks on the card itself are the
        # card's business. Nothing may fall through to the window behind, which
        # is why this accepts either way.
        if not self.card.geometry().contains(event.pos()):
            self.close_overlay()
        event.accept()

    def keyPressEvent(self, event):
        """Terminate every key here. This is a correctness guard, not a nicety.

        A QDialog stopped key propagation for free: an ignored key stops at the
        first widget whose isWindow() is true. A child widget is not a window, so
        anything this overlay ignores keeps bubbling to the host, and the host is
        ManualWindow, whose keyPressEvent maps Enter to display_predictions()
        without calling super() or ignore(). Left alone, pressing Enter with the
        manual open would start a model run behind it, and Delete would reach the
        canvas and soft-delete an annotation the user cannot see.

        Every widget in the card is a descendant of this one, so a key the Close
        button or the scroll area ignores arrives here before it can reach the
        host. Accepting without calling super() is what ends the chain; QWidget's
        default implementation ignores, which would restart it.
        """
        if event.key() == QtCore.Qt.Key_Escape:
            self.close_overlay()
        event.accept()
