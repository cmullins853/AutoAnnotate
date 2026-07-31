"""Splash screen and the main menu window."""
import os

from PyQt5 import QtWidgets, QtGui, QtCore

from ..config import WEIGHTS_DIR
from .llm import LLMWorker
from .style import BTN_BLUE, BTN_GAP, BTN_GREEN, BTN_GREY, BTN_PURPLE, BTN_RED, btn_qss

# Windows that have to outlive whatever created them.
#
# Every navigation edge in this app builds its destination and abandons its
# source, and the source is normally the only thing holding a reference to the
# destination. Without an owner here, destroying the source would take the new
# window down with it. A plain module-level list is enough: the app shows one
# window at a time and these live for the process.
_LIVE_WINDOWS = []


def hand_off(new_window, old_window=None):
    """Show `new_window`, then destroy `old_window`, moving ownership across.

    Use this for every window-to-window transition. Abandoning a window used to
    mean hide(), which leaves a live NATIVE window behind: close() and hide()
    both only take a top-level widget off screen. On macOS a fullscreen window
    owns a Space, so each abandoned one is an empty Space the app can be sent
    to, and the user lands on a blank screen they have to navigate out of. Every
    window here goes fullscreen in its init_ui, so they all qualify. Measured
    before this existed, one Menu -> Manual -> Back round trip leaked two of
    them, and three round trips left six windows where there should be one.

    Order matters. The new window is shown BEFORE the old one closes, so there
    is never a moment with no visible window: quitOnLastWindowClosed defaults to
    true, and closing the only visible window would end the application.
    """
    _LIVE_WINDOWS.append(new_window)
    new_window.show()
    if old_window is None:
        return
    old_window.close()
    # Drop our own reference before the queued delete, so the registry does not
    # accumulate wrappers around deleted C++ objects for the life of the process.
    try:
        _LIVE_WINDOWS.remove(old_window)
    except ValueError:
        pass
    # Queued: safe to call on the window whose method is running right now,
    # because it only fires once the current event-loop pass unwinds.
    old_window.deleteLater()


class SplashScreen(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.model = None
        self.processor = None
        self.init_ui()
        self.start_model_loading()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        self.setStyleSheet("background-color: black;")  # Splash screen background

        # Logo with transparency
        label = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap(os.path.join(WEIGHTS_DIR, "AMS_Logo_Final_Removed.png"))
        label.setPixmap(pixmap.scaledToWidth(400, QtCore.Qt.SmoothTransformation))
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setStyleSheet("background: transparent;")  # Ensures QLabel doesn't add its own background
        layout.addWidget(label)

        # Live log output
        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("color: white; background-color: #111; font-size: 18px;")
        layout.addWidget(self.log_box)

        self.setLayout(layout)
        self.resize(800, 600)
        self.center()


    def center(self):
        frameGm = self.frameGeometry()
        screen = QtWidgets.QApplication.primaryScreen()
        centerPoint = screen.geometry().center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())

    def start_model_loading(self):
        self.thread = QtCore.QThread()
        self.worker = LLMWorker()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.model_ready)
        self.worker.log.connect(self.append_log)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def append_log(self, text):
        self.log_box.appendPlainText(text)

    def model_ready(self, model, processor):
        self.model = model
        self.processor = processor
        # Always open main window; model=None just disables LLM features
        QtCore.QTimer.singleShot(1000, self.show_main_window)

    def show_main_window(self):
        # Destroys the splash rather than hiding it; see hand_off.
        hand_off(MainWindow(self.model, self.processor), self)


class MainWindow(QtWidgets.QWidget):
    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Prompt and Confidence Tuning")
        self.showFullScreen()
        self.setStyleSheet("background-color: #454545;")

        # Scale to screen so the menu fits any display (was hardcoded 800x150 /
        # 36px, which overflowed smaller laptop screens).
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        screen_h, screen_w = screen.height(), screen.width()
        font       = max(13, screen_h // 58)
        menu_font  = max(20, screen_h // 30)
        menu_w     = min(800, int(screen_w * 0.6))
        menu_h     = max(90, screen_h // 6)
        exit_h     = max(40, screen_h // 16)

        layout = QtWidgets.QVBoxLayout()

        # Top row rather than a fourth menu button: the three menu buttons are
        # already menu_h tall each, and a fourth of that size crowds a short
        # laptop screen. The manual is a quick reference, not a mode, so it gets
        # Exit's small footprint.
        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(BTN_GAP)

        exit_btn = QtWidgets.QPushButton("Exit")
        exit_btn.setStyleSheet(btn_qss(BTN_GREY, font))
        exit_btn.setFixedSize(200, exit_h)
        exit_btn.setToolTip("Close the application.")
        exit_btn.clicked.connect(self.close)
        top_row.addWidget(exit_btn)

        manual_help_btn = QtWidgets.QPushButton("User Manual")
        manual_help_btn.setStyleSheet(btn_qss(BTN_GREEN, font))
        manual_help_btn.setFixedSize(200, exit_h)
        manual_help_btn.setToolTip(
            "Step-by-step instructions for the manual annotation window: "
            "prompts, box annotation, carrying prompts forward, editing, "
            "synthetic images and the keyboard shortcuts.")
        manual_help_btn.clicked.connect(self.open_user_manual)
        top_row.addWidget(manual_help_btn)

        top_row.addStretch()
        layout.addLayout(top_row)

        button_layout = QtWidgets.QVBoxLayout()
        # Breathing room between the stacked menu buttons.
        button_layout.setSpacing(30)

        manual_btn = QtWidgets.QPushButton("Manual Prompt and Confidence Tuning")
        manual_btn.setStyleSheet(btn_qss(BTN_BLUE, menu_font))
        manual_btn.setFixedSize(menu_w, menu_h)
        manual_btn.setToolTip("Hand-tune prompts, boxes and confidence on each image, "
                              "with manual editing and synthetic-image generation.")
        manual_btn.clicked.connect(self.select_manual)
        button_layout.addWidget(manual_btn, alignment=QtCore.Qt.AlignCenter)

        automated_btn = QtWidgets.QPushButton("Automated Prompt and Confidence Tuning")
        automated_btn.setStyleSheet(btn_qss(BTN_RED, menu_font))
        automated_btn.setFixedSize(menu_w, menu_h)
        automated_btn.setToolTip("Auto-search the best prompt and confidence against a "
                                 "labelled reference set, then batch-annotate a folder.")
        automated_btn.clicked.connect(self.select_automated)
        button_layout.addWidget(automated_btn, alignment=QtCore.Qt.AlignCenter)

        side_by_side_btn = QtWidgets.QPushButton("View Images Side by Side")
        side_by_side_btn.setStyleSheet(btn_qss(BTN_PURPLE, menu_font))
        side_by_side_btn.setFixedSize(menu_w, menu_h)
        side_by_side_btn.setToolTip("Compare two image folders (e.g. synthetic vs. "
                                    "ground truth) paired by filename.")
        side_by_side_btn.clicked.connect(self.select_side_by_side)
        button_layout.addWidget(side_by_side_btn, alignment=QtCore.Qt.AlignCenter)

        layout.addStretch()
        layout.addLayout(button_layout)
        layout.addStretch()
        layout.setAlignment(button_layout, QtCore.Qt.AlignCenter)
        self.setLayout(layout)

    # The window modules import MainWindow back for their Back buttons, so
    # import them lazily here to keep module loading acyclic.
    def select_manual(self):
        from .manual_window import ManualWindow
        hand_off(ManualWindow(self.model, self.processor), self)

    def select_automated(self):
        from .automated_window import AutomatedWindow
        hand_off(AutomatedWindow(self.model, self.processor), self)

    def select_side_by_side(self):
        from .side_by_side import SideBySideWindow
        hand_off(SideBySideWindow(self.model, self.processor), self)

    def open_user_manual(self):
        """Show the manual over the menu.

        An overlay INSIDE this window, not a separate one: this window is
        fullscreen, and macOS merges new windows opened from a fullscreen window
        into it as native tabs. See UserManualOverlay for the full story. Built
        once and reused, so reopening is instant and its expanded sections are
        where you left them."""
        from .user_manual import UserManualOverlay
        if getattr(self, "_manual_overlay", None) is None:
            self._manual_overlay = UserManualOverlay(self)
        self._manual_overlay.show_over()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # The overlay sits outside the layout, so nothing else will resize it.
        # getattr rather than hasattr: showFullScreen() in init_ui fires a resize
        # while init_ui is still running, before the attribute exists.
        ov = getattr(self, "_manual_overlay", None)
        if ov is not None and ov.isVisible():
            ov.setGeometry(self.rect())
