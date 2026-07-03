"""Splash screen and the main menu window."""
import os

from PyQt5 import QtWidgets, QtGui, QtCore

from ..config import WEIGHTS_DIR
from .llm import LLMWorker
from .style import BTN_BLUE, BTN_GREY, BTN_PURPLE, BTN_RED, btn_qss

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
        self.main_window = MainWindow(self.model, self.processor)
        self.main_window.show()
        self.close()


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

        exit_btn = QtWidgets.QPushButton("Exit")
        exit_btn.setStyleSheet(btn_qss(BTN_GREY, font))
        exit_btn.setFixedSize(200, exit_h)
        exit_btn.setToolTip("Close the application.")
        exit_btn.clicked.connect(self.close)
        layout.addWidget(exit_btn, alignment=QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

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
        self.manual_window = ManualWindow(self.model, self.processor)
        self.manual_window.show()
        self.hide()

    def select_automated(self):
        from .automated_window import AutomatedWindow
        self.automated_window = AutomatedWindow(self.model, self.processor)
        self.automated_window.show()
        self.hide()

    def select_side_by_side(self):
        from .side_by_side import SideBySideWindow
        self.side_by_side_window = SideBySideWindow(self.model, self.processor)
        self.side_by_side_window.show()
        self.hide()
