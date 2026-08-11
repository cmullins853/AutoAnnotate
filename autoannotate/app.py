"""Application entry point: builds the Qt app and shows the splash screen."""
import sys

# torch is imported HERE, before PyQt5, and the order matters on Windows.
#
# Qt installs its own DLL search path when it loads. If it gets there first,
# torch's _load_dll_libraries can fail to bring up c10.dll with "OSError:
# [WinError 1114] A dynamic link library (DLL) initialization routine failed",
# which surfaces as the app dying on startup with a message naming a file
# nobody has heard of. Importing torch first avoids it, costs nothing on macOS
# and Linux, and this is the first place the application touches either
# library, so it is the right place to decide the order.
#
# Deliberately not done in autoannotate/__init__.py: that would pull torch into
# tools that have no use for it, such as python -m autoannotate.coco.
import torch  # noqa: F401  (imported for DLL load order, not for use here)

from PyQt5 import QtWidgets

from . import config  # noqa: F401  (env setup side effects must run first)
from .gui.splash import SplashScreen

def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    splash = SplashScreen()
    splash.show()
    app.exec_()
