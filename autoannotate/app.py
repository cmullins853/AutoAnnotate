"""Application entry point: builds the Qt app and shows the splash screen."""
import sys

from PyQt5 import QtWidgets

from . import config  # noqa: F401  (env setup side effects must run first)
from .gui.splash import SplashScreen

def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    splash = SplashScreen()
    splash.show()
    app.exec_()
