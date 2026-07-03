"""AutoAnnotate: semi-automatic image annotation pipeline and GUI.

The package is split in two layers:
  autoannotate.pipeline  -- model wrappers and label I/O (no Qt)
  autoannotate.gui       -- the PyQt5 application built on the pipeline

Launch the app with `python -m autoannotate` (or run_app.py at the repo
root). The old notebook, auto-annotate-gui.ipynb, is now a thin launcher
that imports this package.
"""
