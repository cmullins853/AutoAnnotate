"""AutoAnnotate: semi-automatic image annotation pipeline and GUI.

The package is split in two layers:
  autoannotate.pipeline  -- model wrappers and label I/O (no Qt)
  autoannotate.gui       -- the PyQt5 application built on the pipeline

Launch the app from a terminal with `python -m autoannotate` (or
`python run_app.py` at the repo root). No notebook or IDE is involved;
the old Jupyter launcher has been retired.
"""
