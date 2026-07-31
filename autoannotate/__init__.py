"""AutoAnnotate: semi-automatic image annotation pipeline and GUI.

The package is split in two layers:
  autoannotate.pipeline  -- model wrappers and label I/O (no Qt)
  autoannotate.gui       -- the PyQt5 application built on the pipeline

Launch the app from a terminal with `python -m autoannotate` (or
`python run_app.py` at the repo root). No notebook or IDE is involved;
the old Jupyter launcher has been retired.
"""
import os

# CUDA allocator strategy, set HERE rather than in config.py because it has to
# be in the environment before torch is imported and gui/manual_window.py
# imports torch ABOVE it imports ..config. This module runs first for every
# entry point (python -m autoannotate, run_app.py, the headless test harness),
# so it is the only place that is reliably early enough.
#
# expandable_segments lets the caching allocator grow a segment in place instead
# of reserving fixed-size blocks it can never merge again. Without it a long
# batch run fragments VRAM until an allocation that would fit has no contiguous
# block left, which is why CUDA out-of-memory hit the LAST few images of a run
# rather than the first. setdefault so an explicit shell value still wins; a
# torch build that does not know the flag warns and ignores it.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
