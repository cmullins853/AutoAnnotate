# Windows Test Checklist

Things to verify on the Windows + NVIDIA machine before trusting this branch.
Run these after pulling the branch, in the project venv, from the repo root.
See WINDOWS_GPU_SETUP_HANDOFF.md for the full desktop setup; a machine with no
prior dev tooling needs STEP 0 of "GUI and Pipeline/HOW_TO_RUN.txt" first
(Python 3.13, Git, CUDA Toolkit, VS Build Tools).

NOTE after pulling a branch that touches the vendored GroundingDINO sources
(ms_deform_attn_cuda.cu, transformer.py, setup.py): re-run the editable
install once so the CUDA extension is rebuilt from the patched source:
`pip install --no-build-isolation -e "autoannotate study/GroundingDINO"`.

## 1. Install and CUDA

- [ ] Fresh venv on Python 3.13: `py -3.13 -m venv .venv` then `.\.venv\Scripts\Activate.ps1`.
- [ ] `pip uninstall -y torch torchvision torchaudio` (clear any CPU build).
- [ ] `pip install -r requirements-windows11-cuda.txt` completes with no resolver errors.
- [ ] `pip install wheel setuptools ninja` installs the GroundingDINO build helpers.
- [ ] Confirm the GPU wheels were installed, not CPU:
      `python -c "import torch; print(torch.__version__)"` shows a `+cu132` build.
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` prints `True`.
- [ ] `python -c "import torch; print(torch.version.cuda); print(torch.cuda.get_device_name(0))"`
      prints the CUDA version and your GPU name.
- [ ] Installing did NOT pull a second/CPU torch afterward (re-check `torch.__version__`
      still shows `+cu132` once ultralytics/diffusers finished installing).

For the separate Windows 10 CPU-only machine, use
`requirements-windows10-cpu.txt`, confirm `torch==2.6.0+cpu`,
`torchvision==0.21.0+cpu`, and confirm `torch.cuda.is_available()` is `False`.
Do not install `requirements-windows11-cuda.txt` or any `cuXXX` wheel there.

## 2. GroundingDINO build (the torch-adaptation fixes)

- [ ] `$env:CL = "/Zc:preprocessor"` then
      `pip install --no-build-isolation -e "autoannotate study/GroundingDINO"`.
- [ ] The CUDA source compiles with NO `DeprecatedTypeProperties` / `c10::ScalarType`
      error (this is the `value.scalar_type()` patch in ms_deform_attn_cuda.cu).
- [ ] `python -c "import groundingdino"` imports cleanly.
- [ ] Run a DINO text detection once and confirm NO crash or error from
      `torch.cuda.amp.autocast` (this is the `torch.amp.autocast` fix in transformer.py).

## 3. Environment report

- [ ] `python "GUI and Pipeline/check_environment.py"` runs and reports: CUDA
      available, torch CUDA version, GPU name, GroundingDINO path found, weights
      present, effective max-area value, and the Stable Diffusion device.
- [ ] It reports `GroundingDINO _C extension : [ OK ] importable`. If this is a
      warning on the CUDA machine, rebuild the editable GroundingDINO install
      before testing SwinT or SwinB performance.

## 4. App launch and GPU usage

- [ ] `python run_app.py` opens the window with no console errors.
- [ ] Run a detection and confirm it uses the GPU, not CPU (watch Task Manager
      GPU load, or nvidia-smi), and that it is noticeably faster than CPU.
- [ ] No silent fallback to CPU on any detector (DINO, YOLOE-vis, YOLOE-seg, SAM3).

## 5. Prompt UI (this branch)

- [ ] Type a prompt in the first field, use "+ Add prompt" for a second concept,
      confirm both feed detection (e.g. person / car).
- [ ] "+ Add prompt" and "+ Add negative" grey out at 5 fields; re-enable after removing one.
- [ ] Removing the last field clears it instead of deleting it.
- [ ] Negatives with something typed actually suppress overlapping detections;
      leaving negatives blank never blocks a run.
- [ ] Collapse the "Prompts" dropdown: the fields hide but the info dot stays visible.
- [ ] Hover the info dot: the color legend shows each class with its color, negatives in red.
- [ ] Switch to Boxes mode / YOLOE-vis: the whole Prompts section hides, returns in Text mode.
- [ ] Draw Boxes menu has "Class for new boxes"; picking a class tags a hand-drawn box with it.

## 6. Layout (verify on the Windows display / DPI)

- [ ] No button in the left column is clipped on the right by the scrollbar.
- [ ] Detector and Segmenter dropdowns are equal width.
- [ ] "Previous IMG" and "Next IMG" sit on one row at half width each.
- [ ] Auto Annotate Remaining and the right-side tools are all visible (not pushed off-screen).
- [ ] Scrollbar appears only when the column is taller than the window, and scrolls to everything.

## 7. Max detection size slider

- [ ] Slider shows under "Segmenter confidence" and initializes from any
      `AUTOANNOTATE_MAX_AREA_FRAC` in `.env` (else 0.50).
- [ ] Low value (e.g. 0.10-0.20) drops stray oversized masks on small objects (blueberries).
- [ ] High value (near 1.00) keeps large detections (red leaf).
- [ ] The setting persists as you move between images and applies during Auto Annotate Remaining.

## 8. Batch behavior

- [ ] Auto Annotate Remaining processes the expected images.
- [ ] "Use First Image as Prompt" ON while on the first image: output folder gets
      all N images (not N-1).
- [ ] "Include Earlier Images" ON while starting mid-folder: earlier images are
      appended and the whole folder is covered, with the current image not duplicated.

## 8b. Review Side by Side (post-batch)

- [ ] "Review Side by Side (post)" On, two-stage pipeline (e.g. DINO + SAM2):
      after both end-of-run popups the "Which annotated images" prompt appears,
      because the run saved BOTH boxes and masks. It must ask regardless of which
      of the Bounding Box / Segmentation checkboxes is ticked; those two untick
      each other and no longer decide this.
- [ ] Each answer opens the matching folder (`output/annotated_<model>/boxes` or
      `.../masks`), paired by filename. Cancel, on the left edge, opens nothing.
- [ ] A run that saved only ONE kind (bbox-only pipeline) opens it with no
      prompt.
- [ ] Prev/Next steps both panes together, the swap arrow moves titles + folder
      buttons + images together, and per-pane zoom works.
- [ ] Back, Esc AND the window's close button each return to the MAIN MENU, at
      full screen size. A quarter-size window here is the bug this replaced.
- [ ] Toggle Off: the run ends exactly as before, with no viewer.
- [ ] Cancel a run partway with the toggle On: the viewer still opens on what
      finished.

## 8c. CUDA memory over a long batch

- [ ] `nvidia-smi` during a long two-stage run (YOLOE or DINO -> SAM3): VRAM
      plateaus instead of climbing image after image.
- [ ] The last images of the run complete rather than failing with "CUDA out of
      memory" the way they did before.
- [ ] If an out-of-memory does happen, the console prints
      `[oom] ... freeing every cached model and retrying once` and the image
      completes on the retry.
- [ ] An image that fails anyway is reported with the AUTOANNOTATE_MODEL_BUDGET_GB
      / AUTOANNOTATE_BATCH_CHUNK hint, in the popup and in review_report.csv.
- [ ] `AUTOANNOTATE_MODEL_BUDGET_GB=0` restores the old unbounded behavior
      (useful to confirm the derived budget is what changed things).

## 9. VS Code .env toast

- [ ] Open the repo in VS Code, save the `.env` file, and confirm the toast
      "An environment file is configured but terminal environment injection is
      disabled" no longer appears (fixed by `.vscode/settings.json`).

## 9b. Box prompting (multi-class positive + red negative)

- [ ] Pick a box-capable detector (YOLOE-vis / YOLOE-seg / SAM3) and tick the Boxes radio;
      confirm the text prompt fields disappear and the box section appears
      ("Draw box as" dropdown + info dot + "Draw Negative Box" toggle).
- [ ] The "Draw box as" dropdown lists 5 colored slots (Class 0-4). Draw boxes of two
      different classes and confirm they render in different colors on the image.
- [ ] Regenerate: confirm each class's boxes label detections with that class (the whole
      point that needs a GPU) and class_colors.txt lists class_0..class_N.
- [ ] Turn "Draw Negative Box" ON, draw a red box over an object type you want gone, and
      Regenerate: confirm matching detections are suppressed. Turn it OFF to draw
      positives again.
- [ ] Run Auto Annotate Remaining and confirm the red negative box suppresses look-alikes
      on the OTHER images too (appearance-based / cross-image).
- [ ] Press Next IMG instead of batch: the suppression should keep working on the
      following images even though the red box is no longer drawn on them.
- [ ] Return to the image where the red box was drawn, delete it, Regenerate: the
      suppression stops from then on (delete means gone).
- [ ] Switch back to Text mode and confirm the box section hides and the text fields return.

## 9c. Synthetic variations (Stable Diffusion)

The regenerate now runs on a worker thread. Windows is where an inline model call
shows up as a greyed-out window titled "Not Responding" with an offer to kill the
app, so this section is really only testable here.

- [ ] Annotate an image, click "Generate Variation", and confirm the side-by-side
      preview opens with the original on the left and the variation on the right.
- [ ] Click Regenerate. While "Generating..." is showing: drag the window, resize
      it, and confirm it keeps REPAINTING (no white/frozen client area, no
      "Not Responding" in the title bar).
- [ ] While it generates, Regenerate AND Save are both greyed out; both come back
      when the new variation appears.
- [ ] Save writes the image you are looking at, into `synthetic images/`, with a
      matching label file.
- [ ] Click Regenerate and then Cancel WITHOUT waiting: the dialog closes
      immediately (it must not hang until the inpaint finishes), the app stays
      alive, and no "QThread: Destroyed while thread is still running" or
      "Regenerate failed" box appears afterwards.
- [ ] Force a failure (e.g. an image with no annotations, or set the strength to
      an extreme) and confirm a "Regenerate failed" dialog appears rather than the
      button silently springing back.
- [ ] "Variations for Folder" on a folder with labels: the run finishes, the
      flipper opens, and "Delete this variation" removes BOTH the .jpg and the
      .txt (check the labels folder), leaving no `.deleting` files behind.

## 10. Regression sanity

- [ ] `set QT_QPA_PLATFORM=offscreen` then
      `python "GUI and Pipeline/test_semiauto_headless.py"` prints all checks passing (611/611, 0 skipped; a SKIP line means something on this machine could not be tested).
- [ ] Save labels, flip images, and confirm saved annotations reload unchanged.
