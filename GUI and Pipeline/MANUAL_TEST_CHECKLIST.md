# AutoAnnotate Manual GUI Test Checklist

Everything in the GUI is currently **headless/logic-verified only**; this is the
list to walk through in the live app before trusting it. Order matters: do
Section 0 (setup), then Section A (this session's changes, the highest risk), then
B (core regression), then C/D as time allows.

Tick `[x]` as you confirm each. Note the machine + OS you tested on at the top of
your run, and keep **Activity Monitor (macOS) / Task Manager** open the whole
time, because SAM3 (~3.3 GB) on an 8 GB box is the documented silent-OOM risk.

Tested on: __________   OS: __________   Date: __________

---

## 0. Setup / environment
- [ ] Activate the venv, then run `python "GUI and Pipeline/check_environment.py"`
      → all required packages OK, weights present, the SD device line looks right.
- [ ] Launch the notebook on the **venv kernel** ("Python (AutoAnnotate .venv)").
- [ ] Run cells 0 to 5 in order with no errors. Cell 0 prints the `[HF]` token-free
      line (you can leave HF_TOKEN blank).
- [ ] Splash screen loads SmolVLM, then the main menu opens automatically.
- [ ] Greyed dropdown entries: temporarily move `sam3.pt` aside → SAM3 entries
      (detector + segmenter) grey out with a tooltip; restore it → re-enabled.

---

## A. Changes made this session (verify these first)

### A1. Box-prompt + carry-forward matrix (all detectors × both modes)
For each detector, load a folder, pick it, set an output folder, and run.
- [ ] **DINO (SwinT)**: Text mode only (box radio greyed). Type a prompt → boxes
      appear. Box radio is disabled.
- [ ] **DINO (SwinB)**: same as SwinT (heavier).
- [ ] **YOLOE-vis**: Boxes mode only (text radio greyed). Draw a yellow box →
      Auto annotate → it finds similar objects. Drawing routes to the **yellow**
      prompt bucket.
- [ ] **YOLOE-seg (one-shot)**: both modes. Text mode: prompt → boxes+masks.
      Boxes mode: draw box → boxes+masks. Pair with segmenter "(none)".
- [ ] **SAM3 (one-shot)**: both modes. Text and Boxes each produce boxes+masks.
- [ ] **"Use First Image as Prompt" toggle** is **enabled** for YOLOE-vis/-seg and
      SAM3, and **greyed** for DINO (with the "DINO is text-only…" tooltip).

### A2. Carry-forward across images (the toggle ON)
- [ ] YOLOE-vis: draw a box on image 1, toggle ON, **Next Image** → image 2 runs
      automatically and finds the object via the carried box (no redraw needed).
- [ ] YOLOE-vis: **Auto Annotate Remaining** with toggle ON → every remaining
      image uses image 1's box as the reference; labels land in `output/boxes/`.
- [ ] SAM3 (Boxes mode): box on image 1, toggle ON, Next Image → SAM3 finds the
      same-LOOKING object on image 2 (crop-composite), NOT whatever sits at the
      old box coordinates. **Watch for** masks "burned in" top-left (patch block).
- [ ] SAM3 / YOLOE-seg (Text mode): the typed prompt carries automatically across
      Next Image / Auto Annotate Remaining (toggle is irrelevant in text mode).
- [ ] DINO: typed prompt carries across images regardless of the (greyed) toggle.

### A3. SAM3 text↔boxes mode switch (must not crash)
- [ ] SAM3, **Text** mode: run with a prompt (e.g. "berry") → note the result.
- [ ] Switch to **Boxes** mode, draw a box, Regenerate → it must RUN (no
      `IndexError: list index out of range` in set_classes). This previously
      crashed because a "clear stale text" guard set an empty text list; that
      guard is removed.
- [ ] Switch back to Text, change the prompt, Regenerate → clean text result.
      NOTE: there is now NO text-concept clear, so a Boxes run right after a Text
      run may be slightly biased by the prior text. That's tolerable (non-
      crashing); a proper clear is deferred until the SAM3 prompt API is verified.

### A4. Regenerate purple-mask loss (index-alignment fix)
- [ ] Seg mode, any detector+SAM. Auto annotate → get **purple** (detector) masks.
- [ ] Hand-draw a mask (Manual Masks / Semi-Auto Points) over one object → it's a
      **green** sticky mask.
- [ ] **Regenerate.** Confirm: the green hand-mask survives, AND the other purple
      masks do **not** vanish/shift. Toggle Bounding Box ↔ Segmentation a few
      times → no masks disappear on the round-trip.
- [ ] Watch the console for `[SEG-SYNC] WARNING…`. It should **not** appear; if it
      does, capture it.

### A5. Memory budget + chunked batch (OOM behaviour) on an 8 GB machine
- [ ] Baseline: YOLOE-vis + SAM3 on one image with **no** env vars set → still
      works (default unbounded cache).
- [ ] Set `AUTOANNOTATE_MODEL_BUDGET_GB=4.5` before launching. Run YOLOE-vis +
      SAM3 → memory pressure stays out of the red; only one heavy model resident
      at a time. With `AUTOANNOTATE_DEBUG=True` you'll see `[model-cache] evict …`.
- [ ] Set `AUTOANNOTATE_BATCH_CHUNK=4` + the budget, then **Auto Annotate
      Remaining** on a ~12-image folder with YOLOE→SAM3 → the dialog shows
      "Detecting … / Segmenting …" phases, completes without the silent crash,
      and `boxes/` + `segments/` are populated for every image.
- [ ] DINO+SAM2 and SAM3-standalone batch runs still finish (regression of the
      non-two-stage path).

### A6. Large-detection + segments/boxes parity (2026-06-22 fixes)
- [ ] **Batch one-shot segments == boxes.** YOLOE-seg one-shot (and SAM3 one-shot),
      run **Auto Annotate Remaining** on a folder. Open `boxes/` vs `segments/` for
      the same image: the seg view must have the **same count and extent** as the
      box view, with no extra detections, no giant boxes/masks around leaves that the
      box view doesn't have. (Was: segments re-derived every RAW detection.)
- [ ] **Max-area default is now 0.5.** Any detector: a box/mask covering more than
      ~half the image is dropped. Confirm large leaf-spanning detections are gone
      across DINO, YOLOE, SAM3. If it now drops a legitimately large subject, raise
      `AUTOANNOTATE_MAX_AREA_FRAC` (e.g. 0.8) and re-run.
- [ ] **DINO honors the knob.** With `AUTOANNOTATE_MAX_AREA_FRAC=0.3`, DINO drops
      more large boxes than at default (previously DINO ignored this env var).
- [ ] **Per-model review folder.** Force a review case (0 detections on an image).
      Confirm copies land in `<output>/_review/<model_tag>/{boxes,segments}/` with a
      `review_report.csv`, and the completion dialog's "Review folder:" path includes
      the model tag. Run a DIFFERENT model → its review goes to a separate subfolder;
      the first model's review survives. A clean run (no problems) leaves no
      `_review/<model_tag>` (and removes an empty `_review`).

---

## B. Core functionality (regression)

### B1. Folders + saving
- [ ] Select image folder (jpg/png) → first image shows; "Image X of N" indicator.
- [ ] Select output folder → labels go to `boxes/`, masks to `segments/`.
- [ ] Saved label files are named after the **real image** (no `tmp*.txt` orphans).
- [ ] Annotated reference images land in `annotated_<model>/` (e.g. `_SwinT_SAM2`),
      and switching detector/segmenter writes to a **different** folder.
- [ ] Re-running Auto Annotate overwrites that image's label cleanly (no pile-up).

### B2. Detector × segmenter combinations
- [ ] DINO(SwinT)+SAM2, DINO(SwinB)+SAM2, DINO+SAM3 → boxes then masks.
- [ ] YOLOE-vis+SAM3, YOLOE-seg+SAM2 → two-stage masks.
- [ ] One-shot standalone (YOLOE-seg / SAM3 with segmenter "(none)") → own masks.

### B3. Display modes & editing
- [ ] Bounding Box vs Segmentation checkboxes switch the view; switching preserves
      edits (no wipe).
- [ ] Draw a new box → appears; **manual boxes are green**, detector magenta/purple.
- [ ] Click a box → select; red **X** badge deletes it.
- [ ] Rect corner/edge handles resize.
- [ ] Deleted region is **not** re-added by the next Regenerate (reject list).
- [ ] **Manual box wins**: draw a box overlapping a detector box → Regenerate →
      the drawn box persists, the detector duplicate is dropped (no double box).
- [ ] **Select Multiple** marquee → multi-select → delete several at once.

### B4. Semi-auto SAM tools (need SAM2/SAM3 selected)
- [ ] **Semi-Auto Points**: click points → live SAM mask preview → Enter commits.
- [ ] **Manual Masks**: multi-point outline → SAM fills, clipped to the outline
      (no leaf-bleed past it).
- [ ] Edit a committed mask's vertices; switching tools mid-draft prompts before
      discarding uncommitted points.
- [ ] Switching to a non-SAM detector while a SAM tool is in use prompts
      "Keep SAM model / Switch anyway".

### B5. Zoom / pan (Image Resize)
- [ ] Toggle Image Resize → wheel/trackpad-pinch zooms toward cursor; drag pans.
- [ ] Overlays, handles, and click hit-tests line up correctly while zoomed.
- [ ] Zoom persists after untoggling; "Default" resets to fit; Next Image starts
      at fit.

### B6. Navigation
- [ ] Next Image steps forward; Save & Confirm equivalent runs (edits not lost).
- [ ] Last image → "folder complete" alert; input/output folders deselect.

---

## C. Synthetic images (Stable Diffusion), optional, slow on CPU
- [ ] Expand the Synthetic Images panel; edit prompt/negative in the popup.
- [ ] Generate one variation → foreground preserved, background varied; strength
      slider behaves (low = realistic).
- [ ] Confirm the device matches `AUTOANNOTATE_SD_DEVICE` (check_environment line).

---

## D. Edge cases / known-fragile
- [ ] Empty prompt + no boxes → friendly "please enter a prompt / draw a box"
      alert, no crash.
- [ ] First SAM3 text call: buttons grey for several seconds (model load) then
      return. This is normal.
- [ ] Run a whole folder, then load a NEW folder → no stale boxes/marquee/carry
      anchor bleed into the new session.
- [ ] Cancel an Auto Annotate Remaining mid-run → partial results kept, UI
      recovers, buttons re-enable.

---

## Notes / bugs found while testing
(Write console output verbatim, esp. any `[SEG-SYNC]`, `[sam3]`, `[carry]`,
`[model-cache]`, or traceback lines, plus the detector/mode/steps to reproduce.)

-
-
-
