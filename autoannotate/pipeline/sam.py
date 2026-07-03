"""SAM2/SAM3 segmenter loading and the SAM3 semantic (text / box-exemplar) modes."""

from ..config import weights_path

import torch
from ultralytics import SAM

# --- SAM / YOLOE wrappers (used by pipeline preset dropdown) -----------------
SAM_VARIANTS = {
    "sam2_t": "sam2_t.pt",
    "sam2_b": "sam2_b.pt",
    "sam3":   "sam3.pt",
}

def load_sam(variant="sam2_t"):
    """Load a SAM-family segmenter via ultralytics. Weights auto-download on first call."""
    ckpt = SAM_VARIANTS.get(variant, variant if variant.endswith(".pt") else f"{variant}.pt")
    return SAM(weights_path(ckpt))

# Module-level cache: building the SAM3SemanticPredictor loads a multi-hundred-MB
# model variant separate from the interactive SAM3, so reuse the instance across
# calls. Cleared implicitly when the kernel restarts.
_sam3_text_predictor = None
# Last prompt mode driven through the shared predictor ('text'|'boxes'|None).
# Tracked for a future safe text->boxes clear; the box path does NOT clear today
# (an empty text list crashes ultralytics set_classes -> text[0] IndexError).
_sam3_last_mode = None

def run_sam3_text(image_path, names, conf=0.25, max_area_frac=0.9):
    """Run SAM3 with text/class prompts (open-vocabulary semantic segmentation).

    The public `ultralytics.SAM('sam3.pt')` class wraps SAM3Predictor (interactive),
    which only accepts box/point prompts. Text prompts live in a separate
    `SAM3SemanticPredictor` that uses `build_sam3_image_model` instead of
    `build_interactive_sam3`: same .pt file, different architecture.

    Args:
      names: a class string ("cat" or "cat, dog") or a list of class strings.
      conf:  detection-score threshold (post-NMS filter inside the predictor).
      max_area_frac: drop any box that covers more than this fraction of the
        image (matches run_yoloe_*'s "kill the giant whole-image box" filter).

    Returns:
      (out_boxes_xyxy_list, raw_results). `raw_results[0].masks.xyn` carries
      normalised polygon coords aligned with `raw_results[0].boxes.xyxy`.
    """
    global _sam3_text_predictor, _sam3_last_mode
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    if not names:
        return [], None
    try:
        from ultralytics.models.sam.predict import SAM3SemanticPredictor
    except ImportError as e:
        raise RuntimeError(
            "SAM3 text mode needs an ultralytics version that ships "
            f"SAM3SemanticPredictor. Import failed: {e}"
        ) from e
    if _sam3_text_predictor is None:
        try:
            _sam3_text_predictor = SAM3SemanticPredictor(overrides=dict(
                conf=conf, task="segment", mode="predict", imgsz=1024, model=weights_path("sam3.pt"),
                save=False, save_txt=False, save_crop=False,
            ))
        except FileNotFoundError as e:
            raise RuntimeError(
                "sam3.pt not found. Fetch it from "
                "https://huggingface.co/facebook/sam3 (gated; accept the "
                "license) and drop it next to the notebook in 'GUI and Pipeline/'."
            ) from e
    else:
        _sam3_text_predictor.args.conf = conf
    # Force a full feature refresh for THIS image. dataset=None alone is
    # not enough - the SAM3SemanticPredictor caches image embeddings via
    # set_image() (the same mechanism the box path uses). Without calling
    # set_image() on every text call, the predictor reuses the first
    # image's features -> the same masks "burn in" at the same pixel
    # positions on every subsequent image AND switching from box mode to
    # text mode keeps the box-mode features.
    _sam3_text_predictor.dataset = None
    _sam3_text_predictor.set_image(image_path)
    _sam3_text_predictor.set_prompts({"text": names})
    _sam3_last_mode = "text"
    results = _sam3_text_predictor(source=image_path)
    out_boxes = []
    if results and getattr(results[0], "boxes", None) is not None:
        shape = results[0].orig_shape
        max_area = shape[0] * shape[1] * max_area_frac
        for box in results[0].boxes.xyxy.tolist():
            if (box[2] - box[0]) * (box[3] - box[1]) < max_area:
                out_boxes.append(box)
    return out_boxes, results

def run_sam3_boxes(image_path, bboxes, conf=0.25, max_area_frac=0.9):
    """SAM3 semantic segmentation from BBOX EXAMPLES.

    Same SAM3SemanticPredictor as run_sam3_text, but driven by example
    boxes instead of a text concept: SAM3 finds and segments OTHER objects
    in the image that are similar to the examples (exemplar re-detection),
    not the fixed box regions. Reuses the cached predictor on purpose --
    SAM3 weights are multi-GB and a second instance would OOM an 8 GB box.

    Args:
      bboxes: list of [x1, y1, x2, y2] example boxes in image pixel coords.
    Returns:
      (out_boxes_xyxy_list, raw_results) - same shape as run_sam3_text;
      raw_results[0].masks.xyn aligns with raw_results[0].boxes.xyxy.
    """
    global _sam3_text_predictor, _sam3_last_mode
    if not bboxes:
        return [], None
    try:
        from ultralytics.models.sam.predict import SAM3SemanticPredictor
    except ImportError as e:
        raise RuntimeError(
            "SAM3 box-exemplar mode needs an ultralytics version that ships "
            f"SAM3SemanticPredictor. Import failed: {e}"
        ) from e
    if _sam3_text_predictor is None:
        try:
            _sam3_text_predictor = SAM3SemanticPredictor(overrides=dict(
                conf=conf, task="segment", mode="predict", imgsz=1024, model=weights_path("sam3.pt"),
                save=False, save_txt=False, save_crop=False,
            ))
        except FileNotFoundError as e:
            raise RuntimeError(
                "sam3.pt not found. Fetch it from "
                "https://huggingface.co/facebook/sam3 (gated; accept the "
                "license) and drop it next to the notebook in 'GUI and Pipeline/'."
            ) from e
    else:
        _sam3_text_predictor.args.conf = conf
    # Same source-cache reset as run_sam3_text -- defensive, since the box
    # path also reuses the same cached predictor instance.
    _sam3_text_predictor.dataset = None
    _sam3_text_predictor.set_image(image_path)
    # IMPORTANT: do NOT try to clear a leftover text concept here. An EMPTY
    # text list (set_prompts({"text": []})) makes ultralytics' SAM3 set_classes()
    # index text[0] -> IndexError and crashes the box run; reset_prompts() drops
    # model.names -> KeyError 'language_features'. So no safe "empty" state
    # exists via the current API. A leftover NON-EMPTY text concept from a prior
    # run_sam3_text does not crash (set_classes handles it), so we leave it: the
    # original, working behavior. A real text->boxes clear needs a verified SAM3
    # prompt API and must be tested live before re-adding (see [[memory]]).
    results = _sam3_text_predictor(bboxes=[[float(v) for v in b] for b in bboxes])
    _sam3_last_mode = "boxes"
    out_boxes = []
    if results and getattr(results[0], "boxes", None) is not None:
        shape = results[0].orig_shape
        max_area = shape[0] * shape[1] * max_area_frac
        for box in results[0].boxes.xyxy.tolist():
            if (box[2] - box[0]) * (box[3] - box[1]) < max_area:
                out_boxes.append(box)
    return out_boxes, results

def release_sam3_text_predictor():
    """Drop the cached SAM3 semantic predictor (the run_sam3_text/run_sam3_boxes
    singleton) and return its multi-GB weights to the OS. Call this when the
    active detector is no longer SAM3 so it does not sit resident behind another
    pipeline. Safe to call when nothing is loaded (no-op)."""
    global _sam3_text_predictor, _sam3_last_mode
    if _sam3_text_predictor is None:
        return False
    del _sam3_text_predictor
    _sam3_text_predictor = None
    _sam3_last_mode = None
    import gc
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                and hasattr(torch.mps, "empty_cache")):
            torch.mps.empty_cache()
    except Exception:
        pass
    return True

def sam3_semantic_loaded():
    """True when the cached SAM3 semantic predictor is resident. The GUI's
    memory-budget accounting reads this instead of poking at module globals."""
    return _sam3_text_predictor is not None


def _box_iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = ua + ub - inter
    return inter / denom if denom > 0 else 0.0


# How far (as a fraction of the box's own width/height) a mask may extend past
# its prompt box before it is clipped. A segmenter answering "what object is in
# this box" legitimately spills a little past the box edge; whole-plant concept
# spill is what this kills.
_SEG_BOX_PAD_FRAC = 0.15

def segment_with_boxes(sam_model, image_path, boxes, conf=1e-6):
    """Segment `boxes` with an interactive SAM2/SAM3 model, returning results
    whose masks are EXACTLY one-per-box and index-aligned with `boxes`.

    The GUI pairs segmenter masks with detector boxes BY INDEX. Two SAM
    behaviors silently break that pairing when the model is called raw:

      1. Ultralytics' SAM postprocess drops any mask whose quality score falls
         below `conf` (default 0.25). SAM3's interactive head scores weak
         prompts low, so a middle box's mask vanishes and every later mask
         shifts onto the WRONG box -- annotations appear on objects that were
         never detections.
      2. SAM3 is concept-driven: a box prompt can come back as a mask spanning
         the whole plant/cluster instead of the boxed object, producing giant
         segments unrelated to the box.

    Fix: run at a near-zero conf so nothing is dropped, then rebuild the masks
    tensor ourselves -- match each returned mask to the prompt box it overlaps
    best, clip it to the box padded by _SEG_BOX_PAD_FRAC, and leave an empty
    mask (no polygon) at any index whose box got nothing usable. Downstream
    code already skips empty masks per-index, so alignment survives.

    Returns the (mutated) ultralytics results list, or None on no output.
    """
    if not boxes:
        return None
    results = sam_model(image_path, bboxes=boxes, conf=conf, verbose=False, save=False)
    if not results or getattr(results[0], "masks", None) is None:
        return results
    data = results[0].masks.data  # (M, H, W) bool/float tensor at orig size
    if data is None or data.shape[0] == 0:
        return results
    mh, mw = data.shape[-2:]
    n = len(boxes)

    # Bounding box of each returned mask, in mask-pixel coords.
    mask_boxes = []
    for i in range(data.shape[0]):
        m = data[i]
        ys, xs = torch.where(m > 0.5)
        if len(xs) == 0:
            mask_boxes.append(None)
            continue
        mask_boxes.append([float(xs.min()), float(ys.min()),
                           float(xs.max()) + 1, float(ys.max()) + 1])

    # Greedy one-to-one match: best IoU(mask bbox, prompt box) first. Ties
    # prefer the identity pairing (mask i -> box i) because ultralytics keeps
    # surviving masks in prompt order -- a giant spill mask that covers several
    # boxes equally must stay on its own box, not drift to another. A mask
    # that overlaps no prompt box at all is discarded -- that is exactly the
    # "segment around something that was never a detection" failure.
    pairs = []
    for i, mb in enumerate(mask_boxes):
        if mb is None:
            continue
        for j, pb in enumerate(boxes):
            iou = _box_iou_xyxy(mb, [float(v) for v in pb])
            if iou > 0.0:
                pairs.append((-iou, 0 if i == j else 1, i, j))
    pairs.sort()
    mask_for_box = {}
    used_masks = set()
    for _neg_iou, _pref, i, j in pairs:
        if i in used_masks or j in mask_for_box:
            continue
        used_masks.add(i)
        mask_for_box[j] = i

    aligned = torch.zeros((n, mh, mw), dtype=torch.bool, device=data.device)
    for j, i in mask_for_box.items():
        x1, y1, x2, y2 = [float(v) for v in boxes[j]]
        pad_x = (x2 - x1) * _SEG_BOX_PAD_FRAC
        pad_y = (y2 - y1) * _SEG_BOX_PAD_FRAC
        cx1 = max(0, int(x1 - pad_x)); cy1 = max(0, int(y1 - pad_y))
        cx2 = min(mw, int(x2 + pad_x) + 1); cy2 = min(mh, int(y2 + pad_y) + 1)
        aligned[j, cy1:cy2, cx1:cx2] = data[i, cy1:cy2, cx1:cx2] > 0.5

    # Rebuild boxes alongside so results stays self-consistent: prompt box
    # coords with the matched mask's score (1.0 when the score is unknown).
    old_boxes = results[0].boxes
    scores = {}
    if old_boxes is not None and old_boxes.data is not None and old_boxes.data.shape[1] >= 5:
        for j, i in mask_for_box.items():
            if i < old_boxes.data.shape[0]:
                scores[j] = float(old_boxes.data[i, 4])
    box_rows = torch.tensor(
        [[float(b[0]), float(b[1]), float(b[2]), float(b[3]), scores.get(j, 1.0), 0.0]
         for j, b in enumerate(boxes)], dtype=torch.float32)
    results[0].update(boxes=box_rows, masks=aligned.float())
    return results
