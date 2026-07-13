"""YOLOE detection wrappers (text prompts and visual box prompts)."""
import os

from ..config import AUTOANNOTATE_DEBUG, weights_path

def load_yoloe(model_path="yoloe-11l-seg.pt"):
    """yoloe-11l-seg.pt supports text (set_classes) and visual prompting.
    yoloe-11l-seg-pf.pt is the prompt-free variant: fixed vocabulary, no set_classes."""
    from ultralytics import YOLOE
    if AUTOANNOTATE_DEBUG:
        print(f"[YOLOE-LOAD] load_yoloe('{model_path}')  [v2 hardened]")
    m = YOLOE(weights_path(model_path))
    try:
        head = m.model.model[-1]
        has_lrpc = hasattr(head, "lrpc")
    except Exception as e:
        head, has_lrpc = None, "?"
        print(f"[YOLOE-LOAD] could not introspect head: {e}")
    if AUTOANNOTATE_DEBUG:
        print(f"[YOLOE-LOAD] loaded {type(head).__name__ if head else '?'}, lrpc={has_lrpc}")
    wants_text_vis = "-pf" not in os.path.basename(model_path)
    if wants_text_vis and has_lrpc is True:
        raise RuntimeError(
            f"{model_path} loaded as a prompt-free model (has lrpc head), "
            "but the caller wanted text/visual prompting. The file on disk "
            "appears to be the -pf variant under the wrong name. Re-download "
            "yoloe-11l-seg.pt from the ultralytics release page."
        )
    return m

def run_yoloe_text(model, image_path, names, conf=0.05, max_area_frac=0.9):
    """Run YOLOE with text/class prompts. `names` is a list of class strings.
    Returns (xyxy boxes list, raw results)."""
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    if not names:
        # No class prompt -> nothing to detect. Feeding an empty token batch to
        # the CLIP text encoder crashes the YOLOE TorchScript with
        # "cannot reshape tensor of 0 elements into shape [0, 77, ...]".
        return [], None
    # Ultralytics quirk: the OUTER YOLOE.set_classes (yolo/model.py:316) guards
    # its work with
    #   if sorted(list(self.model.names.values())) != sorted(classes):
    # It compares names as SETS, so re-prompting with the same class names in a
    # different ORDER looks like "no change" and the whole body is skipped: the
    # model keeps its previous name order AND its previous embedding order. The
    # detections then come back indexed against the OLD order, so reordering the
    # prompt fields silently swaps every class id (wrong label, wrong colour,
    # wrong per-class threshold). That same line also crashes when names is a
    # list rather than a dict.
    #
    # The INNER YOLOEModel.set_classes (nn/tasks.py:1124) sets pe / nc / names
    # unconditionally, so drive it directly and skip the guard entirely.
    _inner = getattr(model, "model", None)
    if _inner is not None and isinstance(getattr(_inner, "names", None), list):
        _inner.names = {i: n for i, n in enumerate(_inner.names)}
    if _inner is not None and hasattr(_inner, "set_classes"):
        _inner.set_classes(names, model.get_text_pe(names))
        # Outer wrapper's own bookkeeping, normally done by YOLOE.set_classes.
        if getattr(model, "predictor", None) is not None:
            model.predictor.model.names = _inner.names
    else:
        model.set_classes(names, model.get_text_pe(names))
    # Defensive: also coerce the outer wrapper in case downstream code
    # resets it back to a list.
    if isinstance(getattr(model, "names", None), list):
        model.names = {i: n for i, n in enumerate(model.names)}
    # save=False/save_txt=False: keep ultralytics from dumping its own copy
    # of the prediction into runs/segment/predict. The GUI writes labels
    # only to the user's selected output folder.
    results = model.predict(image_path, conf=conf, imgsz=1036, verbose=False,
                            save=False, save_txt=False, save_crop=False)
    out_boxes = []
    for result in results:
        if result.boxes is None:   # no detections -> nothing to filter
            continue
        shape = result.orig_shape
        max_area = shape[0] * shape[1] * max_area_frac
        for box in result.boxes.xyxy.tolist():
            if (box[2] - box[0]) * (box[3] - box[1]) < max_area:
                out_boxes.append(box)
    return out_boxes, results

def run_yoloe_vis(model, image_path, visual_prompts, conf=0.05, max_area_frac=0.9, refer_image=None):
    """Run YOLOE with visual (bbox) prompts.
    `visual_prompts` = {'bboxes': np.ndarray(N,4) xyxy, 'cls': np.ndarray(N,) int32}
    `refer_image` (optional): path to a SEPARATE reference image the bboxes are
    drawn on. When given, YOLOE runs TRUE one-shot -- it learns the prompt
    objects' appearance from refer_image and finds similar objects in
    image_path. When None, the bboxes are taken to lie in image_path itself.
    Returns (xyxy boxes list, raw results)."""
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
    predict_kwargs = dict(
        source=image_path,
        visual_prompts=visual_prompts,
        predictor=YOLOEVPSegPredictor,
        conf=conf,
        imgsz=1036,
        verbose=False,
        # Don't let ultralytics save into runs/segment/predict.
        save=False,
        save_txt=False,
        save_crop=False,
    )
    if refer_image is not None:
        predict_kwargs["refer_image"] = refer_image
    results = model.predict(**predict_kwargs)
    out_boxes = []
    for result in results:
        if result.boxes is None:   # no detections -> nothing to filter
            continue
        shape = result.orig_shape
        max_area = shape[0] * shape[1] * max_area_frac
        for box in result.boxes.xyxy.tolist():
            if (box[2] - box[0]) * (box[3] - box[1]) < max_area:
                out_boxes.append(box)
    return out_boxes, results
