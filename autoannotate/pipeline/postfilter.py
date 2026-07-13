"""Detection post-filters (no Qt, no models).

Currently: negative-prompt suppression. The GUI runs the detector ONCE over
positive + negative classes together, then this module removes every negative
detection and any positive detection that overlaps a negative one. Keeping it
model-free makes it cheap to unit-test headlessly.
"""


def _box_iou(a, b):
    """IoU of two absolute xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def suppress_by_neg_boxes(boxes, classes, polys, neg_boxes, iou_thresh=0.5):
    """Drop positive detections that overlap a negative-box match.

    `neg_boxes` are absolute-xyxy boxes where a negative appearance exemplar was
    found on this image (the GUI runs the red negative-box crops as visual
    exemplars, so these generalize across images). Any positive detection whose
    IoU with some negative box exceeds `iou_thresh` is removed; `classes` and
    `polys` are shrunk in lockstep. `polys` may be None (returned as None).
    Returns (boxes, classes, polys). No-op when there are no negative boxes.
    """
    boxes = list(boxes or [])
    classes = list(classes or [])
    if not boxes or not neg_boxes:
        return boxes, classes, polys
    out_boxes, out_classes = [], []
    out_polys = [] if polys is not None else None
    for i, b in enumerate(boxes):
        if any(_box_iou(b, nb) > iou_thresh for nb in neg_boxes):
            continue
        out_boxes.append(b)
        if i < len(classes):
            out_classes.append(int(classes[i]))
        if out_polys is not None:
            out_polys.append(polys[i])
    return out_boxes, out_classes, out_polys


def suppress_negative_hits(boxes, classes, polys, n_pos, iou_thresh=0.5):
    """Apply negative-prompt suppression to one image's aligned detections.

    `classes[i] < n_pos` marks a positive detection, `>= n_pos` a negative one
    (the caller appended the negative class names after the positives when it
    prompted the model). Every negative detection is dropped, and so is any
    positive detection whose IoU with some negative detection exceeds
    `iou_thresh`. Returns (boxes, classes, polys) still aligned; `polys` may
    be None and is then returned as None.
    """
    if not boxes:
        return list(boxes or []), list(classes or []), polys
    if n_pos <= 0:
        # No positive classes at all, so by the classes[i] < n_pos rule EVERY
        # detection is a negative one. Passing them through would bake the
        # negatives in as annotations, which is the exact opposite of the ask.
        return [], [], ([] if polys is not None else None)
    neg_boxes = [b for b, c in zip(boxes, classes) if int(c) >= n_pos]
    if not neg_boxes:
        return list(boxes), list(classes), polys
    out_boxes, out_classes, out_polys = [], [], [] if polys is not None else None
    for i, (b, c) in enumerate(zip(boxes, classes)):
        if int(c) >= n_pos:
            continue
        if any(_box_iou(b, nb) > iou_thresh for nb in neg_boxes):
            continue
        out_boxes.append(b)
        out_classes.append(int(c))
        if out_polys is not None:
            out_polys.append(polys[i])
    return out_boxes, out_classes, out_polys
