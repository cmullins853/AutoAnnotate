"""YOLO label I/O and mask-to-polygon conversion (no Qt, no models)."""
import os
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon

def save_boxes_yolo(boxes_xyxy, image_path, save_dir):
    """Overwrite the YOLO label file with the given absolute xyxy boxes (post-edit truth).
    Use this from the GUI after the user has finalized edits so the saved file matches the screen."""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    stem = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/{stem}.txt', 'w') as f:
        for x1, y1, x2, y2 in boxes_xyxy:
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')

def verify_boxes_round_trip(boxes_xyxy, image_path, save_dir, tol_px=1.5):
    """Read back the saved YOLO file and confirm every box matches `boxes_xyxy` within `tol_px`.
    Returns (ok: bool, max_err_px: float). Used by the output fact-check audit."""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    stem = os.path.splitext(os.path.basename(image_path))[0]
    path = f'{save_dir}/{stem}.txt'
    if not os.path.exists(path):
        return False, float('inf')
    loaded = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = parts[:5]
            cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
            x1 = (cx - bw/2) * w
            y1 = (cy - bh/2) * h
            x2 = (cx + bw/2) * w
            y2 = (cy + bh/2) * h
            loaded.append((x1, y1, x2, y2))
    if len(loaded) != len(boxes_xyxy):
        return False, float('inf')
    max_err = 0.0
    for src, ld in zip(boxes_xyxy, loaded):
        for a, b in zip(src, ld):
            max_err = max(max_err, abs(a - b))
    return max_err <= tol_px, max_err

def result_clean_polys(result):
    """Per-detection polygon as a SINGLE largest contour, in normalized
    (0-1) xy coords, aligned 1:1 with result.boxes / result.masks.

    Ultralytics' masks.xyn concatenates every contour of a multi-blob mask
    into one point list, so a mask with two separate pieces renders (and
    saves) as one polygon with a straight line joining the pieces. Deriving
    the polygon straight from the raw binary mask and keeping only the
    largest connected contour avoids that. An entry is None when the mask
    is empty or degenerate."""
    polys = []
    masks = getattr(result, "masks", None)
    if masks is None or masks.data is None:
        return polys
    data = masks.data.cpu().numpy()
    for m in data:
        binary = (m > 0.5).astype(np.uint8)
        mh, mw = binary.shape[:2]
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts or mw == 0 or mh == 0:
            polys.append(None)
            continue
        c = max(cnts, key=cv2.contourArea)
        if len(c) < 3 or cv2.contourArea(c) < 1.0:
            polys.append(None)
            continue
        polys.append([[float(p[0][0]) / mw, float(p[0][1]) / mh] for p in c])
    return polys

def save_masks(sam_results, output_dir, image_path):
    # Use single-largest-contour polygons (not masks.xyn) so a multi-blob
    # mask isn't written as one polygon with a stray connecting line.
    # Name the file after the REAL image (image_path), never sam_results[0].path
    # -- SAM3 box-carry/ROI runs segment a temp composite image, whose path is a
    # random tempfile name; deriving the stem from it produced orphan tmp*.txt
    # files that never overwrote and piled up in segments/. Mirrors save_boxes_yolo.
    if not sam_results:
        return
    segments = result_clean_polys(sam_results[0])
    save_polys_yolo(segments, output_dir, image_path)

def save_polys_yolo(segments, output_dir, image_path):
    # Shared YOLO-seg polygon writer: one '0 x y x y ...' line per polygon,
    # named after the REAL image (mirrors save_boxes_yolo). save_masks feeds
    # it polys derived from a SAM results object; the one-shot detector paths
    # (YOLOE-seg / SAM3 standalone) feed their already max-area + NMS filtered,
    # box-aligned polys directly, so segments/ matches boxes/ instead of
    # re-deriving every raw detection (giant leaf masks, dropped dup boxes).
    stem = os.path.splitext(os.path.basename(image_path))[0]
    with open(f"{Path(output_dir) / stem}.txt", "w") as f:
        for s in segments:
            if not s or len(s) < 3:
                continue
            if Polygon(s).area > 0.001:  # 0.05 was too large for small objects like blueberries
                flat = " ".join(str(v) for pt in s for v in pt)
                f.write("0 " + flat + "\n")

def _mask_to_polys(result, min_area_frac=0.04):
    """Interactive-SAM extraction: return a LIST of crop-normalized polygons,
    ONE per significant connected blob of the (best) mask, largest first, with
    NO bridge between them; disconnected pieces stay SEPARATE masks. Tiny
    specks (< min_area_frac of the largest blob) are dropped. Returns [] if none.

    A negative point that splits one blob into two simply yields two contours
    here, so the split pieces become two separate masks on commit."""
    masks = getattr(result, "masks", None)
    if masks is None or masks.data is None:
        return []
    data = masks.data.cpu().numpy()
    if len(data) == 0:
        return []
    binary = (data[0] > 0.5).astype(np.uint8)
    mh, mw = binary.shape[:2]
    if mw == 0 or mh == 0:
        return []
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scored = [(cv2.contourArea(c), c) for c in cnts if len(c) >= 3]
    scored = [(a, c) for a, c in scored if a >= 1.0]
    if not scored:
        return []
    thresh = max(1.0, min_area_frac * max(a for a, _ in scored))
    scored = [(a, c) for a, c in scored if a >= thresh]
    scored.sort(key=lambda ac: ac[0], reverse=True)
    return [[[float(p[0][0]) / mw, float(p[0][1]) / mh] for p in c] for _a, c in scored]
