"""YOLO label I/O and mask-to-polygon conversion (no Qt, no models)."""
import os
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon

from ..imageio import imread_unicode
from ..palette import (class_color_image_rgb,
                       class_color_name, rgb_to_hex)

def _validate_class_ids(classes, n_items, fn_name, item_name):
    """Coerce `classes` to a list of ints aligned with `n_items`, or None.

    Both label writers truncate their output file the moment they open it, so a
    length mismatch or an unconvertible id has to be caught here, before the
    open: raising halfway through the write leaves the previous labels destroyed
    and the new ones incomplete. Returns None when `classes` is None (the
    all-zeros default).
    """
    if classes is None:
        return None
    classes = list(classes)
    if len(classes) != n_items:
        raise ValueError(
            f'{fn_name}: {n_items} {item_name} but {len(classes)} classes')
    try:
        ids = [int(c) for c in classes]
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{fn_name}: class ids must be integers, got {classes!r}') from exc
    # A YOLO class id indexes into the class-name table, so it cannot be negative: -1 is
    # the "no class" sentinel some detectors return, and exporting it produces a
    # label file no importer can read.
    bad = [c for c in ids if c < 0]
    if bad:
        raise ValueError(f'{fn_name}: class ids must be >= 0, got {bad!r}')
    return ids


def _atomic_write_lines(path, lines):
    """Write `lines` to `path` via a temp file + os.replace.

    The label writers used to open the destination in 'w' (which truncates) and
    format rows as they went, so anything that raised part-way through the loop,
    a malformed box, a non-numeric coordinate, left the previous labels gone and
    the new ones half-written. Every row is now rendered up front and the real
    file is only swapped in once the whole set is on disk: the label file is
    always either the complete old one or the complete new one.

    Written with an explicit newline="\\n" and utf-8: the default 'w' mode
    translates \\n to \\r\\n on Windows, so the same annotations saved on Windows
    and on macOS produced byte-different label files. os.replace is atomic and
    overwrites an existing destination on POSIX and Windows alike.
    """
    tmp = f'{path}.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(lines)
    os.replace(tmp, path)


def save_boxes_yolo(boxes_xyxy, image_path, save_dir, classes=None):
    """Overwrite the YOLO label file with the given absolute xyxy boxes (post-edit truth).
    Use this from the GUI after the user has finalized edits so the saved file matches the screen.
    `classes` is an optional per-box class-id list aligned with boxes_xyxy; omitted = all 0."""
    boxes_xyxy = list(boxes_xyxy)
    class_ids = _validate_class_ids(classes, len(boxes_xyxy), 'save_boxes_yolo', 'boxes')
    img = imread_unicode(image_path)
    if img is None:
        raise ValueError(f'save_boxes_yolo: cannot read image {image_path}')
    h, w = img.shape[:2]
    stem = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(save_dir, exist_ok=True)
    # Render every row BEFORE touching the file: a malformed box (wrong arity,
    # non-numeric) raises here, with the previous labels still intact on disk.
    lines = []
    for i, box in enumerate(boxes_xyxy):
        try:
            x1, y1, x2, y2 = (float(v) for v in box)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f'save_boxes_yolo: box {i} is not 4 numbers: {box!r}') from exc
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        cls = class_ids[i] if class_ids is not None else 0
        lines.append(f'{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')
    _atomic_write_lines(f'{save_dir}/{stem}.txt', lines)

def verify_boxes_round_trip(boxes_xyxy, image_path, save_dir, tol_px=1.5):
    """Read back the saved YOLO file and confirm every box matches `boxes_xyxy` within `tol_px`.
    Returns (ok: bool, max_err_px: float). Used by the output fact-check audit."""
    img = imread_unicode(image_path)
    if img is None:
        # An unreadable image is a FAILED verification, not a crash: this runs
        # inside the audit pass, which must survive one bad file.
        return False, float('inf')
    h, w = img.shape[:2]
    stem = os.path.splitext(os.path.basename(image_path))[0]
    path = f'{save_dir}/{stem}.txt'
    if not os.path.exists(path):
        return False, float('inf')
    loaded = []
    with open(path, encoding='utf-8') as f:
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

def save_masks(sam_results, output_dir, image_path, classes=None):
    # Use single-largest-contour polygons (not masks.xyn) so a multi-blob
    # mask isn't written as one polygon with a stray connecting line.
    # Name the file after the REAL image (image_path), never sam_results[0].path
    # -- SAM3 box-carry/ROI runs segment a temp composite image, whose path is a
    # random tempfile name; deriving the stem from it produced orphan tmp*.txt
    # files that never overwrote and piled up in segments/. Mirrors save_boxes_yolo.
    if not sam_results:
        return
    segments = result_clean_polys(sam_results[0])
    save_polys_yolo(segments, output_dir, image_path, classes=classes)

def save_polys_yolo(segments, output_dir, image_path, classes=None):
    # Shared YOLO-seg polygon writer: one 'cls x y x y ...' line per polygon,
    # named after the REAL image (mirrors save_boxes_yolo). save_masks feeds
    # it polys derived from a SAM results object; the one-shot detector paths
    # (YOLOE-seg / SAM3 standalone) feed their already max-area + NMS filtered,
    # box-aligned polys directly, so segments/ matches boxes/ instead of
    # re-deriving every raw detection (giant leaf masks, dropped dup boxes).
    # `classes` is aligned with `segments` BEFORE any skipping, so degenerate
    # entries (None / tiny) drop their class id in lockstep; omitted = all 0.
    segments = list(segments)
    # Same pre-write contract as save_boxes_yolo: validate, render every row, and
    # only then swap the file in, so a bad polygon cannot destroy the previous
    # segments and leave a half-written one behind.
    class_ids = _validate_class_ids(classes, len(segments), 'save_polys_yolo', 'segments')
    stem = os.path.splitext(os.path.basename(image_path))[0]
    # save_boxes_yolo creates its save_dir; this one did not, so a first-ever run
    # into a fresh segments/ folder died with FileNotFoundError.
    os.makedirs(output_dir, exist_ok=True)
    lines = []
    for i, s in enumerate(segments):
        if not s or len(s) < 3:
            continue
        if Polygon(s).area > 0.001:  # 0.05 was too large for small objects like blueberries
            flat = " ".join(str(v) for pt in s for v in pt)
            cls = class_ids[i] if class_ids is not None else 0
            lines.append(f"{cls} " + flat + "\n")
    _atomic_write_lines(f"{Path(output_dir) / stem}.txt", lines)

def _validate_class_names(names, fn_name):
    """Reject class names carrying a newline before anything is written.

    The class table is positional: row N IS class id N. A name containing a
    newline would occupy two lines and silently shift the id of every class
    after it, so the labels would read against the wrong names. Cheaper to
    refuse than to write a file that is quietly wrong.
    """
    for i, n in enumerate(names or []):
        if "\n" in str(n) or "\r" in str(n):
            raise ValueError(
                f'{fn_name}: class name {i} ({n!r}) contains a newline; '
                f'one class per line is what makes the class id meaningful')


def save_class_colors_txt(names, output_dir):
    """Write class_colors.txt: the id -> name -> colour table for a run.

    This is THE record of what each saved class id means; the old classes.txt
    duplicated the name column and was retired, so a stale one from an earlier
    run is removed here rather than left to drift out of sync with the labels.

    The quoted colours are the ones in the annotated_<model> review images that
    sit beside this file (class_color_image_rgb), which is also what the legend
    image uses. The file is just the aligned table, one row per class under a
    single commented column-header line; the user asked for the explanatory
    comment block to go, so keep it out. (Canvas-only provenance colours, the
    manual green and negative red, are documented in class_legend.png instead;
    they never appear in a saved review image.)

    Returns the path written, or None when `names` is empty.
    """
    if not names:
        return None
    _validate_class_names(names, 'save_class_colors_txt')
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "class_colors.txt")
    rows = []
    for i, n in enumerate(names):
        rgb = class_color_image_rgb(i)
        rows.append((str(i), n, class_color_name(i), rgb_to_hex(rgb),
                     ",".join(str(v) for v in rgb)))

    head = ("id", "name", "colour", "hex", "rgb")
    widths = [max(len(r[c]) for r in (head,) + tuple(rows)) for c in range(len(head))]

    def _line(cells):
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells)).rstrip()

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# " + _line(head) + "\n")
        for r in rows:
            f.write("  " + _line(r) + "\n")
    stale = os.path.join(output_dir, "classes.txt")
    try:
        if os.path.exists(stale):
            os.remove(stale)
    except OSError:
        pass
    return path

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
