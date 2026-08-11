"""Convert a saved AutoAnnotate output folder into a single COCO JSON file.

The app writes YOLO format: one .txt per image under boxes/ (cls cx cy w h,
normalized) and segments/ (cls x y x y ..., normalized polygon). That is what
YOLO training wants, but most other tooling, and every dataset viewer worth
using, speaks COCO. This module is the bridge, and it reads only what is
already on disk, so it can be run long after the annotation session.

Run it from the repo root:

    python -m autoannotate.coco <images_dir> <output_dir>
    python -m autoannotate.coco <images_dir> <output_dir> --source boxes
    python -m autoannotate.coco <images_dir> <output_dir> -o dataset.json

Two conversions are worth stating explicitly, because they are where COCO and
YOLO disagree and where a silently wrong export comes from:

* COCO category ids conventionally start at 1, YOLO class ids start at 0, so a
  category id here is always the YOLO id plus one. The YOLO id is preserved in
  each category's "yolo_id" field so the mapping stays auditable.
* COCO stores absolute pixels with a top-left origin: a box is [x, y, w, h]
  rather than YOLO's normalized centre point.
"""
import argparse
import json
import os
import re
import sys

from .imageio import imread_unicode

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def load_class_names(output_dir):
    """Read the id -> name table from <output_dir>/class_colors.txt.

    Returns a list indexed by class id, or None when the file is absent (older
    runs predate it, and a hand-assembled folder may never have had one). The
    file is a column-aligned table under a single commented header line, so
    fields are split on runs of two or more spaces: a class named "green berry"
    has to survive, and splitting on any whitespace would truncate it to
    "green".
    """
    path = os.path.join(output_dir, "class_colors.txt")
    if not os.path.exists(path):
        return None
    by_id = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) < 2:
                continue
            try:
                by_id[int(parts[0])] = parts[1]
            except ValueError:
                continue
    if not by_id:
        return None
    return [by_id.get(i, f"class_{i}") for i in range(max(by_id) + 1)]


def image_size(path):
    """Return (width, height) for an image, or None if it cannot be read.

    PIL is tried first because opening an image does not decode it: only the
    header is read, which matters when a folder holds hundreds of photos and
    all that is wanted is the dimensions. imread_unicode is the fallback rather
    than the first choice for the same reason, but it is kept because it is the
    reader the rest of the project trusts on Windows paths.
    """
    try:
        from PIL import Image
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        pass
    img = imread_unicode(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    return int(w), int(h)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def parse_box_file(path, width, height):
    """YOLO box rows -> [(cls, x, y, w, h)] in absolute pixels.

    Malformed rows are skipped rather than raised on, matching the reader the
    GUI already uses: one bad line in a label file must not cost the export of
    the other 300 images.
    """
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            x1 = _clamp((cx - bw / 2) * width, 0.0, width)
            y1 = _clamp((cy - bh / 2) * height, 0.0, height)
            x2 = _clamp((cx + bw / 2) * width, 0.0, width)
            y2 = _clamp((cy + bh / 2) * height, 0.0, height)
            if x2 <= x1 or y2 <= y1:
                continue
            out.append((cls, x1, y1, x2 - x1, y2 - y1))
    return out


def parse_segment_file(path, width, height):
    """YOLO polygon rows -> [(cls, [x1, y1, x2, y2, ...])] in absolute pixels."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            # cls plus at least 3 xy pairs; anything less is not a polygon.
            if len(parts) < 7:
                continue
            try:
                cls = int(float(parts[0]))
                coords = [float(v) for v in parts[1:]]
            except ValueError:
                continue
            if len(coords) % 2:
                coords = coords[:-1]
            if len(coords) < 6:
                continue
            flat = []
            for i in range(0, len(coords), 2):
                flat.append(_clamp(coords[i] * width, 0.0, width))
                flat.append(_clamp(coords[i + 1] * height, 0.0, height))
            out.append((cls, flat))
    return out


def polygon_area(flat):
    """Shoelace area of a flat [x1, y1, x2, y2, ...] polygon."""
    n = len(flat) // 2
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        x1, y1 = flat[2 * i], flat[2 * i + 1]
        j = (i + 1) % n
        x2, y2 = flat[2 * j], flat[2 * j + 1]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def polygon_bbox(flat):
    xs = flat[0::2]
    ys = flat[1::2]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return x1, y1, x2 - x1, y2 - y1


def list_images(images_dir):
    return sorted(f for f in os.listdir(images_dir)
                  if f.lower().endswith(IMAGE_EXTS)
                  and os.path.isfile(os.path.join(images_dir, f)))


def build_coco(images_dir, output_dir, source="auto", class_names=None):
    """Build the COCO dict for every image in `images_dir`.

    `source` picks which label folder an image's annotations come from:
      "auto"     segments/ when that image has a non-empty polygon file,
                 otherwise boxes/. Chosen per image, never mixed within one.
      "segments" polygons only.
      "boxes"    boxes only.

    The per-image choice is deliberate. A two-stage run saves both a box file
    and a segment file for the same image, and pairing them row by row across
    two files is exactly the index-alignment trap that has produced wrong masks
    in this project before. A polygon already carries its own bounding box, so
    taking segments whole and deriving the bbox from the polygon needs no
    pairing at all.

    Images with no labels are still emitted as image entries with zero
    annotations, because a COCO file that silently drops the negatives is a
    different dataset from the one on disk.

    Returns (coco_dict, stats).
    """
    if source not in ("auto", "segments", "boxes"):
        raise ValueError(f"build_coco: unknown source {source!r}")

    boxes_dir = os.path.join(output_dir, "boxes")
    segs_dir = os.path.join(output_dir, "segments")
    if class_names is None:
        class_names = load_class_names(output_dir)

    images, annotations = [], []
    stats = {"images": 0, "unreadable": 0, "from_segments": 0, "from_boxes": 0,
             "empty": 0, "annotations": 0}
    seen_class_ids = set()
    ann_id = 1

    for image_id, fname in enumerate(list_images(images_dir), start=1):
        path = os.path.join(images_dir, fname)
        size = image_size(path)
        if size is None:
            # Counted and reported, never guessed: without real dimensions every
            # normalized coordinate in that image's labels would be wrong.
            stats["unreadable"] += 1
            continue
        width, height = size
        stem = os.path.splitext(fname)[0]
        images.append({"id": image_id, "file_name": fname,
                       "width": width, "height": height})
        stats["images"] += 1

        polys, boxes = [], []
        if source in ("auto", "segments"):
            polys = parse_segment_file(os.path.join(segs_dir, f"{stem}.txt"),
                                       width, height)
        if source == "boxes" or (source == "auto" and not polys):
            boxes = parse_box_file(os.path.join(boxes_dir, f"{stem}.txt"),
                                   width, height)

        if polys:
            stats["from_segments"] += 1
            for cls, flat in polys:
                x, y, w, h = polygon_bbox(flat)
                seen_class_ids.add(cls)
                annotations.append({
                    "id": ann_id, "image_id": image_id,
                    "category_id": cls + 1,
                    "bbox": [round(v, 2) for v in (x, y, w, h)],
                    "area": round(polygon_area(flat), 2),
                    "segmentation": [[round(v, 2) for v in flat]],
                    "iscrowd": 0,
                })
                ann_id += 1
        elif boxes:
            stats["from_boxes"] += 1
            for cls, x, y, w, h in boxes:
                seen_class_ids.add(cls)
                annotations.append({
                    "id": ann_id, "image_id": image_id,
                    "category_id": cls + 1,
                    "bbox": [round(v, 2) for v in (x, y, w, h)],
                    "area": round(w * h, 2),
                    "segmentation": [],
                    "iscrowd": 0,
                })
                ann_id += 1
        else:
            stats["empty"] += 1

    stats["annotations"] = len(annotations)

    # Every class the labels actually used gets a category even when the name
    # table is missing or too short, otherwise the file references category ids
    # it never defines and strict COCO readers reject it outright.
    n_classes = max(len(class_names or []), (max(seen_class_ids) + 1) if seen_class_ids else 0)
    categories = []
    for cls in range(n_classes):
        if class_names and cls < len(class_names):
            name = class_names[cls]
        else:
            name = f"class_{cls}"
        categories.append({"id": cls + 1, "name": name,
                           "supercategory": "object", "yolo_id": cls})

    coco = {
        "info": {
            "description": "Exported from AutoAnnotate",
            "images_dir": os.path.abspath(images_dir),
            "labels_dir": os.path.abspath(output_dir),
            "label_source": source,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    return coco, stats


def export_coco(images_dir, output_dir, json_path=None, source="auto",
                class_names=None):
    """Build and write the COCO file. Returns (path, stats)."""
    coco, stats = build_coco(images_dir, output_dir, source=source,
                             class_names=class_names)
    if json_path is None:
        json_path = os.path.join(output_dir, "annotations_coco.json")
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    # Same write contract as the label writers: utf-8 with explicit newlines, so
    # a file exported on Windows is byte-identical to one exported on macOS, and
    # a non-ASCII class name does not depend on the machine's locale encoding.
    tmp = f"{json_path}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(coco, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    os.replace(tmp, json_path)
    return json_path, stats


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m autoannotate.coco",
        description="Convert an AutoAnnotate YOLO output folder to COCO JSON.")
    p.add_argument("images_dir", help="folder holding the original images")
    p.add_argument("output_dir",
                   help="the annotation output folder (the one containing boxes/ and segments/)")
    p.add_argument("-o", "--out", default=None,
                   help="destination .json (default: <output_dir>/annotations_coco.json)")
    p.add_argument("--source", choices=("auto", "segments", "boxes"), default="auto",
                   help="which labels to export (default: auto, per image)")
    args = p.parse_args(argv)

    if not os.path.isdir(args.images_dir):
        print(f"[coco] not a folder: {args.images_dir}")
        return 1
    if not os.path.isdir(args.output_dir):
        print(f"[coco] not a folder: {args.output_dir}")
        return 1
    has_labels = any(os.path.isdir(os.path.join(args.output_dir, d))
                     for d in ("boxes", "segments"))
    if not has_labels:
        print(f"[coco] {args.output_dir} has neither a boxes/ nor a segments/ folder. "
              f"Point this at the annotation output folder, not its parent.")
        return 1

    path, stats = export_coco(args.images_dir, args.output_dir,
                              json_path=args.out, source=args.source)
    print(f"[coco] wrote {path}")
    print(f"[coco] {stats['images']} images, {stats['annotations']} annotations "
          f"({stats['from_segments']} images from segments, "
          f"{stats['from_boxes']} from boxes, {stats['empty']} with no labels)")
    if stats["unreadable"]:
        print(f"[coco] {stats['unreadable']} image(s) could not be read and were left out")
    return 0


if __name__ == "__main__":
    sys.exit(main())
