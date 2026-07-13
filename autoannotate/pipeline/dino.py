"""GroundingDINO detection wrappers and the DINO+SAM convenience runner."""
import csv
import os
import re
import time as t

from ..config import AUTOANNOTATE_DEBUG, BASE_DIR, GROUNDING_DINO_DIR

import cv2
import torch
from groundingdino.util.inference import load_model, load_image, predict

from ..imageio import imread_unicode
from .labels import save_masks
from .sam import load_sam

def clean_labels(boxes, max_area):
    clean_boxes = []
    box_list = boxes.tolist()
    for box in box_list:
        # if width * height < 0.9, add box to list.
        if (box[2] * box[3]) < max_area:
            clean_boxes.append(box)
    if len(clean_boxes) < 1:
        # Every box exceeded max_area, usually GroundingDINO's spurious
        # whole-image detection. Return EMPTY, not the unfiltered set: the
        # old `return boxes` handed the full-frame box right back, which is
        # the "one giant mask around the image border" bug on hard images.
        return boxes[0:0]
    return torch.FloatTensor(clean_boxes)

def load_dino_model(model_size='swint'):
    #choose swinb or swint
    if model_size == 'swint':
        config_path = os.path.join(GROUNDING_DINO_DIR, "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
        checkpoint_path = os.path.join(GROUNDING_DINO_DIR, "weights", "groundingdino_swint_ogc.pth")
    elif model_size == 'swinb':
        checkpoint_path = os.path.join(GROUNDING_DINO_DIR, "weights", "groundingdino_swinb_cogcoor.pth")
        config_path = os.path.join(GROUNDING_DINO_DIR, "groundingdino", "config", "GroundingDINO_SwinB_cfg.py")

    model = load_model(config_path, checkpoint_path)
    return model

# Set True only to calibrate the confidence slider. It runs a SECOND full DINO
# forward pass per image just to print raw scores, which DOUBLES inference time.
DINO_SCORE_DIAGNOSTICS = False

def run_dino_from_model(model, img_path, prompt, box_threshold, text_threshold, maxarea=0.7, save_dir=None,
                        class_names=None, return_classes=False, return_scores=False):
    # Default label dir is anchored to BASE_DIR (not the working directory)
    # so the app and the optimizer agree on it no matter where it launched.
    # `class_names` (optional) is the ordered class list parsed from the
    # prompt; with `return_classes=True` the return value becomes
    # (absolute_boxes, cls_ids) with cls_ids aligned to the boxes. The default
    # single-value return is unchanged for existing callers.
    # `return_scores=True` appends the per-box confidence scores, aligned the
    # same way, so the caller can re-filter per class after a pass that ran at
    # the loosest class threshold.
    if save_dir is None:
        save_dir = os.path.join(BASE_DIR, "DINO-labels")
        os.makedirs(save_dir, exist_ok=True)
    # Read the image FIRST. load_image() raises on a corrupt/missing/unsupported
    # file, which would kill the whole batch, so the skip-and-continue fallback
    # has to be reachable before it runs. The decoded array is kept for the
    # dimensions further down rather than decoding the same file a second time.
    _img = imread_unicode(img_path)
    if _img is None:
        print(f"[DINO] {os.path.basename(img_path)}: cv2 could not read the image "
              f"(corrupt/unsupported format); skipping it.")
        empty = [[]] * (1 + int(return_classes) + int(return_scores))
        return tuple(empty) if len(empty) > 1 else []
    image_source, image = load_image(img_path)

    # GroundingDINO expects a lowercase, period-terminated caption. Passing
    # the raw user text (mixed case, no period) hurts text-token matching
    # and is a known cause of the model latching onto background objects.
    caption = (prompt or "").strip().lower()
    if caption and not caption.endswith('.'):
        caption = caption + ' .'

    # Optional diagnostic: a SECOND full DINO forward pass at a near-zero
    # threshold, only to print top raw scores for slider calibration. OFF by
    # default because it doubles per-image inference time.
    if DINO_SCORE_DIAGNOSTICS:
        _, raw_scores, _ = predict(model=model, image=image, caption=caption, box_threshold=0.01, text_threshold=0.01)
        if len(raw_scores) > 0:
            top = sorted(raw_scores.tolist(), reverse=True)[:5]
            print(f"[SCORES] {os.path.basename(img_path)}: {[f'{s:.3f}' for s in top]}")
        else:
            print(f"[SCORES] {os.path.basename(img_path)}: no detections at all (check prompt)")

    boxes, accuracy, obj_name = predict(model = model, image = image, caption = caption, box_threshold = box_threshold, text_threshold = text_threshold)

    # Phrase filter: GroundingDINO assigns each box a phrase built from the
    # caption tokens it actually matched. Boxes that didn't match the prompt
    # (background leaves, etc.) come back with an EMPTY phrase, so drop those
    # always; that is the spurious-detection case behind "segments on leaves".
    # For multi-concept captions ("blueberry . leaf .") also drop boxes whose
    # phrase shares no token with the prompt, so concepts don't bleed. A
    # single-concept caption keeps every non-empty phrase (a partial wordpiece
    # phrase must not nuke a valid box).
    prompt_tokens = set(re.findall(r'[a-z0-9]+', caption))
    multi_concept = len(prompt_tokens) > 1

    # Per-class token sets for multi-class prompts: each kept box is assigned
    # the class whose name tokens overlap its matched phrase the most (tie ->
    # lowest class index; no overlap -> class 0). With class_names omitted or
    # a single class, everything maps to 0, exactly the old behavior.
    cls_tokens = None
    if class_names:
        cls_tokens = [set(re.findall(r'[a-z0-9]+', (n or '').strip().lower()))
                      for n in class_names]

    def _classify(ph_tokens):
        if not cls_tokens:
            return 0
        best_j, best_overlap = 0, 0
        for j, toks in enumerate(cls_tokens):
            overlap = len(ph_tokens & toks)
            if overlap > best_overlap:
                best_j, best_overlap = j, overlap
        return best_j

    keep_idx = []
    keep_cls = []
    keep_scores = []
    for i, ph in enumerate(obj_name):
        ph_tokens = set(re.findall(r'[a-z0-9]+', (ph or '').strip().lower()))
        if not ph_tokens:
            continue
        if multi_concept and not (ph_tokens & prompt_tokens):
            continue
        keep_idx.append(i)
        keep_cls.append(_classify(ph_tokens))
        try:
            keep_scores.append(float(accuracy[i]))
        except (TypeError, ValueError, IndexError):
            keep_scores.append(None)
    dropped = len(obj_name) - len(keep_idx)
    if dropped and AUTOANNOTATE_DEBUG:
        print(f"[DINO-FILTER] {os.path.basename(img_path)}: dropped {dropped} "
              f"box(es) with no prompt-matching phrase")
    boxes = boxes[keep_idx] if keep_idx else boxes[0:0]

    #Convert boxes from YOLOv8 format to xyxy
    img_height, img_width = _img.shape[:2]
    # Area filter by kept INDEX (not clean_labels, which rebuilds the tensor
    # without saying which rows survived) so cls_ids shrinks in lockstep.
    box_list = boxes.tolist()
    area_keep = [i for i, b in enumerate(box_list) if (b[2] * b[3]) < maxarea]
    clean_boxes = torch.FloatTensor([box_list[i] for i in area_keep]) if area_keep else boxes[0:0]
    cls_ids = [keep_cls[i] for i in area_keep] if keep_cls else []
    det_scores = [keep_scores[i] for i in area_keep] if keep_scores else []
    absolute_boxes = [[(box[0]-(box[2]/2))*img_width,
                       (box[1]-(box[3]/2))*img_height,
                       (box[0]+(box[2]/2))*img_width,
                       (box[1]+(box[3]/2))*img_height] for box in clean_boxes.tolist()]
    # Zero detections: say WHY (and optionally RESCUE), so an empty image is
    # never a silent mystery. One extra low-threshold pass distinguishes "the
    # model found NOTHING for this prompt" (prompt/quality mismatch -- no
    # threshold helps) from "it found weak candidates your slider filtered out"
    # (a low-quality image whose scores all sit below threshold). In the latter
    # case, if AUTOANNOTATE_DINO_RESCUE=1, keep the candidates above a small
    # floor so the user gets REVIEWABLE boxes instead of a blank. Runs ONLY on a
    # zero-result image, so normal runs are unaffected. Rescue is OFF by default
    # (env opt-in) so it never silently injects weak boxes into a batch run.
    if not absolute_boxes:
        base = os.path.basename(img_path)
        RESCUE_FLOOR = 0.05
        rescue_on = os.environ.get("AUTOANNOTATE_DINO_RESCUE", "0") not in ("0", "false", "False", "")
        try:
            _b, _s, _ph = predict(model=model, image=image, caption=caption,
                                  box_threshold=0.01, text_threshold=0.01)
            if len(_s) == 0:
                print(f"[DINO] {base}: 0 detections -- the model found NOTHING for "
                      f"'{caption.strip()}' even at threshold 0.01. This image is "
                      f"likely too low quality / the prompt doesn't fit it; no "
                      f"threshold will help. Try a more specific term (e.g. "
                      f"'blueberry'), enhance the image, or annotate it by hand.")
            else:
                scores = _s.tolist()
                top = sorted(scores, reverse=True)[:5]
                # Same phrase filter as the main pass, plus the rescue floor.
                rescued = []
                rescued_cls = []
                for i, sc in enumerate(scores):
                    if sc < RESCUE_FLOOR:
                        continue
                    ph = _ph[i] if i < len(_ph) else ''
                    ph_tokens = set(re.findall(r'[a-z0-9]+', (ph or '').strip().lower()))
                    if not ph_tokens:
                        continue
                    if multi_concept and not (ph_tokens & prompt_tokens):
                        continue
                    rescued.append(i)
                    rescued_cls.append(_classify(ph_tokens))
                if rescue_on and rescued:
                    r_list = _b[rescued].tolist()
                    r_keep = [i for i, b in enumerate(r_list) if (b[2] * b[3]) < maxarea]
                    clean_boxes = (torch.FloatTensor([r_list[i] for i in r_keep])
                                   if r_keep else _b[0:0])
                    cls_ids = [rescued_cls[i] for i in r_keep]
                    det_scores = [scores[rescued[i]] for i in r_keep]
                    absolute_boxes = [[(box[0]-(box[2]/2))*img_width,
                                       (box[1]-(box[3]/2))*img_height,
                                       (box[0]+(box[2]/2))*img_width,
                                       (box[1]+(box[3]/2))*img_height] for box in clean_boxes.tolist()]
                    print(f"[DINO] {base}: your box_threshold={box_threshold:.2f} gave 0, "
                          f"so RESCUED {len(absolute_boxes)} low-confidence box(es) at "
                          f"floor {RESCUE_FLOOR:.2f} (top scores {[f'{v:.3f}' for v in top]}). "
                          f"REVIEW them -- they are weaker than usual.")
                elif rescued:
                    print(f"[DINO] {base}: 0 detections KEPT, but the model DID find "
                          f"candidates -- top raw scores {[f'{v:.3f}' for v in top]} vs "
                          f"your box_threshold={box_threshold:.2f} (Mask slider sets "
                          f"text_threshold={text_threshold:.2f}). Lower the Detector "
                          f"confidence slider, or set AUTOANNOTATE_DINO_RESCUE=1 to "
                          f"auto-keep the ones above {RESCUE_FLOOR:.2f}.")
                else:
                    print(f"[DINO] {base}: 0 detections -- candidates exist but all "
                          f"below the {RESCUE_FLOOR:.2f} floor (top scores "
                          f"{[f'{v:.3f}' for v in top]}). Scores this low are usually "
                          f"noise; this image is too weak for '{caption.strip()}'.")
        except Exception as _e:
            print(f"[DINO] {base}: 0 detections (diagnostic probe failed: {_e})")

    save_labels = True
    if save_labels:
        clean_boxes = clean_boxes.tolist()
        for i, x in enumerate(clean_boxes):
            x.insert(0, cls_ids[i] if i < len(cls_ids) else 0)
        # newline='' is the csv module's contract; encoding and an explicit LF
        # lineterminator keep DINO labels byte-identical to the ones the other
        # writers produce (csv's default dialect emits CRLF on every platform).
        with open(f'{save_dir}/{os.path.splitext(os.path.basename(img_path))[0]}.txt', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ', lineterminator='\n')
            writer.writerows(clean_boxes)
    if return_scores and len(det_scores) != len(absolute_boxes):
        det_scores = [None] * len(absolute_boxes)
    if return_classes:
        if len(cls_ids) != len(absolute_boxes):
            cls_ids = [0] * len(absolute_boxes)
        if return_scores:
            return absolute_boxes, cls_ids, det_scores
        return absolute_boxes, cls_ids
    if return_scores:
        return absolute_boxes, det_scores
    return absolute_boxes

def run_image(DINO, img_dir, output_dir, prompt, conf, box_threshold, save_dir):
    sam_model = "sam2_t.pt"
    dino_model = "swint"
    start = t.time()
    fname = os.path.basename(img_dir)
    path = img_dir
    if not os.path.exists(save_dir):
        print(f"{save_dir} does not exist, creating")
        os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(output_dir):
        print(f"{output_dir} does not exist, creating")
        os.makedirs(output_dir, exist_ok=True)

    boxes = run_dino_from_model(DINO, img_dir, prompt, conf, 0.1, box_threshold, save_dir=save_dir)
    if not boxes:
        print(f"No detections for {fname}, skipping SAM.")
        return [], []
    model = load_sam(sam_model)
    sam_results = model(img_dir, bboxes=boxes, verbose=False)
    save_masks(sam_results, output_dir, img_dir)

    print(f"Completed in: {t.time() - start} seconds, masks saved in {output_dir}")
    return sam_results, boxes
