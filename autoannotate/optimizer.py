"""Prompt/confidence optimizers: score candidate DINO prompts against ground truth."""
import os
import time as t

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .config import BASE_DIR
from .pipeline.dino import run_dino_from_model

def prompt_optimizer(prompts_file, gt_path, img_path, save_file, threshold, DINO):
    print('entered prompt optimizer')
    # Ensure inference path exists
    inf_path = os.path.join(BASE_DIR, "DINO-labels")
    os.makedirs(inf_path, exist_ok=True)

    # Initialize result dictionary from prompt file
    with open(prompts_file, 'r') as file:
        result_dict = {x.strip(): {} for x in file}

    # Process each prompt
    for prompt in result_dict.keys():
        print(f'Trying prompt: "{prompt}"')

        # Run prediction and save labels
        run_dino_from_model(DINO, img_path, prompt, box_threshold=0.3, text_threshold=0.1, maxarea=threshold)

        # Process single predicted and ground truth file
        predicted_mask_file = os.path.join(inf_path, f"{os.path.splitext(os.path.basename(img_path))[0]}.txt")
        #print(predicted_mask_file)
        metrics = process_file(predicted_mask_file, gt_path, threshold)

        # Save the IoU score for the prompt
        result_dict[prompt]['iou_scores'] = np.mean(metrics['iou_scores'])

    # Sort and save results
    results = sorted(result_dict.items(), key=lambda a: a[1]['iou_scores'], reverse=True)
    print("Results:", results)

    with open(save_file, 'w') as output:
        for prompt_stats in results:
            output.write(str(prompt_stats) + '\n')
    return results

def process_mask_arrays(predicted_mask_array, ground_truth_mask_array):
    # Resize predicted mask to match the ground truth mask's dimensions
    if predicted_mask_array.shape != ground_truth_mask_array.shape:
        predicted_mask_array = cv2.resize(predicted_mask_array, (ground_truth_mask_array.shape[1], ground_truth_mask_array.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Initialize metrics dictionary
    metrics = {
        'iou_scores': [],
        #'pixel_accuracies': [],
        'precision_scores': [],
        'recall_scores': [],
        'f1_scores': [],
        'mcc_scores': [],
        'specificity_scores': []
    }

    # Convert masks to binary based on threshold
    _, predicted_mask_bin = cv2.threshold(predicted_mask_array, 127, 255, cv2.THRESH_BINARY)
    _, ground_truth_mask_bin = cv2.threshold(ground_truth_mask_array, 127, 255, cv2.THRESH_BINARY)

    # Normalize binary masks for calculation
    predicted_mask_bin = predicted_mask_bin / 255
    ground_truth_mask_bin = ground_truth_mask_bin / 255

    # Calculate true positives, true negatives, false positives, and false negatives
    tp = np.float64(np.sum(np.logical_and(predicted_mask_bin == 1, ground_truth_mask_bin == 1)))
    tn = np.float64(np.sum(np.logical_and(predicted_mask_bin == 0, ground_truth_mask_bin == 0)))
    fp = np.float64(np.sum(np.logical_and(predicted_mask_bin == 1, ground_truth_mask_bin == 0)))
    fn = np.float64(np.sum(np.logical_and(predicted_mask_bin == 0, ground_truth_mask_bin == 1)))

    # Calculate IoU and pixel accuracy
    intersection = np.logical_and(predicted_mask_bin, ground_truth_mask_bin)
    union = np.logical_or(predicted_mask_bin, ground_truth_mask_bin)
    metrics['iou_scores'].append(np.sum(intersection) / np.sum(union))
    #metrics['pixel_accuracies'].append(pixel_accuracy(predicted_mask_bin, ground_truth_mask_bin))

    # Calculate precision, recall, f1-score, MCC, and specificity
    precision, recall, f1, mcc, specificity = calculate_metrics(tp, fp, fn, tn)
    metrics['precision_scores'].append(precision)
    metrics['recall_scores'].append(recall)
    metrics['f1_scores'].append(f1)
    metrics['mcc_scores'].append(mcc)
    metrics['specificity_scores'].append(specificity)

    return metrics

def draw_boxes(boxes, image_dim=(1280, 720)):
    """
    Draw bounding boxes directly from a list of absolute boxes.

    Parameters:
    boxes (list): List of absolute box coordinates in xyxy format.
    image_dim (tuple): Dimensions of the output image (width, height).

    Returns:
    np.array: Binary image with boxes drawn.
    """
    # Create a blank image to draw the boxes
    image = Image.new('L', image_dim, 0)
    draw = ImageDraw.Draw(image)

    # Draw each box on the image
    for box in boxes:
        draw.rectangle(box, fill=255)

    return np.array(image, dtype=np.uint8)

def confidence_optimizer(prompt, DINO, gt_path, img_path, threshold):
    inf_path = os.path.join(BASE_DIR, "DINO-labels")
    os.makedirs(inf_path, exist_ok=True)

    best_iou = 0
    best_conf = 0

    image = cv2.imread(img_path)
    shape = image.shape

    # Step 1: Precision 1 sweep (coarse) from 0.0 to 0.9 in steps of 0.1
    for conf in np.arange(0.0, 0.91, 0.1):
        box_threshold = conf
        text_threshold = 0.1
        boxes = run_dino_from_model(DINO, img_path, prompt, box_threshold, text_threshold, maxarea=threshold)
        pred_masks = draw_boxes(boxes, (shape[1], shape[0]))
        gt_masks = read_and_draw_boxes_from_file(gt_path)

        metrics = process_mask_arrays(pred_masks, gt_masks)
        iou = np.mean(metrics['iou_scores'])
        print("P1")
        print(f"[Precision 1] Confidence: {conf:.1f}, IoU: {iou:.4f}")

        if iou > best_iou:
            best_iou = iou
            best_conf = conf

    print(f"Best from Precision 1: Confidence = {best_conf:.1f}, IoU = {best_iou:.4f}")

    # Step 2: Precision 2 sweep from (best_conf - 0.1) to (best_conf + 0.1) in steps of 0.01
    lower = best_conf - 0.1
    upper = best_conf + 0.1
    step = 0.01

    for conf in np.arange(lower, upper + step, step):
        box_threshold = conf
        text_threshold = 0.01
        boxes = run_dino_from_model(DINO, img_path, prompt, box_threshold, text_threshold, maxarea=threshold)
        pred_masks = draw_boxes(boxes, (shape[1], shape[0]))
        gt_masks = read_and_draw_boxes_from_file(gt_path)

        metrics = process_mask_arrays(pred_masks, gt_masks)
        iou = np.mean(metrics['iou_scores'])
        print('P2')
        print(f"[Precision 2] Confidence: {conf:.2f}, IoU: {iou:.4f}")

        if iou > best_iou:
            best_iou = iou
            best_conf = conf

    print(f"Final Best: Confidence = {best_conf:.2f}, IoU = {best_iou:.4f}")
    return best_iou, best_conf


def read_and_draw_boxes_from_file(file_path, image_dim=(1280, 720)):
    boxes = []
    with open(file_path, 'r') as file:
        for line in file:
            class_id, x, y, width, height = map(float, line.strip().split())
            x1 = (x-(width/2))*image_dim[0]
            x2 = (x+(width/2))*image_dim[0]
            y1 = (y-(height/2))*image_dim[1]
            y2 = (y+(height/2))*image_dim[1]
            boxes.append([x1, y1, x2, y2])
    image = Image.new('L', image_dim, 0)
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box, fill=255)
        #draw.rectangle([1,1,20,20], fill=255)
    #image.save("test.jpg")
    return np.array(image, dtype=np.uint8)


def process_file(predicted_mask_file, ground_truth_mask_file, threshold):
    # Initialize metrics dictionary
    metrics = {
        'iou_scores': [],
        'precision_scores': [],
        'recall_scores': [],
        'f1_scores': [],
        'mcc_scores': [],
        'specificity_scores': []
    }

    # Preprocess predicted mask
    clean_labels_from_file(predicted_mask_file, threshold)
    predicted_mask = read_and_draw_boxes_from_file(predicted_mask_file)
    ground_truth_mask = read_and_draw_boxes_from_file(ground_truth_mask_file)

    # Convert masks to binary
    _, predicted_mask_bin = cv2.threshold(predicted_mask, 127, 255, cv2.THRESH_BINARY)
    _, ground_truth_mask_bin = cv2.threshold(ground_truth_mask, 127, 255, cv2.THRESH_BINARY)

    predicted_mask_bin = predicted_mask_bin / 255
    ground_truth_mask_bin = ground_truth_mask_bin / 255

    # Calculate true positives, true negatives, false positives, and false negatives
    tp = np.float64(np.sum(np.logical_and(predicted_mask_bin == 1, ground_truth_mask_bin == 1)))
    tn = np.float64(np.sum(np.logical_and(predicted_mask_bin == 0, ground_truth_mask_bin == 0)))
    fp = np.float64(np.sum(np.logical_and(predicted_mask_bin == 1, ground_truth_mask_bin == 0)))
    fn = np.float64(np.sum(np.logical_and(predicted_mask_bin == 0, ground_truth_mask_bin == 1)))

    # Calculate metrics
    intersection = np.logical_and(predicted_mask_bin, ground_truth_mask_bin)
    union = np.logical_or(predicted_mask_bin, ground_truth_mask_bin)
    metrics['iou_scores'].append(np.sum(intersection) / np.sum(union))
    # Calculate precision, recall, f1-score, MCC, and specificity
    precision, recall, f1, mcc, specificity = calculate_metrics(tp, fp, fn, tn)
    metrics['precision_scores'].append(precision)
    metrics['recall_scores'].append(recall)
    metrics['f1_scores'].append(f1)
    metrics['mcc_scores'].append(mcc)
    metrics['specificity_scores'].append(specificity)
    #print(metrics['iou_scores'])
    return metrics


def multi_optimizer(img_dir, gt_label_dir, DINO, prompts, threshold=0.9, callback=None):
    start = t.time()
    best_iou = 0
    best_prompt = ""
    best_conf = 0

    for i, prompt in enumerate(prompts):
        if callback:
            callback(prompt, i, len(prompts))
        iou, conf = confidence_optimizer(prompt, DINO, gt_label_dir, img_dir, threshold)
        if iou > best_iou:
            best_iou = iou
            best_conf = conf
            best_prompt = prompt

    print(f"\nFinal Best: prompt = '{best_prompt}', conf = {best_conf}, IOU = {best_iou}")
    print(f"final time: {t.time() - start}")
    return best_prompt, best_conf

def calculate_metrics(tp, fp, fn, tn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) \
        if np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) > 0 else 0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0
    return precision, recall, f1, mcc, specificity

def pixel_accuracy(predicted, ground_truth):
    correct = np.sum(predicted == ground_truth)
    total = predicted.shape[0] * predicted.shape[1]
    return correct / total

def read_and_draw_boxes(results, image_dim=(1280, 720)):
    boxes = results.boxes
    for box in boxes:
        class_id, x, y, width, height = map(float, box.strip().split())
        x1 = (x - (width / 2)) * image_dim[0]
        x2 = (x + (width / 2)) * image_dim[0]
        y1 = (y - (height / 2)) * image_dim[1]
        y2 = (y + (height / 2)) * image_dim[1]
        boxes.append([x1, y1, x2, y2])
    image = Image.new('L', image_dim, 0)
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box, fill=255)
        # draw.rectangle([1,1,20,20], fill=255)
    image.save("test.jpg")
    return np.array(image, dtype=np.uint8)


def clean_labels_from_file(file_path, cleaning_threshold=0.6):
    # Read the file and check if it has more than one line
    with open(file_path, 'r') as f:
        lines = f.readlines()

    if len(lines) > 1:
        accepted_lines = []

        # Process each line
        for line in lines:
            class_id, x, y, width, height = map(float, line.strip().split())
            # if width * height < 0.9:
            if (width * height) < cleaning_threshold:
                accepted_lines.append(line)

        # Overwrite the file with accepted lines
        with open(file_path, 'w') as f:
            if len(accepted_lines) > 0:
                for line in accepted_lines:
                    f.write(line)

def process_files(predicted_mask_dir, ground_truth_mask_dir, threshold):
    predicted_files = os.listdir(ground_truth_mask_dir)
    metrics = {
        'iou_scores': [],
        'pixel_accuracies': [],
        'precision_scores': [],
        'recall_scores': [],
        'f1_scores': [],
        'mcc_scores': [],
        'specificity_scores': []
    }

    for fname in predicted_files:
        predicted_mask_path = os.path.join(predicted_mask_dir, fname)
        ground_truth_mask_path = os.path.join(ground_truth_mask_dir, os.path.splitext(fname)[0] + '.txt')

        if not os.path.exists(ground_truth_mask_path):
            metrics['iou_scores'].append(0)
            metrics['pixel_accuracies'].append(0)
            metrics['precision_scores'].append(0)
            metrics['recall_scores'].append(0)
            metrics['f1_scores'].append(0)
            metrics['mcc_scores'].append(0)
            metrics['specificity_scores'].append(0)
            continue

        clean_labels_from_file(predicted_mask_path, threshold)
        predicted_mask = read_and_draw_boxes(predicted_mask_path)
        ground_truth_mask = read_and_draw_boxes(ground_truth_mask_path)

        common_height, common_width = 1280, 720  # or any other desired size

        predicted_mask = cv2.resize(predicted_mask, (common_width, common_height))

        ground_truth_mask = cv2.resize(ground_truth_mask, (common_width, common_height))

        _, predicted_mask_bin = cv2.threshold(predicted_mask, 127, 255, cv2.THRESH_BINARY)
        _, ground_truth_mask_bin = cv2.threshold(ground_truth_mask, 127, 255, cv2.THRESH_BINARY)

        predicted_mask_bin = predicted_mask_bin / 255
        ground_truth_mask_bin = ground_truth_mask_bin / 255
        tp = np.float64(np.sum(np.logical_and(predicted_mask_bin == 1, ground_truth_mask_bin == 1)))
        tn = np.float64(np.sum(np.logical_and(predicted_mask_bin == 0, ground_truth_mask_bin == 0)))
        fp = np.float64(np.sum(np.logical_and(predicted_mask_bin == 1, ground_truth_mask_bin == 0)))
        fn = np.float64(np.sum(np.logical_and(predicted_mask_bin == 0, ground_truth_mask_bin == 1)))

        intersection = np.logical_and(predicted_mask_bin, ground_truth_mask_bin)
        union = np.logical_or(predicted_mask_bin, ground_truth_mask_bin)
        metrics['iou_scores'].append(np.sum(intersection) / np.sum(union))
        metrics['pixel_accuracies'].append(pixel_accuracy(predicted_mask_bin, ground_truth_mask_bin))
        precision, recall, f1, mcc, specificity = calculate_metrics(tp, fp, fn, tn)
        metrics['precision_scores'].append(precision)
        metrics['recall_scores'].append(recall)
        metrics['f1_scores'].append(f1)
        metrics['mcc_scores'].append(mcc)
        metrics['specificity_scores'].append(specificity)

    return metrics
