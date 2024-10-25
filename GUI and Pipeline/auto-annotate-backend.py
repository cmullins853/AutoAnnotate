from groundingdino.util.inference import load_model, load_image, predict
import cv2
import torch
import csv
import os
from ultralytics import SAM
from pathlib import Path
import time as t
from PIL import Image, ImageDraw
import numpy as np


def clean_labels(boxes, max_area):
    clean_boxes = []
    box_list = boxes.tolist()
    for box in box_list:
        # if width * height < 0.9, add box to list.
        if (box[2] * box[3]) < max_area:
            clean_boxes.append(box)
    if len(clean_boxes) < 2:
        return boxes
    return torch.FloatTensor(clean_boxes)


def run_dino(img_path, prompt, box_threshold, text_threshold, model_size, max_area=0.8, save_dir="DINO-labels"):
    # choose swinb or swint
    if model_size == 'swint':
        config_path = r"GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py"
        checkpoint_path = r"GroundingDINO\weights\groundingdino_swint_ogc.pth"
    else:
        checkpoint_path = r"GroundingDINO\weights\groundingdino_swinb_cogcoor.pth"
        config_path = r"GroundingDINO\groundingdino\config\GroundingDINO_SwinB_cfg.py"

    model = load_model(config_path, checkpoint_path)

    image_source, image = load_image(img_path)

    boxes, accuracy, obj_name = predict(model=model,
                                        image=image,
                                        caption=prompt,
                                        box_threshold=box_threshold,
                                        text_threshold=text_threshold)

    # print(boxes, accuracy, obj_name)
    # Convert boxes from YOLOv8 format to xyxy
    img_height, img_width = cv2.imread(img_path).shape[:2]
    clean_boxes = clean_labels(boxes, max_area)
    absolute_boxes = [[(box[0] - (box[2] / 2)) * img_width,
                       (box[1] - (box[3] / 2)) * img_height,
                       (box[0] + (box[2] / 2)) * img_width,
                       (box[1] + (box[3] / 2)) * img_height] for box in clean_boxes.tolist()]
    # annotated_frame = annotate(image_source=image_source, boxes=clean_boxes, logits=accuracy, phrases=obj_name)
    # sv.plot_image(annotated_frame, (16,16))
    save_labels = True
    if save_labels:
        clean_boxes = clean_boxes.tolist()

        for x in clean_boxes:
            x.insert(0, 0)

        with open(f'{save_dir}/{os.path.splitext(os.path.basename(img_path))[0]}.txt', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ')
            writer.writerows(clean_boxes)
            # print("Labels saved in /DINO-labels")

    return absolute_boxes


def save_masks(sam_results, output_dir):
    segments = sam_results[0].masks.xyn
    with open(f"{Path(output_dir) / Path(sam_results[0].path).stem}.txt", "w") as f:
        for i in range(len(segments)):
            s = segments[i]
            if len(s) == 0:
                continue
            segment = map(str, segments[i].reshape(-1).tolist())
            f.write(f"0 " + " ".join(segment) + "\n")


def run(img_dir, output_dir, prompt, conf, box_threshold):
    sam_model = "sam2_t.pt"
    dino_model = "swint"
    start = t.time()
    for fname in os.listdir(img_dir):
        path = img_dir + "\\" + fname
        boxes = run_dino(dino_model, path, prompt, conf, 0.1, box_threshold)
        model = SAM(sam_model)
        sam_results = model(os.path.join(img_dir, fname), model=sam_model, bboxes=boxes, verbose=False)
        save_masks(sam_results, output_dir)
    print(f"Completed in: {t.time() - start} seconds, masks saved in {output_dir}")


def optimize_prompts(prompts_file, gt_path, img_dir, save_file, threshold):
    inf_path = fr"GroundingDINO\DINO-labels"

    with open(prompts_file, 'r') as file:
        result_dict = {}
        for x in file:
            result_dict[x.strip()] = {}

    # result_dict = dict.fromkeys(prompts,{})
    for prompt in result_dict.keys():
        print(f'Trying prompt: "{prompt}"')
        for fname in os.listdir(img_dir):
            box_threshold = 0.3
            text_threshold = 0.1
            model_size = 'swint'
            run_dino(os.path.join(img_dir, fname), prompt, box_threshold, text_threshold, model_size)

        metrics = process_files(inf_path, gt_path, threshold=threshold)

        result_dict[prompt]['iou_scores'] = np.mean(metrics['iou_scores'])

    results = sorted(list(result_dict.items()), key=lambda a: a[1]['iou_scores'], reverse=True)
    print(results)

    with open(save_file, 'w') as output:
        for prompt_stats in results:
            output.write(str(prompt_stats) + '\n')

    return results


def optimize_confidence(prompt, model_size, gt_path, img_dir, threshold):
    inf_path = r"C:\Users\Mechanized Systems\DataspellProjects\WSU_joint_data\Auto Annotate\GroundingDINO\DINO-labels"
    best_iou = 0
    best_conf = 0
    # number of decimal points in confidence
    final_precision = 5
    ubound = 0.9
    lbound = 0.0
    for precision in [x + 1 for x in range(final_precision)]:
        esc = 0
        for conf in [x / (10 ** precision) for x in
                     range(int(lbound * (10 ** precision)), int(ubound * (10 ** precision)))]:
            for fname in os.listdir(img_dir):
                prompt = prompt
                box_threshold = conf
                text_threshold = 0.01
                model_size = model_size
                run_dino(os.path.join(img_dir, fname), prompt, box_threshold, text_threshold, model_size)
            metrics = process_files(inf_path, gt_path, threshold)
            iou = np.mean(metrics['iou_scores'])
            if iou > best_iou:
                best_iou = iou
                best_conf = conf
            else:
                esc += 1
                if esc > 2 * precision:
                    break

            print(f"confidence: {conf}, IOU: {iou} (best: {best_iou})")
        print(f"Best IOU at p{precision} is {best_iou} with confidence = {best_conf}")
        lbound = max(0, best_conf - (1 / (10 ** precision)))
        ubound = min(0.9, best_conf + (1 / (10 ** precision)))

        if (best_conf > (0.2 * (10 ** precision))) and precision >= 2:
            print(f"Final Result: Best IOU is {best_iou} with confidence = {best_conf}")
            return best_iou, best_conf

    return best_iou, best_conf


def multi_optimize(img_dir, gt_label_dir, model_size, prompts, threshold=0.4):
    print("Be sure to change the category folders and model size in each function!")
    t.sleep(2)
    start = t.time()
    best_iou = 0
    best_prompt = ""
    best_conf = 0
    for prompt in prompts:
        print(f"Trying prompt: '{prompt}'")
        iou, conf = optimize_confidence(prompt, model_size, gt_label_dir, img_dir, threshold)
        if iou > best_iou:
            best_iou = iou
            best_conf = conf
            best_prompt = prompt
        print(f"So far: best prompt is '{best_prompt}', conf is {best_conf}, resulting in {best_iou} IOU)")
    print(f"\n\n\n\n\nFinal Result: best prompt is '{best_prompt}', conf is {best_conf}, resulting in {best_iou} IOU)")
    print(f"final time: {t.time() - start}")
    return {"prompt": best_prompt, "conf": best_conf, "iou": best_iou}


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


def read_and_draw_boxes(file_path, image_dim=(1280, 720)):
    boxes = []
    with open(file_path, 'r') as file:
        for line in file:
            class_id, x, y, width, height = map(float, line.strip().split())
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


def calculate_pixel_metrics(mask1, mask2):
    """
    Calculate IoU based on pixel values from two masks.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union


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
