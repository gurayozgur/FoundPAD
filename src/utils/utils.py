import csv
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    result = f"all params: {all_param} || trainable params: {trainable_params} || trainable%: {100 * trainable_params / all_param}"
    return result


def get_file_count(directory):
    file_count = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        file_count += len(filenames)
    return file_count


def sort_directories_by_file_count(base_path):
    directories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    directories_file_counts = [(d, get_file_count(os.path.join(base_path, d))) for d in directories]
    directories_file_counts.sort(key=lambda x: x[1], reverse=True)
    return directories_file_counts


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def accuracy(output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_eer_threhold_cross_db(fpr, tpr, threshold):
    differ_tpr_fpr_1 = tpr + fpr - 1.0

    right_index = np.nanargmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    eer = fpr[right_index]

    return eer, best_th, right_index


def performances_cross_db(prediction_scores, gt_labels, pos_label=1, verbose=False):

    fpr, tpr, threshold = roc_curve(gt_labels, prediction_scores, pos_label=pos_label)

    val_eer, val_threshold, right_index = get_eer_threhold_cross_db(fpr, tpr, threshold)
    test_auc = auc(fpr, tpr)

    FRR = 1 - tpr  # FRR = 1 - TPR
    HTER = (fpr + FRR) / 2.0  # error recognition rate &  reject recognition rate

    if verbose is True:
        print(
            f"AUC@ROC is {test_auc}, HTER is {HTER[right_index]}, APCER: {fpr[right_index]}, BPCER: {FRR[right_index]}, EER is {val_eer}, TH is {val_threshold}"
        )

    return (
        test_auc,
        fpr[right_index],
        FRR[right_index],
        HTER[right_index],
        val_eer,
        val_threshold,
    )


def evalute_threshold_based(prediction_scores, gt_labels, threshold):
    data = [
        {"map_score": score, "label": label}
        for score, label in zip(prediction_scores, gt_labels)
    ]
    num_real = len([s for s in data if s["label"] == 1])
    num_fake = len([s for s in data if s["label"] == 0])

    type1 = len([s for s in data if s["map_score"] <= threshold and s["label"] == 1])
    type2 = len([s for s in data if s["map_score"] > threshold and s["label"] == 0])

    # test_threshold_ACC = 1-(type1 + type2) / count
    test_threshold_APCER = type2 / num_fake
    test_threshold_BPCER = type1 / num_real
    test_threshold_ACER = (test_threshold_APCER + test_threshold_BPCER) / 2.0

    return test_threshold_APCER, test_threshold_BPCER, test_threshold_ACER


def compute_video_score(video_ids, predictions, labels):

    predictions_dict, labels_dict = defaultdict(list), defaultdict(list)

    for i in range(len(video_ids)):
        video_key = video_ids[i]
        predictions_dict[video_key].append(predictions[i])
        labels_dict[video_key].append(labels[i])

    new_predictions, new_labels, new_video_ids = [], [], []

    for video_indx in list(set(video_ids)):
        new_video_ids.append(video_indx)
        scores = np.mean(predictions_dict[video_indx])

        label = labels_dict[video_indx][0]
        new_predictions.append(scores)
        new_labels.append(label)

    return new_predictions, new_labels, new_video_ids


def write_scores(img_paths, prediction_scores, gt_labels, output_path):
    """
    Write detection scores to CSV file
    Args:
        img_paths: List of image paths
        prediction_scores: List of prediction scores
        gt_labels: List of ground truth labels
        output_path: Path to save the CSV file
    """
    try:
        # Verify all inputs have same length
        lengths = [len(img_paths), len(prediction_scores), len(gt_labels)]
        if len(set(lengths)) != 1:
            print(
                f"Warning: Length mismatch - Paths: {lengths[0]}, Predictions: {lengths[1]}, Labels: {lengths[2]}"
            )
            length = min(lengths)
        else:
            length = lengths[0]

        # Prepare data for writing
        save_data = []
        for idx in range(length):
            save_data.append(
                {
                    "image_path": img_paths[idx],
                    "label": str(gt_labels[idx]).replace(" ", ""),
                    "prediction_score": prediction_scores[idx],
                }
            )

        # Write to CSV
        with open(output_path, mode="w", newline="") as csv_file:
            fieldnames = ["image_path", "label", "prediction_score"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for data in save_data:
                writer.writerow(data)

        print(f"Successfully saved prediction scores in {output_path}")

    except Exception as e:
        print(f"Error in write_scores: {str(e)}")
        raise

def write_video_scores(video_ids, prediction_scores, gt_labels, output_path):
    """
    Write video-level detection scores to CSV file
    Args:
        video_ids: List of video identifiers
        prediction_scores: List of prediction scores per video
        gt_labels: List of ground truth labels per video
        output_path: Path to save the CSV file
    """
    try:
        # Verify all inputs have same length
        lengths = [len(video_ids), len(prediction_scores), len(gt_labels)]
        if len(set(lengths)) != 1:
            print(
                f"Warning: Length mismatch - Videos: {lengths[0]}, Predictions: {lengths[1]}, Labels: {lengths[2]}"
            )
            length = min(lengths)
        else:
            length = lengths[0]

        # Prepare data for writing
        save_data = []
        for idx in range(length):
            save_data.append(
                {
                    "video_id": str(video_ids[idx]),
                    "label": str(gt_labels[idx]).replace(" ", ""),
                    "prediction_score": prediction_scores[idx],
                }
            )

        # Write to CSV
        with open(output_path, mode="w", newline="") as csv_file:
            fieldnames = ["video_id", "label", "prediction_score"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for data in save_data:
                writer.writerow(data)

        print(f"Successfully saved video prediction scores in {output_path}")
        return True

    except Exception as e:
        print(f"Error in write_video_scores: {str(e)}")
        return False
