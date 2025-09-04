import torch
import numpy as np


def calculate_hamming_loss(predictions, targets):
    """
    Calculate Hamming Loss for multi-label classification.

    Args:
        predictions (torch.Tensor): Predicted labels (after thresholding)
        targets (torch.Tensor): Ground truth labels

    Returns:
        float: Hamming Loss value
    """
    n_samples, n_labels = targets.shape
    incorrect = (predictions != targets).float().sum()
    return incorrect / (n_samples * n_labels)


def calculate_hamming_distance(predictions, targets):
    """
    Calculate the Hamming distance (fraction of different elements).

    Args:
        predictions: Binary predictions array
        targets: Binary target array

    Returns:
        float: Hamming distance
    """
    return np.mean(predictions != targets)


def calculate_confusion_matrix(predictions, targets):
    """
    Calculate TP, TN, FP, FN values.

    Args:
        predictions: Binary predictions array
        targets: Binary target array

    Returns:
        dict: Dictionary with TP, TN, FP, FN values
    """
    tp = np.sum((predictions == 1) & (targets == 1))
    tn = np.sum((predictions == 0) & (targets == 0))
    fp = np.sum((predictions == 1) & (targets == 0))
    fn = np.sum((predictions == 0) & (targets == 1))

    return {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)}


def calculate_precision_recall_f1(confusion_matrix):
    """
    Calculate precision, recall, and F1 score.

    Args:
        confusion_matrix: Dictionary with TP, TN, FP, FN values

    Returns:
        dict: Dictionary with precision, recall, and F1 score
    """
    tp = confusion_matrix["TP"]
    tn = confusion_matrix["TN"]
    fp = confusion_matrix["FP"]
    fn = confusion_matrix["FN"]

    # Calculate class 1 metrics
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score_1 = (
        2 * (precision_1 * recall_1) / (precision_1 + recall_1)
        if (precision_1 + recall_1) > 0
        else 0
    )

    # Calculate class 0 metrics
    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score_0 = (
        2 * (precision_0 * recall_0) / (precision_0 + recall_0)
        if (precision_0 + recall_0) > 0
        else 0
    )

    # Average metrics
    precision = (precision_0 + precision_1) / 2
    recall = (recall_0 + recall_1) / 2
    f1_score = (f1_score_0 + f1_score_1) / 2

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "precision_0": precision_0,
        "recall_0": recall_0,
        "f1_score_0": f1_score_0,
        "precision_1": precision_1,
        "recall_1": recall_1,
        "f1_score_1": f1_score_1,
    }


def calculate_f1_score(tp, fp, fn, tn):
    """
    Calculate F1 score from confusion matrix values.

    Args:
        tp (int): True positives
        fp (int): False positives
        fn (int): False negatives
        tn (int): True negatives

    Returns:
        tuple: F1 scores for class 1, class 0, and their average
    """
    # Class 1 metrics
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score_1 = (
        2 * (precision_1 * recall_1) / (precision_1 + recall_1)
        if (precision_1 + recall_1) > 0
        else 0
    )

    # Class 0 metrics
    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score_0 = (
        2 * (precision_0 * recall_0) / (precision_0 + recall_0)
        if (precision_0 + recall_0) > 0
        else 0
    )

    # Average F1 score
    avg_f1 = (f1_score_1 + f1_score_0) / 2

    return f1_score_1, f1_score_0, avg_f1


def calculate_jaccard_index(tp, fp, fn):
    """
    Calculate Jaccard Index (IoU) for binary classification.

    Args:
        tp (int): True positives
        fp (int): False positives
        fn (int): False negatives

    Returns:
        float: Jaccard Index value
    """
    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0


def calculate_jaccard_index(confusion_matrix):
    """
    Calculate the Jaccard index (IoU).

    Args:
        confusion_matrix: Dictionary with TP, TN, FP, FN values

    Returns:
        float: Jaccard index
    """
    tp = confusion_matrix["TP"]
    fp = confusion_matrix["FP"]
    fn = confusion_matrix["FN"]

    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0


def calculate_metrics(predictions, targets):
    """
    Calculate all metrics for multi-label classification.

    Args:
        predictions: Binary predictions array of shape [samples, num_labels]
        targets: Binary target array of shape [samples, num_labels]

    Returns:
        dict: Dictionary with all calculated metrics
    """
    num_samples, num_labels = predictions.shape

    # Initialize per-feature metrics
    feature_metrics = []

    # Calculate metrics for each feature
    for feature_idx in range(num_labels):
        pred_feature = predictions[:, feature_idx]
        target_feature = targets[:, feature_idx]

        # Calculate Hamming distance
        hamming_distance = calculate_hamming_distance(pred_feature, target_feature)

        # Calculate confusion matrix
        confusion_matrix = calculate_confusion_matrix(pred_feature, target_feature)

        # Calculate precision, recall, and F1 score
        pr_metrics = calculate_precision_recall_f1(confusion_matrix)

        # Calculate Jaccard index
        jaccard_index = calculate_jaccard_index(confusion_matrix)

        # Combine all metrics for this feature
        feature_metric = {
            "hamming_distance": hamming_distance,
            "jaccard_index": jaccard_index,
            **confusion_matrix,
            **pr_metrics,
        }

        feature_metrics.append(feature_metric)

    # Calculate overall metrics by summing confusion matrices across features
    overall_cm = {
        "TP": sum(m["TP"] for m in feature_metrics),
        "TN": sum(m["TN"] for m in feature_metrics),
        "FP": sum(m["FP"] for m in feature_metrics),
        "FN": sum(m["FN"] for m in feature_metrics),
    }

    overall_pr_metrics = calculate_precision_recall_f1(overall_cm)
    overall_jaccard_index = calculate_jaccard_index(overall_cm)

    # Calculate average Hamming distance
    overall_hamming_distance = np.mean([m["hamming_distance"] for m in feature_metrics])

    # Combine all metrics
    metrics = {
        "hamming_distance": overall_hamming_distance,
        "jaccard_index": overall_jaccard_index,
        "feature_metrics": feature_metrics,
        **overall_cm,
        **overall_pr_metrics,
    }

    return metrics


def get_metrics_from_predictions(predictions, targets):
    """
    Compute comprehensive metrics from binary predictions and targets.

    Args:
        predictions (torch.Tensor): Binary predictions (after thresholding)
        targets (torch.Tensor): Binary ground truth labels

    Returns:
        dict: Dictionary containing all metrics
    """
    n_samples, n_labels = targets.shape
    metrics = {
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0,
        "hamming_loss": 0,
        "feature_metrics": {},
    }

    # Calculate per-feature metrics
    for feature_idx in range(n_labels):
        preds = predictions[:, feature_idx]
        trues = targets[:, feature_idx]

        tp = (preds & trues).sum().item()
        tn = (~preds & ~trues).sum().item()
        fp = (preds & ~trues).sum().item()
        fn = (~preds & trues).sum().item()

        metrics["TP"] += tp
        metrics["TN"] += tn
        metrics["FP"] += fp
        metrics["FN"] += fn

        feature_hamming_loss = (fp + fn) / preds.numel() if preds.numel() > 0 else 0

        metrics["feature_metrics"][feature_idx] = {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "hamming_loss": feature_hamming_loss,
        }

    # Calculate overall hamming loss
    metrics["hamming_loss"] = (metrics["FP"] + metrics["FN"]) / (n_samples * n_labels)

    # Calculate F1 scores
    f1_class1, f1_class0, avg_f1 = calculate_f1_score(
        metrics["TP"], metrics["FP"], metrics["FN"], metrics["TN"]
    )
    metrics["f1_class1"] = f1_class1
    metrics["f1_class0"] = f1_class0
    metrics["f1_avg"] = avg_f1

    # Calculate Jaccard index
    metrics["jaccard_index"] = calculate_jaccard_index(
        metrics["TP"], metrics["FP"], metrics["FN"]
    )

    return metrics
