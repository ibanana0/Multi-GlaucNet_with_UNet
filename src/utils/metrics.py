"""
Evaluation Metrics for Segmentation and Classification tasks
in accordance with MultiGlaucNet Evaluation Metrics (Phase 5).
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix


def compute_segmentation_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict:
    """
    Menghitung metrik segmentasi biner:
    - Dice Coefficient
    - IoU (Jaccard Index)
    - Accuracy
    - Sensitivity (Recall)
    - Specificity
    """
    with torch.no_grad():
        pred_binary = (pred > threshold).float()
        target = (target > threshold).float()

        p_flat = pred_binary.view(-1)
        t_flat = target.view(-1)

        tp = (p_flat * t_flat).sum().item()
        fp = (p_flat * (1.0 - t_flat)).sum().item()
        fn = ((1.0 - p_flat) * t_flat).sum().item()
        tn = ((1.0 - p_flat) * (1.0 - t_flat)).sum().item()

        epsilon = 1e-7

        # Dice = 2TP / (2TP + FP + FN)
        dice = (2.0 * tp + epsilon) / (2.0 * tp + fp + fn + epsilon)

        # IoU = TP / (TP + FP + FN)
        iou = (tp + epsilon) / (tp + fp + fn + epsilon)

        # Accuracy = (TP + TN) / Total
        accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)

        # Sensitivity (Recall) = TP / (TP + FN)
        sensitivity = (tp + epsilon) / (tp + fn + epsilon)

        # Specificity = TN / (TN + FP)
        specificity = (tn + epsilon) / (tn + fp + epsilon)

        return {
            "dice": dice,
            "iou": iou,
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity
        }


def compute_classification_metrics(probs: np.ndarray, targets: np.ndarray, threshold: float = 0.5, return_roc_curve: bool = False) -> dict:
    """
    Menghitung metrik deteksi glaukoma biner (Normal vs Glaucoma):
    - Accuracy
    - Sensitivity (Recall)
    - Specificity
    - Precision
    - F1-Score (Harmonic mean dari Precision & Sensitivity)
    - AUC-ROC
    - Youden's J Statistic (Titik potong optimal)
    - Confusion Matrix
    - (Opsional) ROC Curve coordinates (fpr, tpr, thresholds)
    """
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(targets, preds, labels=[0, 1])

    # cm structure: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    eps = 1e-7

    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    precision = tp / (tp + fp + eps)
    f1 = (2.0 * precision * sensitivity) / (precision + sensitivity + eps)

    # AUC-ROC dan koordinat Kurva ROC
    fpr_list, tpr_list, thresholds_list = [], [], []
    optimal_threshold = threshold
    optimal_j = 0.0

    try:
        from sklearn.metrics import roc_curve
        auc = roc_auc_score(targets, probs)
        fpr, tpr, threshs = roc_curve(targets, probs)
        
        # Hitung Youden's J Statistic: J = Sensitivity + Specificity - 1 = TPR - FPR
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        optimal_threshold = float(threshs[best_idx])
        optimal_j = float(j_scores[best_idx])
        
        if return_roc_curve:
            fpr_list = fpr.tolist()
            tpr_list = tpr.tolist()
            thresholds_list = threshs.tolist()
    except Exception:
        auc = 0.5  # Jika hanya terdapat satu kelas pada batch kecil

    result = {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1_score": float(f1),
        "auc_roc": float(auc),
        "youden_optimal_threshold": float(optimal_threshold),
        "youden_j_score": float(optimal_j),
        "confusion_matrix": cm.tolist()
    }

    if return_roc_curve:
        result["roc_curve"] = {
            "fpr": fpr_list,
            "tpr": tpr_list,
            "thresholds": thresholds_list
        }

    return result
