from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics such as accuracy, precision, recall, F1 score, and ROC AUC.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        y_prob (np.ndarray, optional): Predicted probabilities for the positive class.

    Returns:
        dict: A dictionary with classification metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
    }

    if y_prob is not None:
        metrics.update(_roc_auc_score(y_true, y_prob))
        metrics.update({"log_loss": log_loss(y_true, y_prob)})

    return metrics


def _roc_auc_score(y_true, y_prob):
    unique_classes = np.unique(y_true)

    if len(unique_classes) == 2:
        y_prob_positive = y_prob[:, 1]
        return {"roc_auc": roc_auc_score(y_true, y_prob_positive)}

    else:
        y_true_binarized = label_binarize(y_true, classes=unique_classes)
        return {
            "roc_auc_ovr": roc_auc_score(
                y_true_binarized, y_prob, multi_class="ovr", average="macro"
            )
        }
