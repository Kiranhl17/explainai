"""
Metrics Service
===============
Computes model performance metrics for classification and regression tasks.

For classifiers: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
For regressors:  MSE, RMSE, MAE, R²

Academic Note on Metric Selection:
-----------------------------------
For imbalanced datasets, Accuracy is misleading. Precision, Recall, and F1
are preferred:
  - Precision = TP / (TP + FP)  — of predicted positives, how many are correct?
  - Recall    = TP / (TP + FN)  — of actual positives, how many were found?
  - F1        = 2 × (P × R) / (P + R)  — harmonic mean (penalises extremes)

ROC-AUC is the probability that the model ranks a random positive
instance higher than a random negative one. It is threshold-independent.

Macro vs Weighted averaging:
  - Macro: simple average across classes (treats all classes equally)
  - Weighted: class-frequency-weighted average (reflects actual class distribution)
  Both are reported for transparency.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger("explainai.metrics")


def compute_metrics(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    model_info: Dict,
) -> Dict:
    """
    Compute comprehensive performance metrics.

    Parameters
    ----------
    model : sklearn estimator
    X : pd.DataFrame — feature matrix
    y : pd.Series — true labels/targets
    model_info : dict — from model_handler.load_model()

    Returns
    -------
    dict with structured metrics appropriate to the task type
    """
    y_pred = model.predict(X.values)
    y_true = y.values

    if model_info["is_classifier"]:
        return _classification_metrics(model, X, y_true, y_pred, model_info)
    else:
        return _regression_metrics(y_true, y_pred)


def _classification_metrics(
    model: Any,
    X: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_info: Dict,
) -> Dict:
    """Compute classification performance metrics."""
    is_binary = model_info.get("n_classes", 2) == 2

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Confusion matrix (serialized as nested list for JSON)
    cm = confusion_matrix(y_true, y_pred).tolist()

    # ROC-AUC (requires probability estimates)
    roc_auc = None
    if model_info["supports_probability"]:
        try:
            y_prob = model.predict_proba(X.values)
            if is_binary:
                roc_auc = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                roc_auc = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                )
        except Exception as e:
            logger.warning(f"ROC-AUC computation failed: {e}")

    # Per-class metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    per_class = {
        k: v
        for k, v in report.items()
        if k not in ("accuracy", "macro avg", "weighted avg")
    }

    return {
        "task_type": "classification",
        "n_samples": int(len(y_true)),
        "n_classes": int(len(np.unique(y_true))),
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_weighted": round(f1_weighted, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": cm,
        "per_class_metrics": per_class,
        "notes": {
            "accuracy": "Proportion of correct predictions. Misleading for imbalanced datasets.",
            "f1_macro": "Harmonic mean of precision and recall. Macro treats all classes equally.",
            "f1_weighted": "F1 weighted by class frequency — more informative for imbalanced data.",
            "roc_auc": "Area under the ROC curve. Threshold-independent. 0.5=random, 1.0=perfect.",
        },
    }


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute regression performance metrics."""
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "task_type": "regression",
        "n_samples": int(len(y_true)),
        "mse": round(mse, 6),
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "r2_score": round(r2, 6),
        "notes": {
            "mse": "Mean Squared Error — penalises large errors quadratically.",
            "mae": "Mean Absolute Error — more robust to outliers than MSE.",
            "r2": "Coefficient of determination. 1.0=perfect fit, 0=mean baseline, <0=worse than mean.",
        },
    }
