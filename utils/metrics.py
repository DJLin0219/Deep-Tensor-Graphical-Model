"""
metrics.py
-----------------------------------
Evaluation metrics for DETECT model.
Includes AUC, AUPR, FDR, TPR, FPR, Fβ score, and SHD.

Author: DETECT Team (Dianjun Lin et al.)
"""

import numpy as np
from sklearn import metrics
from pprint import pprint


def get_auc(y_true, y_score):
    """Compute AUROC and AUPR given binary labels and scores."""
    y_true = np.array(y_true).astype(int)
    y_score = np.array(y_score).astype(float)

    if np.all(y_true == 0) or np.all(y_true == 1):
        # Degenerate case: all positives or all negatives
        return np.nan, np.nan

    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    auc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(y_true, y_score)
    return auc, aupr


def report_metrics(true_theta, pred_theta, beta=1.0):
    """
    Compute various structural metrics comparing the true and predicted graphs.

    Args:
        true_theta (np.ndarray): Ground-truth precision matrix (DxD)
        pred_theta (np.ndarray): Predicted precision matrix (DxD)
        beta (float, optional): Beta for the F-beta score. Default = 1.

    Returns:
        dict: A dictionary containing:
            - FDR: False Discovery Rate = FP / P
            - TPR: True Positive Rate = TP / T
            - FPR: False Positive Rate = FP / F
            - SHD: Structural Hamming Distance = E + M
            - nnzTrue: #Non-zeros in ground-truth
            - nnzPred: #Non-zeros in prediction
            - precision, recall, Fbeta
            - AUC, AUPR
    """
    true_theta = np.array(true_theta, dtype=float)
    pred_theta = np.array(pred_theta, dtype=float)
    d = pred_theta.shape[-1]

    # Binary edge indicators (ignore sign)
    G_true = np.where(true_theta != 0, 1, 0)
    G_pred = np.where(pred_theta != 0, 1, 0)

    # Use only upper triangle (since precision matrices are symmetric)
    triu_idx = np.triu_indices(d, 1)
    y_true = G_true[triu_idx]
    y_pred = G_pred[triu_idx]
    y_score = np.abs(pred_theta[triu_idx])

    # --- Core counts ---
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TN = np.sum((y_true == 0) & (y_pred == 0))

    P = np.sum(y_pred)
    T = np.sum(y_true)
    F = len(y_true) - T

    # --- Basic metrics ---
    FDR = FP / (P + 1e-8)
    TPR = TP / (T + 1e-8)
    FPR = FP / (F + 1e-8)
    SHD = np.sum(y_true != y_pred)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)

    # --- F-beta score ---
    num = (1 + beta ** 2) * precision * recall
    den = (beta ** 2) * precision + recall + 1e-8
    Fbeta = num / den

    # --- AUC & AUPR ---
    auc, aupr = get_auc(y_true, y_score)

    return {
        "FDR": FDR,
        "TPR": TPR,
        "FPR": FPR,
        "SHD": SHD,
        "nnzTrue": T,
        "nnzPred": P,
        "precision": precision,
        "recall": recall,
        "Fbeta": Fbeta,
        "auc": auc,
        "aupr": aupr,
    }


def summarize_metrics(results, method_name="DETECT"):
    """
    Aggregate and summarize metrics across multiple runs.

    Args:
        results (list[dict]): A list of metrics dicts (from multiple experiments)
        method_name (str): Method identifier for printing summary.

    Returns:
        dict[str, (float, float)]: Mean and std of each metric.
    """
    if not results:
        print("No results to summarize.")
        return {}

    all_keys = results[0].keys()
    summary = {key: [] for key in all_keys}

    for r in results:
        for key in all_keys:
            summary[key].append(r[key])

    # Compute mean and std
    summary_stats = {k: (np.mean(v), np.std(v)) for k, v in summary.items()}

    print(f"\n=== Summary for {method_name} ===")
    pprint(summary_stats)
    print(f"Total runs: {len(results)}\n")

    return summary_stats
