"""
Evaluation Metrics and Uncertainty Calibration for PeriECG-RiskNet

Revised to be consistent with the multi-label version of PeriECG-RiskNet:
- Model outputs are interpreted as independent class probabilities (sigmoid), not softmax.
- Robust calibration / ECE-MCE computation for multi-label probabilities.
- Uncertainty metrics work for multi-label probabilities and MC-dropout outputs.
- Lead robustness analysis supports both probability-returning models and logit-returning models.
- Added threshold sweep helper for deployment tuning.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Core classification metrics
# ---------------------------------------------------------------------------

def _ensure_2d_numpy(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    return x


def _validate_shapes(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_true = _ensure_2d_numpy(y_true)
    y_pred = _ensure_2d_numpy(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    return y_true, y_pred


def compute_auc_roc(y_true: np.ndarray, y_pred: np.ndarray,
                    average: str = 'macro') -> float:
    """
    Compute multi-label AUC-ROC.

    Args:
        y_true: (N, C) binary ground truth.
        y_pred: (N, C) predicted probabilities in [0, 1].
        average: 'macro', 'micro', or 'weighted'.
    """
    y_true, y_pred = _validate_shapes(y_true, y_pred)
    try:
        return float(roc_auc_score(y_true, y_pred, average=average))
    except ValueError:
        aucs = []
        for i in range(y_true.shape[1]):
            positives = y_true[:, i].sum()
            if 0 < positives < len(y_true):
                aucs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
        return float(np.mean(aucs)) if aucs else 0.0


def compute_auc_pr(y_true: np.ndarray, y_pred: np.ndarray,
                   average: str = 'macro') -> float:
    """Compute multi-label AUC-PR."""
    y_true, y_pred = _validate_shapes(y_true, y_pred)
    try:
        return float(average_precision_score(y_true, y_pred, average=average))
    except ValueError:
        aps = []
        for i in range(y_true.shape[1]):
            if y_true[:, i].sum() > 0:
                aps.append(average_precision_score(y_true[:, i], y_pred[:, i]))
        return float(np.mean(aps)) if aps else 0.0


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray,
               average: str = 'macro', threshold: float = 0.5) -> float:
    """Compute multi-label F1 score after thresholding probabilities."""
    y_true, y_pred = _validate_shapes(y_true, y_pred)
    y_pred_bin = (y_pred >= threshold).astype(int)
    return float(f1_score(y_true, y_pred_bin, average=average, zero_division=0))


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                     threshold: float = 0.5) -> float:
    """Compute exact-match accuracy for multi-label predictions."""
    y_true, y_pred = _validate_shapes(y_true, y_pred)
    y_pred_bin = (y_pred >= threshold).astype(int)
    return float(accuracy_score(y_true, y_pred_bin))


def compute_hamming_loss(y_true: np.ndarray, y_pred: np.ndarray,
                         threshold: float = 0.5) -> float:
    """Compute Hamming loss for multi-label predictions."""
    y_true, y_pred = _validate_shapes(y_true, y_pred)
    y_pred_bin = (y_pred >= threshold).astype(int)
    return float(np.mean(y_true != y_pred_bin))


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def _binary_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                              n_bins: int = 15) -> Tuple[float, float]:
    """
    Compute binary ECE and MCE with fixed-width bins.

    Args:
        y_true: (N,) binary labels
        y_prob: (N,) probabilities in [0, 1]
    Returns:
        (ece, mce)
    """
    y_true = np.asarray(y_true).astype(np.float32)
    y_prob = np.asarray(y_prob).astype(np.float32)

    if y_true.ndim != 1 or y_prob.ndim != 1:
        raise ValueError("_binary_calibration_error expects 1D inputs.")
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError("y_true and y_prob must have same length.")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=False)

    ece = 0.0
    mce = 0.0
    n = len(y_prob)

    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        gap = abs(acc - conf)
        ece += (mask.sum() / n) * gap
        mce = max(mce, gap)

    return float(ece), float(mce)


def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray,
                               n_bins: int = 15) -> float:
    """
    Compute Expected Calibration Error (ECE) for multi-label classification.
    The metric is averaged across valid classes.
    """
    y_true, y_pred = _validate_shapes(y_true, y_pred)
    eces = []
    for i in range(y_true.shape[1]):
        positives = y_true[:, i].sum()
        if positives == 0 or positives == len(y_true):
            continue
        ece, _ = _binary_calibration_error(y_true[:, i], y_pred[:, i], n_bins=n_bins)
        eces.append(ece)
    return float(np.mean(eces)) if eces else 0.0


def maximum_calibration_error(y_true: np.ndarray, y_pred: np.ndarray,
                              n_bins: int = 15) -> float:
    """Compute Maximum Calibration Error (MCE) averaged across valid classes."""
    y_true, y_pred = _validate_shapes(y_true, y_pred)
    mces = []
    for i in range(y_true.shape[1]):
        positives = y_true[:, i].sum()
        if positives == 0 or positives == len(y_true):
            continue
        _, mce = _binary_calibration_error(y_true[:, i], y_pred[:, i], n_bins=n_bins)
        mces.append(mce)
    return float(np.mean(mces)) if mces else 0.0


# ---------------------------------------------------------------------------
# Uncertainty metrics
# ---------------------------------------------------------------------------

def predictive_entropy(probs: np.ndarray, eps: float = 1e-8,
                       normalize: bool = True) -> np.ndarray:
    """
    Compute entropy over per-sample class probabilities.

    For multi-label probabilities this is used as a summary uncertainty score.
    If normalize=True, divides by log(C), yielding values in [0, 1] approximately.
    """
    probs = np.asarray(probs, dtype=np.float32)
    if probs.ndim != 2:
        raise ValueError(f"predictive_entropy expects (N, C), got {probs.shape}")
    ent = -np.sum(probs * np.log(probs + eps), axis=-1)
    if normalize and probs.shape[1] > 1:
        ent = ent / np.log(probs.shape[1])
    return ent.astype(np.float32)


def binary_entropy(probs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Bernoulli entropy for multi-label probabilities.

    Returns per-class entropy with same shape as probs.
    """
    probs = np.asarray(probs, dtype=np.float32)
    return -(probs * np.log(probs + eps) + (1.0 - probs) * np.log(1.0 - probs + eps))


def mean_binary_entropy(probs: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Sample-wise average Bernoulli entropy across classes.

    More appropriate than softmax entropy when classes are independent labels.
    """
    probs = np.asarray(probs, dtype=np.float32)
    ent = binary_entropy(probs).mean(axis=-1)
    if normalize:
        ent = ent / np.log(2.0)
    return ent.astype(np.float32)


def mutual_information(mc_probs: np.ndarray, eps: float = 1e-8,
                       normalize: bool = True) -> np.ndarray:
    """
    Compute epistemic uncertainty from MC-dropout samples for multi-label outputs.

    Args:
        mc_probs: (T, N, C) probabilities from T MC forward passes.

    Returns:
        (N,) mutual information scores.
    """
    mc_probs = np.asarray(mc_probs, dtype=np.float32)
    if mc_probs.ndim != 3:
        raise ValueError(f"mutual_information expects (T, N, C), got {mc_probs.shape}")

    mean_probs = mc_probs.mean(axis=0)  # (N, C)
    entropy_of_mean = binary_entropy(mean_probs, eps=eps).mean(axis=-1)
    expected_entropy = binary_entropy(mc_probs, eps=eps).mean(axis=(0, 2))
    mi = entropy_of_mean - expected_entropy
    mi = np.maximum(mi, 0.0)
    if normalize:
        mi = mi / np.log(2.0)
    return mi.astype(np.float32)


def uncertainty_rejection_auc(y_true: np.ndarray, y_pred: np.ndarray,
                              uncertainty: np.ndarray,
                              threshold: float = 0.5) -> Dict[str, Any]:
    """
    Evaluate whether higher uncertainty correlates with prediction errors.

    Returns:
        Dictionary containing AUROC for error detection and retained-accuracy curve.
    """
    y_true, y_pred = _validate_shapes(y_true, y_pred)
    uncertainty = np.asarray(uncertainty).reshape(-1)
    if len(uncertainty) != len(y_true):
        raise ValueError("uncertainty must have shape (N,)")

    y_pred_bin = (y_pred >= threshold).astype(int)
    errors = (y_pred_bin != y_true).any(axis=-1).astype(int)

    unique_error_values = np.unique(errors)
    if len(unique_error_values) < 2:
        auroc = np.nan
    else:
        auroc = float(roc_auc_score(errors, uncertainty))

    sorted_idx = np.argsort(uncertainty)
    fractions = np.linspace(0.1, 1.0, 10)
    retained_accs = []
    retained_hamming = []
    for frac in fractions:
        n_keep = max(1, int(len(sorted_idx) * frac))
        keep_idx = sorted_idx[:n_keep]
        retained_accs.append(float(accuracy_score(y_true[keep_idx], y_pred_bin[keep_idx])))
        retained_hamming.append(float(np.mean(y_true[keep_idx] != y_pred_bin[keep_idx])))

    return {
        'error_detection_auroc': auroc,
        'fractions_retained': fractions.tolist(),
        'retained_accuracy': retained_accs,
        'retained_hamming_loss': retained_hamming,
        'error_rate': float(errors.mean()),
    }


def compute_alert_burden(y_pred: np.ndarray, uncertainty: np.ndarray,
                         uncertainty_threshold: float = 0.2,
                         risk_threshold: float = 0.5) -> Dict[str, float]:
    """
    Simulate alert burden for clinical deployment.

    Args:
        y_pred: (N, C) predicted probabilities.
        uncertainty: (N,) uncertainty scores.
        uncertainty_threshold: Above this, flag for manual review.
        risk_threshold: Above this, trigger risk alert.
    """
    y_pred = _ensure_2d_numpy(y_pred)
    uncertainty = np.asarray(uncertainty).reshape(-1)
    if len(uncertainty) != len(y_pred):
        raise ValueError("uncertainty must have shape (N,)")

    high_risk = (y_pred >= risk_threshold).any(axis=-1)
    high_uncertainty = uncertainty >= uncertainty_threshold

    auto_alerts = high_risk & (~high_uncertainty)
    manual_reviews = high_uncertainty
    suppressed = (~high_risk) & (~high_uncertainty)

    total = len(y_pred)
    return {
        'total_samples': int(total),
        'auto_alerts': int(auto_alerts.sum()),
        'auto_alert_rate': float(auto_alerts.sum() / total),
        'manual_reviews': int(manual_reviews.sum()),
        'manual_review_rate': float(manual_reviews.sum() / total),
        'suppressed': int(suppressed.sum()),
        'suppression_rate': float(suppressed.sum() / total),
        'alert_burden': float((auto_alerts.sum() + manual_reviews.sum()) / total),
    }


def threshold_sweep(y_true: np.ndarray, y_pred: np.ndarray,
                    thresholds: Optional[np.ndarray] = None,
                    average: str = 'macro') -> Dict[str, Any]:
    """
    Sweep decision thresholds and return the best F1 operating point.
    Useful for deployment tuning in multi-label settings.
    """
    y_true, y_pred = _validate_shapes(y_true, y_pred)
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    rows = []
    best = None
    best_f1 = -1.0
    for thr in thresholds:
        row = {
            'threshold': float(thr),
            'f1': compute_f1(y_true, y_pred, average=average, threshold=float(thr)),
            'exact_match_accuracy': compute_accuracy(y_true, y_pred, threshold=float(thr)),
            'hamming_loss': compute_hamming_loss(y_true, y_pred, threshold=float(thr)),
        }
        rows.append(row)
        if row['f1'] > best_f1:
            best_f1 = row['f1']
            best = row

    return {
        'best': best,
        'curve': rows,
    }


# ---------------------------------------------------------------------------
# Lead robustness
# ---------------------------------------------------------------------------

def _model_outputs_to_probs(output: torch.Tensor) -> np.ndarray:
    """
    Convert model outputs to probabilities.

    Assumes logits for multi-label classification unless values are already in [0, 1].
    """
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError("Model output must be a torch.Tensor or tuple containing one.")

    detached = output.detach()
    if detached.numel() == 0:
        raise ValueError("Empty model output.")

    min_val = float(detached.min().item())
    max_val = float(detached.max().item())
    if min_val >= 0.0 and max_val <= 1.0:
        probs = detached
    else:
        probs = torch.sigmoid(detached)
    return probs.cpu().numpy()


def evaluate_lead_robustness(model: torch.nn.Module,
                             test_loader: torch.utils.data.DataLoader,
                             device: str = 'cuda') -> Dict[str, float]:
    """
    Evaluate robustness to individual lead dropout.

    This version is aligned with the multi-label sigmoid output of PeriECG-RiskNet.
    """
    model.eval()
    baseline_aucs = []
    lead_drop_aucs = {i: [] for i in range(7)}

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y_np = y.cpu().numpy()

            baseline_probs = _model_outputs_to_probs(model(x))
            baseline_aucs.append(compute_auc_roc(y_np, baseline_probs))

            for lead_idx in range(7):
                x_dropped = x.clone()
                x_dropped[:, lead_idx, :] = 0.0
                probs_d = _model_outputs_to_probs(model(x_dropped))
                lead_drop_aucs[lead_idx].append(compute_auc_roc(y_np, probs_d))

    results = {'baseline_auc': float(np.mean(baseline_aucs)) if baseline_aucs else 0.0}
    for lead_idx in range(7):
        drop_auc = float(np.mean(lead_drop_aucs[lead_idx])) if lead_drop_aucs[lead_idx] else 0.0
        results[f'lead_{lead_idx}_drop_auc'] = drop_auc
        results[f'lead_{lead_idx}_degradation'] = results['baseline_auc'] - drop_auc
    return results


# ---------------------------------------------------------------------------
# Comprehensive evaluation
# ---------------------------------------------------------------------------

def evaluate_model(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   uncertainty: Optional[np.ndarray] = None,
                   class_names: Optional[List[str]] = None,
                   threshold: float = 0.5) -> Dict[str, Any]:
    """
    Comprehensive multi-label evaluation suite.

    Args:
        y_true: (N, C) binary ground truth.
        y_pred: (N, C) predicted probabilities.
        uncertainty: (N,) optional uncertainty scores.
        class_names: Optional class names.
        threshold: Decision threshold for discrete metrics.
    """
    y_true, y_pred = _validate_shapes(y_true, y_pred)

    if class_names is None:
        class_names = [f"Class_{i}" for i in range(y_true.shape[1])]

    metrics: Dict[str, Any] = {
        'auc_roc_macro': compute_auc_roc(y_true, y_pred, 'macro'),
        'auc_roc_micro': compute_auc_roc(y_true, y_pred, 'micro'),
        'auc_pr_macro': compute_auc_pr(y_true, y_pred, 'macro'),
        'auc_pr_micro': compute_auc_pr(y_true, y_pred, 'micro'),
        'f1_macro': compute_f1(y_true, y_pred, 'macro', threshold=threshold),
        'f1_micro': compute_f1(y_true, y_pred, 'micro', threshold=threshold),
        'exact_match_accuracy': compute_accuracy(y_true, y_pred, threshold=threshold),
        'hamming_loss': compute_hamming_loss(y_true, y_pred, threshold=threshold),
        'ece': expected_calibration_error(y_true, y_pred),
        'mce': maximum_calibration_error(y_true, y_pred),
        'decision_threshold': float(threshold),
        'mean_predicted_positive_labels': float((y_pred >= threshold).sum(axis=1).mean()),
    }

    per_class = {}
    for i, name in enumerate(class_names):
        positives = y_true[:, i].sum()
        if positives > 0:
            try:
                auc_roc_i = float(roc_auc_score(y_true[:, i], y_pred[:, i])) if positives < len(y_true) else np.nan
            except ValueError:
                auc_roc_i = np.nan
            per_class[name] = {
                'auc_roc': auc_roc_i,
                'auc_pr': float(average_precision_score(y_true[:, i], y_pred[:, i])),
                'prevalence': float(y_true[:, i].mean()),
            }
    metrics['per_class'] = per_class

    if uncertainty is not None:
        uncertainty = np.asarray(uncertainty).reshape(-1)
        metrics['uncertainty'] = uncertainty_rejection_auc(y_true, y_pred, uncertainty, threshold=threshold)
        metrics['alert_burden'] = compute_alert_burden(y_pred, uncertainty)
        metrics['mean_uncertainty'] = float(uncertainty.mean())
        metrics['std_uncertainty'] = float(uncertainty.std())

    metrics['threshold_sweep'] = threshold_sweep(y_true, y_pred)
    return metrics


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Pretty-print evaluation metrics."""
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Decision Threshold:   {metrics['decision_threshold']:.2f}")
    print(f"AUC-ROC (macro):     {metrics['auc_roc_macro']:.4f}")
    print(f"AUC-ROC (micro):     {metrics['auc_roc_micro']:.4f}")
    print(f"AUC-PR (macro):      {metrics['auc_pr_macro']:.4f}")
    print(f"AUC-PR (micro):      {metrics['auc_pr_micro']:.4f}")
    print(f"F1-score (macro):    {metrics['f1_macro']:.4f}")
    print(f"F1-score (micro):    {metrics['f1_micro']:.4f}")
    print(f"Exact Match Acc:     {metrics['exact_match_accuracy']:.4f}")
    print(f"Hamming Loss:        {metrics['hamming_loss']:.4f}")
    print(f"ECE:                 {metrics['ece']:.4f}")
    print(f"MCE:                 {metrics['mce']:.4f}")
    print(f"Mean #Positive Pred: {metrics['mean_predicted_positive_labels']:.4f}")

    if 'uncertainty' in metrics:
        print("\n--- Uncertainty ---")
        print(f"Error Detection AUROC: {metrics['uncertainty']['error_detection_auroc']}")
        print(f"Mean Uncertainty:      {metrics['mean_uncertainty']:.4f}")
        print(f"Std Uncertainty:       {metrics['std_uncertainty']:.4f}")

    if 'alert_burden' in metrics:
        ab = metrics['alert_burden']
        print("\n--- Alert Burden ---")
        print(f"Auto Alert Rate:       {ab['auto_alert_rate']:.2%}")
        print(f"Manual Review Rate:    {ab['manual_review_rate']:.2%}")
        print(f"Suppression Rate:      {ab['suppression_rate']:.2%}")
        print(f"Total Burden:          {ab['alert_burden']:.2%}")

    ts = metrics.get('threshold_sweep', {}).get('best', None)
    if ts is not None:
        print("\n--- Best Threshold Sweep Point ---")
        print(f"Threshold:             {ts['threshold']:.2f}")
        print(f"Best F1:               {ts['f1']:.4f}")
        print(f"Exact Match Acc:       {ts['exact_match_accuracy']:.4f}")
        print(f"Hamming Loss:          {ts['hamming_loss']:.4f}")

    print("=" * 60)


if __name__ == '__main__':
    np.random.seed(42)
    N, C = 1000, 13
    y_true = np.random.binomial(1, 0.2, size=(N, C))
    y_pred = np.random.rand(N, C)
    uncertainty = mean_binary_entropy(y_pred)

    metrics = evaluate_model(y_true, y_pred, uncertainty)
    print_metrics(metrics)
