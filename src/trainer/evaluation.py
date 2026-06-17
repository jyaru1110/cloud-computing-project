"""
Evaluacion multi-etiqueta con bootstrap CI y calibracion.

Metricas:
  - AUC-ROC por clase (con bootstrap IC 95%)
  - F1-score por clase (con bootstrap IC 95%)
  - F2-score por clase (enfasis en recall)
  - Precision / Recall por clase
  - Average precision (area bajo PR curve)
  - Expected Calibration Error (ECE) por clase

Umbral optimo por F2 para cada etiqueta (no 0.5 arbitrario).

Herramientas de IA utilizadas: Claude (generacion de codigo y estructura).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    average_precision_score, brier_score_loss,
)


LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error. Mide que tan calibradas estan las probabilidades."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece


def find_f2_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Encontrar el umbral que maximiza F2 para una etiqueta binaria."""
    best_f2 = 0
    best_t = 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (y_prob >= t).astype(int)
        if preds.sum() > 0 and preds.sum() < len(preds):
            f2 = fbeta_score(y_true, preds, beta=2, zero_division=0)
            if f2 > best_f2:
                best_f2 = f2
                best_t = t
    return best_t, best_f2


def fbeta_score(y_true, y_pred, beta=2, zero_division=0):
    """F-beta score implementado manualmente para evitar warnings."""
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision == 0 and recall == 0:
        return zero_division

    beta2 = beta ** 2
    fbeta = (1 + beta2) * precision * recall / (beta2 * precision + recall)
    return fbeta


def bootstrap_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    **metric_kwargs,
) -> dict:
    """
    Intervalo de confianza bootstrap para una metrica.

    Retorna: {"point", "ci_lower", "ci_upper", "std"}
    """
    rng = np.random.RandomState(42)
    n = len(y_true)
    boot_values = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        try:
            val = metric_fn(y_true[idx], y_prob[idx], **metric_kwargs)
            if np.isfinite(val):
                boot_values.append(val)
        except Exception:
            continue

    boot_values = np.array(boot_values)
    point = metric_fn(y_true, y_prob, **metric_kwargs) if len(boot_values) == 0 else np.mean(boot_values)

    if len(boot_values) > 0:
        ci_lower = np.percentile(boot_values, 100 * alpha / 2)
        ci_upper = np.percentile(boot_values, 100 * (1 - alpha / 2))
        std = boot_values.std()
    else:
        ci_lower, ci_upper, std = point, point, 0.0

    return {
        "point": round(point, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "std": round(std, 4),
    }


def evaluate_multilabel(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_cols: Optional[list[str]] = None,
    n_bootstrap: int = 500,
) -> dict:
    """
    Evaluacion completa multi-etiqueta con bootstrap CI.

    Args:
        y_true: (n, 6) etiquetas verdaderas
        y_prob: (n, 6) probabilidades predichas
        label_cols: nombres de etiquetas

    Returns:
        Diccionario con metricas por clase y agregadas.
    """
    label_cols = label_cols or LABEL_COLS

    results = {"per_label": {}, "macro": {}, "thresholds": {}}

    for j, label in enumerate(label_cols):
        yt = y_true[:, j]
        yp = y_prob[:, j]

        # AUC-ROC con bootstrap CI
        try:
            auc_result = bootstrap_metric(
                yt, yp, roc_auc_score, n_bootstrap=n_bootstrap
            )
        except Exception:
            auc_result = {"point": 0.5, "ci_lower": 0.5, "ci_upper": 0.5, "std": 0}

        # Average precision
        try:
            ap = average_precision_score(yt, yp)
        except Exception:
            ap = 0.0

        # ECE
        ece = compute_ece(yt, yp)

        # Brier score
        brier = brier_score_loss(yt, yp)

        # Umbral F2-optimo
        opt_t, opt_f2 = find_f2_optimal_threshold(yt, yp)

        # Metricas en umbral F2-optimo
        preds_f2 = (yp >= opt_t).astype(int)
        prec = precision_score(yt, preds_f2, zero_division=0)
        rec = recall_score(yt, preds_f2, zero_division=0)
        f1 = f1_score(yt, preds_f2, zero_division=0)
        f2 = fbeta_score(yt, preds_f2, beta=2)

        # F1 con bootstrap CI
        try:
            f1_result = bootstrap_metric(
                yt, preds_f2.astype(float),
                lambda yt, yp: f1_score(yt, (yp >= 0.5).astype(int), zero_division=0),
                n_bootstrap=n_bootstrap,
            )
        except Exception:
            f1_result = {"point": f1, "ci_lower": f1, "ci_upper": f1, "std": 0}

        results["per_label"][label] = {
            "auc_roc": auc_result,
            "avg_precision": round(ap, 4),
            "ece": round(ece, 4),
            "brier": round(brier, 4),
            "f2_threshold": round(opt_t, 2),
            "precision_at_f2": round(prec, 4),
            "recall_at_f2": round(rec, 4),
            "f1_at_f2": round(f1, 4),
            "f2_at_f2": round(f2, 4),
            "f1_bootstrap": f1_result,
        }

        results["thresholds"][label] = round(opt_t, 2)

    # Metricas macro
    auc_values = [r["auc_roc"]["point"] for r in results["per_label"].values()]
    f1_values = [r["f1_at_f2"] for r in results["per_label"].values()]
    f2_values = [r["f2_at_f2"] for r in results["per_label"].values()]
    ece_values = [r["ece"] for r in results["per_label"].values()]

    results["macro"] = {
        "auc_roc": round(np.mean(auc_values), 4),
        "f1": round(np.mean(f1_values), 4),
        "f2": round(np.mean(f2_values), 4),
        "ece": round(np.mean(ece_values), 4),
    }

    return results


def format_results_table(results: dict) -> pd.DataFrame:
    """Formatear resultados como DataFrame para impresion y reporte."""
    rows = []
    for label, metrics in results["per_label"].items():
        rows.append({
            "etiqueta": label,
            "AUC_ROC": metrics["auc_roc"]["point"],
            "AUC_IC95": f"[{metrics['auc_roc']['ci_lower']}, {metrics['auc_roc']['ci_upper']}]",
            "F1": metrics["f1_at_f2"],
            "F2": metrics["f2_at_f2"],
            "Prec": metrics["precision_at_f2"],
            "Recall": metrics["recall_at_f2"],
            "AP": metrics["avg_precision"],
            "ECE": metrics["ece"],
            "Umbral_F2": metrics["f2_threshold"],
        })

    # Agregar fila macro
    macro = results["macro"]
    rows.append({
        "etiqueta": "MACRO",
        "AUC_ROC": macro["auc_roc"],
        "AUC_IC95": "",
        "F1": macro["f1"],
        "F2": macro["f2"],
        "Prec": "",
        "Recall": "",
        "AP": "",
        "ECE": macro["ece"],
        "Umbral_F2": "",
    })

    return pd.DataFrame(rows)
