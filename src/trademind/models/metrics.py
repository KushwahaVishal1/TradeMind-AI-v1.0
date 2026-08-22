"""Model metrics, always reported against a baseline.

## Why baselines are not optional here

Roughly 52% of daily equity sessions close up. A model that predicts "up" every
single day scores 52% accuracy, and 52% sounds like signal to anyone reading a
number without context. Reported alone, accuracy on a directional equity task is
close to meaningless.

Every classification report therefore carries the majority-class and
always-positive baselines beside it, plus ``skill_vs_majority`` — the actual lift.
Negative lift means the model is worse than a constant.

## What to expect

Next-day equity direction is close to unpredictable. Honest, leak-free ROC-AUC
on daily NSE large-caps lands around **0.51-0.54**. R-squared on the return
regression will be **negative** — predicting the mean beats predicting the
individual return, which is normal and not a bug.

A ROC-AUC of 0.65 on this task is a leak until proven otherwise. The first
response is to point the leakage suite at it, not to celebrate.

## Standard error

``auc_stderr`` gives the approximate sampling error of the AUC. With 600
validation observations it is around 0.024, so 0.53 and 0.55 are not
distinguishable. Reporting AUC without it invites reading noise as improvement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def _clean_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    ok = np.isfinite(yt) & np.isfinite(yp)
    return yt[ok], yp[ok]


def auc_stderr(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley-McNeil approximate standard error of the AUC.

    Turns "0.54" into "0.54 +/- 0.02", which is the difference between a claim
    and a number.
    """
    if n_pos < 1 or n_neg < 1:
        return float("nan")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc**2 / (1.0 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) / (
        n_pos * n_neg
    )
    return float(np.sqrt(max(var, 0.0)))


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    """Full directional metrics, with baselines attached."""
    yt, yp = _clean_pair(y_true, y_prob)
    if len(yt) == 0:
        return {}

    yt_bin = (yt > 0.5).astype(int)
    yp = np.clip(yp, 1e-7, 1 - 1e-7)
    pred = (yp >= threshold).astype(int)

    n_pos, n_neg = int(yt_bin.sum()), int((1 - yt_bin).sum())
    base_rate = float(yt_bin.mean())
    # Best achievable by any constant predictor.
    majority_acc = max(base_rate, 1.0 - base_rate)

    out: dict[str, float] = {
        "n": float(len(yt)),
        "base_rate": base_rate,
        "accuracy": float(accuracy_score(yt_bin, pred)),
        "precision": float(precision_score(yt_bin, pred, zero_division=0)),
        "recall": float(recall_score(yt_bin, pred, zero_division=0)),
        "f1": float(f1_score(yt_bin, pred, zero_division=0)),
        "brier": float(brier_score_loss(yt_bin, yp)),
        "log_loss": float(log_loss(yt_bin, yp, labels=[0, 1])),
        # Baselines.
        "majority_accuracy": majority_acc,
        "always_up_accuracy": base_rate,
    }

    if n_pos > 0 and n_neg > 0:
        auc = float(roc_auc_score(yt_bin, yp))
        out["roc_auc"] = auc
        out["auc_stderr"] = auc_stderr(auc, n_pos, n_neg)
        # How many standard errors above chance.
        out["auc_z"] = (auc - 0.5) / out["auc_stderr"] if out["auc_stderr"] else 0.0
        out["pr_auc"] = float(average_precision_score(yt_bin, yp))
        # PR-AUC has a floor at the base rate, unlike ROC-AUC's 0.5.
        out["pr_auc_baseline"] = base_rate
    else:
        out["roc_auc"] = float("nan")

    out["skill_vs_majority"] = out["accuracy"] - majority_acc
    # Brier of the constant base-rate forecast; the bar any probability must clear.
    out["brier_baseline"] = float(base_rate * (1 - base_rate))
    out["brier_skill_score"] = (
        1.0 - out["brier"] / out["brier_baseline"]
        if out["brier_baseline"] > 0
        else float("nan")
    )
    return out


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return-prediction metrics.

    ``information_coefficient`` (Spearman rank correlation) is the one that
    matters for a trading system. A strategy acts on the *ordering* of expected
    returns across names, not their absolute magnitudes, so rank correlation
    measures the useful part. An IC of 0.02-0.05 is a real daily signal; MAE and
    RMSE are dominated by volatility and barely move between a good model and a
    useless one.
    """
    yt, yp = _clean_pair(y_true, y_pred)
    if len(yt) < 2:
        return {}

    out: dict[str, float] = {
        "n": float(len(yt)),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
        "r2": float(r2_score(yt, yp)),
        # Predicting the training mean. R2 below this is worse than a constant.
        "mae_baseline": float(mean_absolute_error(yt, np.full_like(yt, yt.mean()))),
    }

    if np.std(yp) > 1e-12 and np.std(yt) > 1e-12:
        out["pearson_ic"] = float(np.corrcoef(yt, yp)[0, 1])
        out["information_coefficient"] = float(
            pd.Series(yt).corr(pd.Series(yp), method="spearman")
        )
    else:
        # A constant prediction has no correlation. Worth surfacing, since a
        # collapsed model can otherwise look merely mediocre.
        out["pearson_ic"] = 0.0
        out["information_coefficient"] = 0.0

    # Directional hit rate implied by the sign of the predicted return.
    nonzero = yt != 0
    out["directional_accuracy"] = (
        float(np.mean(np.sign(yp[nonzero]) == np.sign(yt[nonzero])))
        if nonzero.any()
        else float("nan")
    )
    out["up_rate"] = float((yt > 0).mean())
    return out


def summarise(metrics: dict[str, float], kind: str = "classification") -> str:
    """One-line human summary that leads with the comparison, not the number."""
    if not metrics:
        return "no metrics"

    if kind == "classification":
        auc = metrics.get("roc_auc", float("nan"))
        se = metrics.get("auc_stderr", float("nan"))
        return (
            f"AUC {auc:.4f} +/- {se:.4f} (z={metrics.get('auc_z', 0):.1f}) | "
            f"acc {metrics['accuracy']:.4f} vs majority "
            f"{metrics['majority_accuracy']:.4f} "
            f"(lift {metrics['skill_vs_majority']:+.4f}) | "
            f"brier {metrics['brier']:.5f} (BSS {metrics.get('brier_skill_score', 0):+.4f})"
        )

    return (
        f"IC {metrics.get('information_coefficient', 0):+.4f} | "
        f"dir_acc {metrics.get('directional_accuracy', 0):.4f} | "
        f"MAE {metrics['mae']:.5f} vs baseline {metrics['mae_baseline']:.5f} | "
        f"R2 {metrics['r2']:+.4f}"
    )


def flag_suspicious(metrics: dict[str, float], kind: str = "classification") -> list[str]:
    """Warnings for results too good to be true on this task.

    Called automatically after every evaluation. On daily equity direction,
    strong results are far more likely to indicate a leak than a discovery, and
    the useful moment to say so is immediately — before the number gets written
    down somewhere.
    """
    warnings: list[str] = []

    if kind == "classification":
        auc = metrics.get("roc_auc", float("nan"))
        if np.isfinite(auc):
            if auc > 0.65:
                warnings.append(
                    f"ROC-AUC {auc:.3f} is implausibly high for next-day equity "
                    "direction. Suspect leakage before skill; run the leakage suite."
                )
            elif auc > 0.58:
                warnings.append(
                    f"ROC-AUC {auc:.3f} is above the plausible range (0.51-0.54). "
                    "Verify feature timing."
                )
        if metrics.get("skill_vs_majority", 0) > 0.10:
            warnings.append(
                f"Accuracy exceeds the majority baseline by "
                f"{metrics['skill_vs_majority']:.3f}, which is far beyond what "
                "this task supports."
            )
    else:
        ic = abs(metrics.get("information_coefficient", 0.0))
        if ic > 0.20:
            warnings.append(
                f"|IC| {ic:.3f} is implausibly high for daily returns "
                "(0.02-0.05 is a genuinely good signal)."
            )
        if metrics.get("r2", -1) > 0.10:
            warnings.append(
                f"R2 {metrics['r2']:.3f} on daily returns is not plausible; "
                "expect a value near or below zero."
            )
    return warnings
