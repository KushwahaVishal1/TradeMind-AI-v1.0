"""Calibration measurement: reliability curves, ECE, and the sharpness trade-off.

## The question this answers

Your original spec asked it directly: when the system says 80%, does it win 80%
of the time? That is *calibration*, and it is a different property from
*discrimination* (AUC — can the model rank winners above losers).

They are independent. A model can rank perfectly and still be badly calibrated
if its probabilities are all squeezed into 0.45-0.55. A model can be perfectly
calibrated and useless, by always predicting the base rate. The decision engine
in Phase 6 thresholds on probability, so calibration is what makes a threshold
mean something; without it, "buy above 0.70" is an arbitrary cut on an
arbitrary scale.

## Reading ECE

Expected Calibration Error is the average gap between predicted probability and
observed frequency, weighted by bin population. Zero is perfect.

Two things make it easy to misread:

**Bin choice matters.** Equal-width bins put almost every prediction in one or
two bins when a model is unsharp, and the near-empty tail bins then swing ECE
around wildly. This module defaults to **quantile bins**, which hold population
constant and are far more stable, and reports equal-width alongside so the
difference is visible rather than hidden in a methodology choice.

**ECE ignores sharpness.** A model that always predicts the base rate has an ECE
near zero and zero value. ``sharpness`` — the standard deviation of the
predictions — is reported next to it for exactly that reason. Low ECE with low
sharpness means a well-calibrated constant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(y_true, y_prob) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_prob, dtype=float).ravel()
    ok = np.isfinite(yt) & np.isfinite(yp)
    return (yt[ok] > 0.5).astype(int), np.clip(yp[ok], 0.0, 1.0)


def reliability_table(
    y_true, y_prob, n_bins: int = 10, strategy: str = "quantile"
) -> pd.DataFrame:
    """Per-bin predicted probability against observed frequency.

    The raw material of a reliability diagram, and more useful than the diagram
    for spotting where a model goes wrong — the ``count`` column shows which
    gaps are real and which are three observations in a tail bin.
    """
    yt, yp = _clean(y_true, y_prob)
    if len(yt) == 0:
        return pd.DataFrame()

    if strategy == "quantile":
        edges = np.unique(np.quantile(yp, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 2:
            edges = np.array([yp.min() - 1e-9, yp.max() + 1e-9])
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    idx = np.clip(np.digitize(yp, edges[1:-1], right=False), 0, len(edges) - 2)

    rows = []
    for b in range(len(edges) - 1):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append(
            {
                "bin": b,
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "count": n,
                "mean_predicted": float(yp[mask].mean()),
                "observed_frequency": float(yt[mask].mean()),
                "gap": float(yp[mask].mean() - yt[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true, y_prob, n_bins: int = 10, strategy: str = "quantile"
) -> float:
    """Population-weighted mean absolute gap between prediction and outcome."""
    table = reliability_table(y_true, y_prob, n_bins, strategy)
    if table.empty:
        return float("nan")
    weights = table["count"] / table["count"].sum()
    return float((weights * table["gap"].abs()).sum())


def maximum_calibration_error(
    y_true, y_prob, n_bins: int = 10, strategy: str = "quantile", min_count: int = 20
) -> float:
    """Worst per-bin gap, ignoring bins too small to be meaningful.

    Without ``min_count`` this metric is dominated by whichever tail bin holds
    four observations.
    """
    table = reliability_table(y_true, y_prob, n_bins, strategy)
    table = table[table["count"] >= min_count]
    return float(table["gap"].abs().max()) if not table.empty else float("nan")


def sharpness(y_prob) -> float:
    """Standard deviation of the predictions.

    The counterweight to ECE. A constant forecast is perfectly calibrated and
    perfectly useless; sharpness is what distinguishes the two cases.
    """
    yp = np.asarray(y_prob, dtype=float).ravel()
    yp = yp[np.isfinite(yp)]
    return float(np.std(yp)) if len(yp) else float("nan")


def calibration_metrics(y_true, y_prob, n_bins: int = 10) -> dict[str, float]:
    """Full calibration report, both binning strategies."""
    yt, yp = _clean(y_true, y_prob)
    if len(yt) == 0:
        return {}

    from sklearn.metrics import brier_score_loss

    base_rate = float(yt.mean())
    brier = float(brier_score_loss(yt, yp))

    return {
        "n": float(len(yt)),
        "ece_quantile": expected_calibration_error(yt, yp, n_bins, "quantile"),
        "ece_uniform": expected_calibration_error(yt, yp, n_bins, "uniform"),
        "mce": maximum_calibration_error(yt, yp, n_bins),
        "sharpness": sharpness(yp),
        "mean_predicted": float(yp.mean()),
        "base_rate": base_rate,
        # Systematic optimism or pessimism, independent of ranking.
        "bias": float(yp.mean() - base_rate),
        "brier": brier,
        "brier_baseline": float(base_rate * (1 - base_rate)),
    }


def decompose_brier(y_true, y_prob, n_bins: int = 10) -> dict[str, float]:
    """Murphy decomposition: Brier = reliability - resolution + uncertainty.

    Separates the two ways a probabilistic forecast can be good.

    ``reliability``
        Calibration error. Lower is better. Fixable by post-hoc calibration.
    ``resolution``
        How far the forecast moves away from the base rate while staying
        correct. **Higher is better.** This is the part that carries actual
        information, and post-hoc calibration cannot create it.
    ``uncertainty``
        Base-rate variance. A property of the problem, not the model.

    Useful because it says whether a mediocre Brier score is a calibration
    problem (fixable) or an information problem (not).
    """
    yt, yp = _clean(y_true, y_prob)
    if len(yt) == 0:
        return {}

    table = reliability_table(yt, yp, n_bins, "quantile")
    if table.empty:
        return {}

    n = len(yt)
    base_rate = float(yt.mean())

    reliability = float(
        (table["count"] * (table["mean_predicted"] - table["observed_frequency"]) ** 2).sum()
        / n
    )
    resolution = float(
        (table["count"] * (table["observed_frequency"] - base_rate) ** 2).sum() / n
    )
    uncertainty = base_rate * (1 - base_rate)

    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": float(uncertainty),
        "brier_reconstructed": reliability - resolution + uncertainty,
    }


def render_reliability(y_true, y_prob, n_bins: int = 10) -> str:
    """Text reliability diagram. Readable in a log without a plotting stack."""
    table = reliability_table(y_true, y_prob, n_bins)
    if table.empty:
        return "no data"

    lines = [f"{'predicted':>10} {'observed':>10} {'gap':>8} {'n':>7}"]
    for _, r in table.iterrows():
        lines.append(
            f"{r['mean_predicted']:>10.4f} {r['observed_frequency']:>10.4f} "
            f"{r['gap']:>+8.4f} {int(r['count']):>7,}"
        )
    m = calibration_metrics(y_true, y_prob, n_bins)
    lines.append(
        f"ECE {m['ece_quantile']:.4f} | MCE {m['mce']:.4f} | "
        f"sharpness {m['sharpness']:.4f} | bias {m['bias']:+.4f}"
    )
    return "\n".join(lines)
