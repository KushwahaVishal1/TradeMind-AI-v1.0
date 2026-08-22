"""Performance monitoring, and the honest limits on what it can detect.

## The detectability problem

A rolling window can only detect a degradation larger than its own sampling
error. With 15 symbols:

=========  ===========  ===============  ==================
window     observations  AUC std. error   detectable drop
=========  ===========  ===============  ==================
7 days     105           0.195            0.39
30 days    450           0.094            0.19
90 days    1350          0.054            0.11
=========  ===========  ===============  ==================

The model's entire edge is roughly 0.02 above chance. Even a 90-day window
cannot detect a drop of 0.11 — the model could lose its whole signal five times
over and the monitor would see nothing but noise.

**Performance monitoring cannot detect degradation of this model at this signal
strength.** That is a real conclusion about the system, not a limitation to work
around, and it has a governance consequence: the retraining policy must not
treat an unchanged performance metric as evidence of health, because the metric
would look identical if the model had failed completely.

Every metric here is therefore reported with its minimum detectable effect, and
``degradation_detected()`` refuses to declare degradation smaller than the MDE.
A monitor that reports differences it cannot distinguish from noise generates
retraining cycles driven entirely by chance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..models.metrics import classification_metrics

log = logging.getLogger(__name__)

# Below this many resolved predictions, a window's metrics are not reportable.
MIN_WINDOW_OBSERVATIONS = 100
DEFAULT_WINDOWS = (7, 30, 60, 90)


def auc_minimum_detectable_effect(n: int, power_z: float = 2.0) -> float:
    """Smallest AUC change a window of ``n`` observations can distinguish.

    ``power_z = 2.0`` is roughly a 95% criterion. Returns the drop from 0.5 that
    would be needed; differences below it are not evidence of anything.
    """
    if n < 4:
        return float("nan")
    # Balanced-class approximation: SE ~ sqrt(1 / (n/4)).
    return float(power_z * np.sqrt(1.0 / (n / 4.0)))


@dataclass(frozen=True)
class WindowMetrics:
    """Metrics for one rolling window, with its detection limit attached."""

    window_days: int
    n_observations: int
    metrics: dict[str, float]
    minimum_detectable_effect: float
    reportable: bool

    @property
    def auc(self) -> float:
        return self.metrics.get("roc_auc", float("nan"))

    def __str__(self) -> str:
        if not self.reportable:
            return (
                f"{self.window_days}d: {self.n_observations} obs — below the "
                f"{MIN_WINDOW_OBSERVATIONS}-observation floor, not reportable"
            )
        return (
            f"{self.window_days}d: n={self.n_observations:,} AUC={self.auc:.4f} "
            f"(can detect changes >= {self.minimum_detectable_effect:.3f})"
        )


def rolling_windows(
    resolved: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    date_col: str = "prediction_date",
    prob_col: str = "calibrated_probability",
    outcome_col: str = "actual_direction",
    min_observations: int = MIN_WINDOW_OBSERVATIONS,
) -> list[WindowMetrics]:
    """Metrics over each trailing window, each with its own detection floor."""
    if resolved.empty:
        return []

    data = resolved.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    as_of = as_of or data[date_col].max()

    out = []
    for days in windows:
        cutoff = as_of - pd.Timedelta(days=days)
        window = data[(data[date_col] > cutoff) & (data[date_col] <= as_of)]
        window = window.dropna(subset=[prob_col, outcome_col])
        n = len(window)

        reportable = n >= min_observations
        metrics = (
            classification_metrics(window[outcome_col], window[prob_col])
            if reportable else {}
        )
        out.append(WindowMetrics(
            window_days=days,
            n_observations=n,
            metrics=metrics,
            minimum_detectable_effect=auc_minimum_detectable_effect(n),
            reportable=reportable,
        ))
    return out


def degradation_detected(
    recent: WindowMetrics,
    baseline_auc: float,
    require_above_mde: bool = True,
) -> tuple[bool, str]:
    """Has performance measurably degraded against a baseline?

    Returns ``(detected, explanation)``. When the observed drop is smaller than
    the window's minimum detectable effect, the answer is no — and the
    explanation says so rather than reporting a number that means nothing.
    """
    if not recent.reportable:
        return False, (
            f"{recent.window_days}d window has {recent.n_observations} "
            f"observations, below the {MIN_WINDOW_OBSERVATIONS} floor."
        )

    auc = recent.auc
    if not np.isfinite(auc) or not np.isfinite(baseline_auc):
        return False, "AUC unavailable."

    drop = baseline_auc - auc
    mde = recent.minimum_detectable_effect

    if drop <= 0:
        return False, f"AUC {auc:.4f} at or above baseline {baseline_auc:.4f}."

    if require_above_mde and drop < mde:
        return False, (
            f"AUC fell {drop:.4f} ({baseline_auc:.4f} -> {auc:.4f}), but the "
            f"{recent.window_days}d window can only detect changes of {mde:.3f}. "
            "Not distinguishable from noise."
        )

    return True, (
        f"AUC fell {drop:.4f} ({baseline_auc:.4f} -> {auc:.4f}), exceeding the "
        f"{mde:.3f} detection floor for a {recent.window_days}d window."
    )


@dataclass
class PerformanceMonitor:
    """Tracks resolved predictions against a fixed baseline."""

    baseline_auc: float
    baseline_brier: float | None = None
    windows: tuple[int, ...] = DEFAULT_WINDOWS

    def evaluate(
        self, resolved: pd.DataFrame, as_of: pd.Timestamp | None = None
    ) -> dict:
        window_metrics = rolling_windows(resolved, as_of, self.windows)
        reportable = [w for w in window_metrics if w.reportable]

        if not reportable:
            return {
                "status": "INSUFFICIENT_DATA",
                "windows": window_metrics,
                "detected": False,
                "explanation": (
                    "No window has enough resolved predictions to report. "
                    "Accumulate more outcomes before drawing conclusions."
                ),
            }

        # Judge on the longest reportable window: it has the most power, and
        # short windows on a weak signal are almost pure noise.
        longest = max(reportable, key=lambda w: w.n_observations)
        detected, explanation = degradation_detected(longest, self.baseline_auc)

        return {
            "status": "DEGRADED" if detected else "HEALTHY",
            "windows": window_metrics,
            "judged_on": longest,
            "detected": detected,
            "explanation": explanation,
            "current_auc": longest.auc,
            "baseline_auc": self.baseline_auc,
        }

    def render(self, result: dict) -> str:
        lines = [f"performance ({result['status']})"]
        for w in result["windows"]:
            lines.append(f"  {w}")
        lines.append(f"  {result['explanation']}")
        return "\n".join(lines)
