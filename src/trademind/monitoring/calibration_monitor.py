"""Calibration monitoring over rolling windows.

Calibration decays differently from discrimination. A model can keep ranking
correctly while its probabilities drift systematically high -- and because the
decision engine thresholds on probability, that alone changes trade frequency
without changing the ranking at all.

The observation floor matters more here than for AUC. ECE over 10 bins needs
enough rows that each bin holds a usable sample; with 105 observations a bin
holds 10, and its observed frequency has a standard error near 0.16. The
resulting ECE is dominated by binning noise. Windows below the floor are
reported as not-reportable rather than given a number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..ensemble.ensemble_metrics import calibration_metrics, reliability_table

# 10 bins x ~30 per bin. Below this, ECE measures the binning, not the model.
MIN_CALIBRATION_OBSERVATIONS = 300
DEFAULT_WINDOWS = (7, 30, 60, 90)


@dataclass(frozen=True)
class CalibrationWindow:
    window_days: int
    n_observations: int
    reportable: bool
    metrics: dict[str, float]

    @property
    def ece(self) -> float:
        return self.metrics.get("ece_quantile", float("nan"))

    def __str__(self) -> str:
        if not self.reportable:
            return (
                f"{self.window_days}d: {self.n_observations} obs — below the "
                f"{MIN_CALIBRATION_OBSERVATIONS} floor for a 10-bin ECE"
            )
        return (
            f"{self.window_days}d: n={self.n_observations:,} "
            f"ECE={self.ece:.4f} bias={self.metrics.get('bias', 0):+.4f} "
            f"sharpness={self.metrics.get('sharpness', 0):.4f}"
        )


def monitor_calibration(
    resolved: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    date_col: str = "prediction_date",
    prob_col: str = "calibrated_probability",
    outcome_col: str = "actual_direction",
    min_observations: int = MIN_CALIBRATION_OBSERVATIONS,
) -> list[CalibrationWindow]:
    """Calibration metrics per trailing window."""
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
        out.append(
            CalibrationWindow(
                window_days=days,
                n_observations=n,
                reportable=reportable,
                metrics=(
                    calibration_metrics(window[outcome_col], window[prob_col])
                    if reportable
                    else {}
                ),
            )
        )
    return out


def calibration_drifted(
    windows: list[CalibrationWindow],
    baseline_ece: float,
    tolerance: float = 0.05,
) -> tuple[bool, str]:
    """Has calibration degraded beyond the baseline plus a tolerance?

    The tolerance is deliberately wide. ECE on a few hundred observations is
    noisy, and a tight band would fire constantly.
    """
    reportable = [w for w in windows if w.reportable]
    if not reportable:
        return False, "No window has enough observations to assess calibration."

    longest = max(reportable, key=lambda w: w.n_observations)
    ece = longest.ece

    if not np.isfinite(ece):
        return False, "ECE unavailable."
    if ece <= baseline_ece + tolerance:
        return False, (
            f"ECE {ece:.4f} within tolerance of baseline {baseline_ece:.4f} "
            f"(+{tolerance:.2f})."
        )
    return True, (
        f"ECE {ece:.4f} exceeds baseline {baseline_ece:.4f} by more than "
        f"{tolerance:.2f} over the {longest.window_days}d window."
    )


def calibration_report(
    resolved: pd.DataFrame,
    prob_col: str = "calibrated_probability",
    outcome_col: str = "actual_direction",
) -> pd.DataFrame:
    """Reliability table over all resolved predictions."""
    data = resolved.dropna(subset=[prob_col, outcome_col])
    if data.empty:
        return pd.DataFrame()
    return reliability_table(data[outcome_col], data[prob_col])
