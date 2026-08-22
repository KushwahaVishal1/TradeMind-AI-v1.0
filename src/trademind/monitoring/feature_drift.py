"""Feature drift detection.

## The multiple-testing problem, quantified

The obvious design runs a KS test per feature per day and alerts on p < 0.05.
With 45 features::

    expected false alarms per day        2.2
    P(at least one false alarm today)    0.90
    expected false alarms per month      47

A monitor that cries wolf on 90% of days is worse than no monitor, because the
team learns to dismiss it and then dismisses the real one too. This is the
single most common defect in drift monitoring and it is invisible until someone
counts.

Three responses, all applied here:

**Effect size, not significance.** PSI measures *how much* a distribution moved;
a p-value measures how confident we are that it moved at all. With thousands of
observations, a p-value detects shifts far too small to matter. PSI is the
primary signal; the conventional bands are < 0.10 stable, 0.10-0.25 moderate,
> 0.25 significant.

**Persistence.** A single day above threshold is noise. A feature must breach
for ``persistence_days`` consecutive checks before it is reported as drifting.
This costs a few days of detection latency and removes most of the false alarms.

**FDR control on the p-values that remain.** Where KS is used, Benjamini-Hochberg
controls the false discovery rate across the whole feature set rather than
per-feature. Bonferroni would be the alternative; it is far too conservative for
45 correlated features and would detect nothing.

## Drift is a symptom, never a trigger

Per the governance policy, feature drift alone never causes retraining. Markets
change distribution constantly — that is what markets do — and a model can be
perfectly healthy on shifted inputs. Drift raises a flag that makes a
*performance* problem interpretable; on its own it means the world moved, not
that the model broke.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
DEFAULT_PERSISTENCE_DAYS = 3
MIN_OBSERVATIONS = 100


@dataclass(frozen=True)
class DriftResult:
    """Drift measurement for one feature."""

    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    n_reference: int
    n_current: int
    reference_mean: float
    current_mean: float
    missing_rate_reference: float
    missing_rate_current: float
    insufficient_data: bool = False

    @property
    def severity(self) -> str:
        if self.insufficient_data:
            return "UNKNOWN"
        if self.psi >= PSI_SIGNIFICANT:
            return "SIGNIFICANT"
        if self.psi >= PSI_MODERATE:
            return "MODERATE"
        return "STABLE"


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, n_bins: int = 10
) -> float:
    """PSI between two samples, using reference quantile bins.

    Bins come from the *reference* distribution so that the comparison is
    against a fixed yardstick. Re-binning on the current sample each period
    would move the yardstick with the data and understate the shift.
    """
    ref = reference[np.isfinite(reference)]
    cur = current[np.isfinite(current)]
    if len(ref) < 2 or len(cur) < 2:
        return float("nan")

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return 0.0  # near-constant feature; no meaningful distribution
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    # Laplace smoothing: an empty bin would otherwise send PSI to infinity.
    ref_pct = (ref_counts + 1) / (ref_counts.sum() + len(ref_counts))
    cur_pct = (cur_counts + 1) / (cur_counts.sum() + len(cur_counts))

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Which hypotheses survive FDR control at level ``alpha``.

    Controls the expected *proportion* of false discoveries among rejections,
    rather than the probability of any false discovery. With 45 correlated
    features, Bonferroni would demand p < 0.0011 and detect essentially nothing.
    """
    p = np.asarray(pvalues, dtype=float)
    valid = np.isfinite(p)
    out = np.zeros(len(p), dtype=bool)
    if not valid.any():
        return out

    idx = np.where(valid)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)

    thresholds = alpha * np.arange(1, m + 1) / m
    passing = p[order] <= thresholds
    if not passing.any():
        return out

    cutoff = np.max(np.where(passing)[0])
    out[order[: cutoff + 1]] = True
    return out


def compute_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    n_bins: int = 10,
    min_observations: int = MIN_OBSERVATIONS,
) -> pd.DataFrame:
    """Per-feature drift between a reference window and the current one."""
    from scipy import stats

    results = []
    for feature in features:
        if feature not in reference.columns or feature not in current.columns:
            continue

        ref_all = reference[feature].to_numpy(dtype=float)
        cur_all = current[feature].to_numpy(dtype=float)
        ref = ref_all[np.isfinite(ref_all)]
        cur = cur_all[np.isfinite(cur_all)]

        insufficient = len(ref) < min_observations or len(cur) < min_observations

        if insufficient:
            psi, ks_stat, ks_p = float("nan"), float("nan"), float("nan")
        else:
            psi = population_stability_index(ref, cur, n_bins)
            ks_stat, ks_p = stats.ks_2samp(ref, cur)

        results.append(
            DriftResult(
                feature=feature,
                psi=psi,
                ks_statistic=float(ks_stat),
                ks_pvalue=float(ks_p),
                n_reference=len(ref),
                n_current=len(cur),
                reference_mean=float(ref.mean()) if len(ref) else float("nan"),
                current_mean=float(cur.mean()) if len(cur) else float("nan"),
                missing_rate_reference=float(1 - len(ref) / max(1, len(ref_all))),
                missing_rate_current=float(1 - len(cur) / max(1, len(cur_all))),
                insufficient_data=insufficient,
            )
        )

    if not results:
        return pd.DataFrame()

    frame = pd.DataFrame([r.__dict__ for r in results])
    frame["severity"] = [r.severity for r in results]

    # FDR across the whole feature set, not per feature.
    frame["ks_significant_fdr"] = benjamini_hochberg(frame["ks_pvalue"].to_numpy())
    return frame.sort_values("psi", ascending=False).reset_index(drop=True)


@dataclass
class DriftTracker:
    """Requires a breach to persist before reporting it.

    A single day above threshold is noise. Holding state across checks converts
    a jumpy per-day signal into one worth acting on, at the cost of a few days
    of latency.
    """

    persistence_days: int = DEFAULT_PERSISTENCE_DAYS
    psi_threshold: float = PSI_SIGNIFICANT
    _streaks: dict[str, int] = field(default_factory=dict)

    def update(self, drift: pd.DataFrame) -> list[str]:
        """Record one check. Returns features that have breached persistently."""
        if drift.empty:
            return []

        breaching = set(drift.loc[drift["psi"] >= self.psi_threshold, "feature"])
        for feature in drift["feature"]:
            if feature in breaching:
                self._streaks[feature] = self._streaks.get(feature, 0) + 1
            else:
                self._streaks[feature] = 0

        confirmed = [
            f for f, streak in self._streaks.items() if streak >= self.persistence_days
        ]
        if confirmed:
            log.warning(
                "Persistent drift in %d feature(s) over %d consecutive checks: %s",
                len(confirmed),
                self.persistence_days,
                confirmed[:5],
            )
        return sorted(confirmed)

    def streak(self, feature: str) -> int:
        return self._streaks.get(feature, 0)

    def reset(self) -> None:
        self._streaks.clear()


def summarise_drift(drift: pd.DataFrame) -> dict[str, float]:
    """Aggregate drift statistics for the health report."""
    if drift.empty:
        return {}

    measurable = drift[~drift["insufficient_data"]]
    return {
        "n_features": float(len(drift)),
        "n_measurable": float(len(measurable)),
        "n_significant": float((measurable["severity"] == "SIGNIFICANT").sum()),
        "n_moderate": float((measurable["severity"] == "MODERATE").sum()),
        "max_psi": float(measurable["psi"].max()) if len(measurable) else float("nan"),
        "mean_psi": float(measurable["psi"].mean()) if len(measurable) else float("nan"),
        # Naive count, kept only to show how much noise FDR removes.
        "n_ks_naive": float((measurable["ks_pvalue"] < 0.05).sum()),
        "n_ks_fdr": float(measurable["ks_significant_fdr"].sum()),
        "max_missing_rate": float(measurable["missing_rate_current"].max())
        if len(measurable)
        else float("nan"),
    }
