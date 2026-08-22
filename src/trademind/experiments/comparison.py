"""Comparing experiments honestly.

A leaderboard sorted by validation metric is the natural output of an
experiment tracker and the most misleading thing it can produce. Two
experiments separated by 0.01 AUC, when the standard error is 0.024, are the
same experiment.

Everything here refuses to declare a winner that the data cannot support.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .tracker import DEFAULT_METRIC_STDERR, selection_inflation


def is_distinguishable(
    a: float, b: float, stderr: float = DEFAULT_METRIC_STDERR, z: float = 2.0
) -> bool:
    """Are two metric values separated by more than sampling noise?

    Uses the standard error of a *difference*, which is sqrt(2) larger than the
    standard error of either estimate -- a detail that is easy to omit and makes
    differences look more significant than they are.
    """
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    return abs(a - b) > z * stderr * np.sqrt(2)


def compare_experiments(
    runs, stderr: float = DEFAULT_METRIC_STDERR
) -> pd.DataFrame:
    """Rank experiments and mark which are actually distinguishable from the best."""
    rows = []
    values = [r.primary_value for r in runs if r.primary_value is not None]
    if not values:
        return pd.DataFrame()

    best = max(values)
    inflation = selection_inflation(len(runs), stderr)

    for r in runs:
        value = r.primary_value
        rows.append({
            "name": r.name,
            "model_type": r.model_type,
            "observed": value,
            "gap_to_best": None if value is None else best - value,
            "distinguishable_from_best": (
                False if value is None else is_distinguishable(value, best, stderr)
            ),
            "selection_adjusted": None if value is None else value - inflation,
            "reproducible": r.reproducible,
        })

    return pd.DataFrame(rows).sort_values(
        "observed", ascending=False, na_position="last"
    ).reset_index(drop=True)


def render_comparison(frame: pd.DataFrame, stderr: float = DEFAULT_METRIC_STDERR) -> str:
    """Comparison table plus the verdict on whether anything actually won."""
    if frame.empty:
        return "no comparable experiments"

    lines = [frame.to_string(index=False), ""]

    tied = frame[~frame["distinguishable_from_best"]]
    if len(tied) > 1:
        lines.append(
            f"{len(tied)} of {len(frame)} experiments are statistically "
            f"indistinguishable from the best (stderr {stderr:.3f}). Choosing "
            "among them on the validation metric alone is choosing on noise; "
            "prefer the simpler model."
        )
    else:
        lines.append(
            "The top experiment is distinguishable from the rest — but that is "
            "still a validation-set result, and the selection adjustment above "
            "applies."
        )
    return "\n".join(lines)
