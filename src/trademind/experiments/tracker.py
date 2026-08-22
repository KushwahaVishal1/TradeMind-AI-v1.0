"""Experiment tracking, and the selection bias tracking creates.

## The tracker's own failure mode

An experiment tracker makes it easy to run many configurations and pick the best
by validation metric. That is what it is for, and it is also a mechanism for
overfitting the validation set.

With an AUC standard error of 0.024 — the value from Phase 4 at realistic
sample sizes — and a model whose true AUC is 0.52:

======  ==================  ============
runs    E[best observed]    inflation
======  ==================  ============
1       0.5200              +0.0000
5       0.5432              +0.0232
10      0.5520              +0.0320
25      0.5625              +0.0425
50      0.5695              +0.0495
100     0.5759              +0.0559
======  ==================  ============

**Fifty experiments on a worthless model produce a best observed AUC of 0.57**,
which is squarely inside the range Phase 4's ``flag_suspicious`` warns about.
The experiment that "won" did so by drawing a lucky validation split, and
nothing about the winning run looks different from a genuine discovery.

This is not a reason to run fewer experiments. It is a reason to record how many
were run and subtract the expected inflation before believing any of them.
``selection_inflation()`` computes it, ``ExperimentTracker`` records
``n_prior_experiments`` on every run automatically, and ``adjusted_best()``
reports the discount.

The only clean answer remains the locked final test, used once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .run import ExperimentRun

log = logging.getLogger(__name__)

# AUC standard error at the sample sizes this project works with.
DEFAULT_METRIC_STDERR = 0.024


def selection_inflation(
    n_experiments: int, metric_stderr: float = DEFAULT_METRIC_STDERR
) -> float:
    """Expected inflation from picking the best of ``n_experiments``.

    Uses the expected maximum of *n* independent normal draws. Real experiments
    are correlated — they share data and often share most of their
    configuration — so this **overstates** the inflation somewhat. It is
    deliberately the conservative direction: better to under-believe a result
    than over-believe it.
    """
    if n_experiments <= 1:
        return 0.0
    from scipy import stats

    return float(metric_stderr * stats.norm.ppf(1 - 1 / (n_experiments + 1)))


def adjusted_best(
    observed_best: float,
    n_experiments: int,
    metric_stderr: float = DEFAULT_METRIC_STDERR,
) -> tuple[float, str]:
    """Discount an observed best for selection. Returns (adjusted, explanation)."""
    inflation = selection_inflation(n_experiments, metric_stderr)
    adjusted = observed_best - inflation

    if n_experiments <= 1:
        return observed_best, "Single experiment; no selection adjustment."

    return adjusted, (
        f"Best of {n_experiments} experiments scored {observed_best:.4f}. "
        f"Selecting the maximum of {n_experiments} noisy estimates inflates it "
        f"by about {inflation:.4f}, so the honest estimate is ~{adjusted:.4f}. "
        "Only the locked final test settles this."
    )


@dataclass
class ExperimentTracker:
    """Persists experiment runs and keeps the selection count honest."""

    conn: object

    def log(self, run: ExperimentRun) -> str:
        """Record an experiment, stamping how many preceded it for this task."""
        run.n_prior_experiments = self.count(task=run.task)

        row = run.to_row()
        cols = ", ".join(row)
        marks = ", ".join("?" * len(row))
        self.conn.execute(
            f"INSERT INTO experiments ({cols}) VALUES ({marks})",
            tuple(row.values()),
        )

        gaps = run.reproducibility_gaps()
        if gaps:
            log.warning(
                "Experiment %s is not fully reproducible: %s",
                run.experiment_id,
                "; ".join(gaps),
            )
        log.info(
            "Logged %s (%s=%s), experiment #%d for task '%s'",
            run.name,
            run.primary_metric,
            f"{run.primary_value:.4f}" if run.primary_value is not None else "n/a",
            run.n_prior_experiments + 1,
            run.task,
        )
        return run.experiment_id

    def count(self, task: str | None = None) -> int:
        sql = "SELECT COUNT(*) FROM experiments"
        params = ()
        if task:
            sql += " WHERE task = ?"
            params = (task,)
        return int(self.conn.execute(sql, params).fetchone()[0])

    def get(self, experiment_id: str) -> ExperimentRun | None:
        row = self.conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        return ExperimentRun.from_row(row) if row else None

    def find_duplicate(self, run: ExperimentRun) -> ExperimentRun | None:
        """An earlier experiment with an identical fingerprint.

        Re-running an identical configuration is fine, but the results should
        match. When they do not, something is non-deterministic and the
        reproducibility claim is false.
        """
        for existing in self.all(task=run.task):
            if existing.fingerprint == run.fingerprint:
                return existing
        return None

    def all(self, task: str | None = None) -> list[ExperimentRun]:
        sql = "SELECT * FROM experiments"
        params = ()
        if task:
            sql += " WHERE task = ?"
            params = (task,)
        sql += " ORDER BY created_at"
        return [ExperimentRun.from_row(r) for r in self.conn.execute(sql, params)]

    def leaderboard(
        self,
        task: str | None = None,
        metric_stderr: float = DEFAULT_METRIC_STDERR,
    ) -> pd.DataFrame:
        """Ranked experiments, with the selection discount attached.

        The ``selection_adjusted`` column is the point. A raw leaderboard
        invites reading the top row as the best model; the adjusted column says
        how much of that lead is expected to be luck.
        """
        runs = self.all(task)
        if not runs:
            return pd.DataFrame()

        n = len(runs)
        inflation = selection_inflation(n, metric_stderr)

        frame = pd.DataFrame(
            [
                {
                    "experiment_id": r.experiment_id,
                    "name": r.name,
                    "task": r.task,
                    "model_type": r.model_type,
                    "metric": r.primary_metric,
                    "observed": r.primary_value,
                    "selection_adjusted": (
                        r.primary_value - inflation if r.primary_value is not None else None
                    ),
                    "reproducible": r.reproducible,
                    "created_at": r.created_at,
                }
                for r in runs
            ]
        )

        return frame.sort_values("observed", ascending=False).reset_index(drop=True)

    def summary(self, task: str | None = None) -> str:
        runs = self.all(task)
        if not runs:
            return "no experiments logged"

        values = [r.primary_value for r in runs if r.primary_value is not None]
        if not values:
            return f"{len(runs)} experiments logged, none with a primary metric"

        best = max(values)
        adjusted, explanation = adjusted_best(best, len(runs))
        n_irreproducible = sum(1 for r in runs if not r.reproducible)

        lines = [
            f"{len(runs)} experiments" + (f" for task '{task}'" if task else ""),
            f"  best observed   {best:.4f}",
            f"  spread          {min(values):.4f} .. {best:.4f} "
            f"(sd {np.std(values, ddof=1) if len(values) > 1 else 0:.4f})",
            f"  {explanation}",
        ]
        if n_irreproducible:
            lines.append(
                f"  WARNING: {n_irreproducible} experiment(s) are not fully reproducible."
            )
        return "\n".join(lines)
