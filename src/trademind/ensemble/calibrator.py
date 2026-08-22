"""Probability calibration.

## The circularity trap

The obvious implementation is wrong::

    calibrator.fit(oof_probs, y)                    # fit
    calibrated = calibrator.predict(oof_probs)      # transform the same rows
    ece = expected_calibration_error(y, calibrated) # measure

That ECE will be near zero, always, for any model. Isotonic regression is
flexible enough to map any monotone input onto the observed frequencies of the
very rows it was fitted on. The number produced is a description of the
calibrator's capacity to fit, not evidence that future predictions will be
calibrated.

It is the same error as evaluating a model on its training set, but it hides
better, because "calibration" sounds like a post-processing step rather than a
fitted model. It is a fitted model.

``calibrate_out_of_fold`` is the honest version: the calibrator is fitted on
earlier folds and applied to a later one, using the same purged walk-forward
machinery as everything else. ``test_in_sample_calibration_is_circular``
demonstrates the gap between the two numbers.

## Isotonic or sigmoid

``sigmoid`` (Platt scaling)
    Fits two parameters. Stable on small samples, but can only apply a
    monotone S-shaped correction — if the miscalibration has a different shape,
    it will not fix it.

``isotonic``
    Non-parametric and monotone. Fixes any monotone distortion, needs far more
    data, and overfits badly below roughly a thousand observations. It also
    produces a step function, so calibrated probabilities cluster on a small set
    of distinct values.

Default is ``auto``: sigmoid under 1000 observations, isotonic above. Boosted
trees are typically overconfident at both extremes, which is a sigmoid-shaped
distortion, so sigmoid is often sufficient here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ISOTONIC_MIN_SAMPLES = 1000


class Calibrator:
    """Maps raw model scores onto calibrated probabilities."""

    def __init__(self, method: str = "auto") -> None:
        if method not in ("auto", "isotonic", "sigmoid", "identity"):
            raise ValueError(f"Unknown calibration method: {method}")
        self.method = method
        self.resolved_method_: str | None = None
        self._model = None

    def fit(self, probs, y_true) -> Calibrator:
        p = np.asarray(probs, dtype=float).ravel()
        y = (np.asarray(y_true, dtype=float).ravel() > 0.5).astype(int)

        ok = np.isfinite(p) & np.isfinite(y)
        p, y = p[ok], y[ok]
        if len(p) == 0:
            raise ValueError("No finite observations to calibrate on")

        method = self.method
        if method == "auto":
            method = "isotonic" if len(p) >= ISOTONIC_MIN_SAMPLES else "sigmoid"
            log.debug("auto-selected %s calibration for %d samples", method, len(p))

        if len(np.unique(y)) < 2:
            # One class only: nothing to learn, and both calibrators would
            # produce a degenerate constant.
            log.warning("Only one outcome class present; calibration is identity")
            method = "identity"

        if method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            if len(p) < ISOTONIC_MIN_SAMPLES:
                log.warning(
                    "Isotonic calibration on %d samples (< %d) will overfit; "
                    "sigmoid is safer at this size.",
                    len(p),
                    ISOTONIC_MIN_SAMPLES,
                )
            self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
                p, y
            )

        elif method == "sigmoid":
            from sklearn.linear_model import LogisticRegression

            self._model = LogisticRegression(solver="lbfgs").fit(p.reshape(-1, 1), y)

        else:
            self._model = None

        self.resolved_method_ = method
        return self

    def predict(self, probs) -> np.ndarray:
        p = np.asarray(probs, dtype=float).ravel()
        if self.resolved_method_ is None:
            raise RuntimeError("Calibrator is not fitted")
        if self.resolved_method_ == "identity":
            return np.clip(p, 0.0, 1.0)
        if self.resolved_method_ == "isotonic":
            return np.clip(self._model.predict(p), 0.0, 1.0)
        return self._model.predict_proba(p.reshape(-1, 1))[:, 1]


@dataclass
class CalibrationResult:
    """Out-of-fold calibrated predictions, with the before/after comparison."""

    predictions: pd.DataFrame  # date, symbol, y_true, raw, calibrated
    method: str
    n_folds: int

    def comparison(self, n_bins: int = 10) -> pd.DataFrame:
        from .ensemble_metrics import calibration_metrics

        before = calibration_metrics(
            self.predictions["y_true"], self.predictions["raw"], n_bins
        )
        after = calibration_metrics(
            self.predictions["y_true"], self.predictions["calibrated"], n_bins
        )
        return pd.DataFrame({"raw": before, "calibrated": after}).T

    def render(self) -> str:
        cmp = self.comparison()
        lines = [f"calibration ({self.method}, {self.n_folds} out-of-fold blocks)"]
        for col in ("ece_quantile", "mce", "brier", "sharpness", "bias"):
            if col in cmp.columns:
                lines.append(
                    f"  {col:<16} {cmp.loc['raw', col]:+.5f} -> "
                    f"{cmp.loc['calibrated', col]:+.5f}"
                )
        return "\n".join(lines)


def calibrate_out_of_fold(
    oof: pd.DataFrame,
    prob_col: str = "y_pred",
    label_col: str = "y_true",
    method: str = "auto",
    n_blocks: int = 4,
    min_fit_rows: int = 500,
) -> CalibrationResult:
    """Calibrate honestly: fit on earlier data, apply to later data.

    Walks forward through the OOF frame in time-ordered blocks. Block *k* is
    calibrated by a calibrator fitted on blocks ``0..k-1``, so no row is ever
    transformed by a calibrator that saw it. The first block has nothing to fit
    on and is dropped.

    This mirrors live behaviour exactly: in production the calibrator is fitted
    on history and applied to today.
    """
    if prob_col not in oof or label_col not in oof:
        raise ValueError(f"OOF frame needs '{prob_col}' and '{label_col}'")

    data = oof.sort_values("date").reset_index(drop=True)
    data = data[data[label_col].notna() & data[prob_col].notna()].reset_index(drop=True)
    if len(data) < min_fit_rows * 2:
        raise ValueError(
            f"Need at least {min_fit_rows * 2} labelled rows to calibrate "
            f"out-of-fold; got {len(data)}."
        )

    dates = pd.Index(sorted(data["date"].unique()))
    edges = np.array_split(np.arange(len(dates)), n_blocks)

    frames, resolved = [], method
    for k, block in enumerate(edges):
        if k == 0 or len(block) == 0:
            continue  # nothing earlier to fit on

        block_dates = set(dates[block])
        is_block = data["date"].isin(block_dates)
        is_prior = data["date"] < dates[block[0]]

        fit_rows = data[is_prior]
        if len(fit_rows) < min_fit_rows:
            log.debug("block %d: only %d prior rows, skipping", k, len(fit_rows))
            continue

        cal = Calibrator(method).fit(fit_rows[prob_col], fit_rows[label_col])
        resolved = cal.resolved_method_

        target = data[is_block].copy()
        frames.append(
            pd.DataFrame(
                {
                    "date": target["date"].to_numpy(),
                    "symbol": target.get("symbol", pd.Series([None] * len(target))).to_numpy(),
                    "block": k,
                    "y_true": target[label_col].to_numpy(),
                    "raw": target[prob_col].to_numpy(),
                    "calibrated": cal.predict(target[prob_col]),
                }
            )
        )

    if not frames:
        raise ValueError(
            "No block had enough prior data to calibrate on. Reduce n_blocks or min_fit_rows."
        )

    return CalibrationResult(
        predictions=pd.concat(frames, ignore_index=True),
        method=resolved,
        n_folds=len(frames),
    )


def fit_production_calibrator(
    oof: pd.DataFrame,
    prob_col: str = "y_pred",
    label_col: str = "y_true",
    method: str = "auto",
) -> Calibrator:
    """Fit the calibrator that ships, on all available development OOF data.

    Distinct from ``calibrate_out_of_fold``, which *measures* how well
    calibration generalises. This one is the artifact used at inference time,
    and it should use every row it legitimately can.

    Its quality is reported by the out-of-fold estimate, never by scoring it on
    its own training rows.
    """
    data = oof[oof[label_col].notna() & oof[prob_col].notna()]
    if data.empty:
        raise ValueError("No labelled rows to fit a production calibrator on")
    return Calibrator(method).fit(data[prob_col], data[label_col])
