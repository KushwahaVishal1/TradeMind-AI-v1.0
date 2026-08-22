"""End-to-end ensemble pipeline: base OOF -> stack -> calibrate -> report.

The sequencing is what matters:

1. Base models produce OOF predictions (Phase 4).
2. Those are joined into meta-features, checked for alignment.
3. The meta-model is fitted and evaluated under purged walk-forward.
4. Calibration is fitted and evaluated *out of fold on the meta-model's own OOF
   output*, never on the rows it was fitted on.
5. A production calibrator is fitted on everything, for use at inference.

Step 4 and step 5 are different artifacts with different jobs, which is the
distinction most implementations collapse. Step 4 tells you how well calibration
will generalise. Step 5 is the thing that ships. Reporting step 5's fit quality
as if it were step 4's result is the circularity described in ``calibrator.py``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from ..models.oof import OOFResult
from ..validation.purged_split import PurgedWalkForwardSplit
from .calibrator import (
    CalibrationResult,
    Calibrator,
    calibrate_out_of_fold,
    fit_production_calibrator,
)
from .ensemble_metrics import calibration_metrics, decompose_brier, render_reliability
from .stacking import StackResult, build_meta_features, fit_stack

log = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    """Everything Phase 6 needs, plus everything the report needs."""

    stack: StackResult
    calibration: CalibrationResult
    production_calibrator: Calibrator
    meta_features: pd.DataFrame

    @property
    def signals(self) -> pd.DataFrame:
        """Calibrated out-of-fold probabilities.

        The input to the decision engine's threshold optimiser. Every row was
        predicted by a meta-model that had not seen it and calibrated by a
        calibrator that had not seen it.
        """
        return self.calibration.predictions[
            ["date", "symbol", "y_true", "raw", "calibrated"]
        ].copy()

    def render(self) -> str:
        parts = [
            self.stack.render(),
            "",
            self.calibration.render(),
            "",
            "reliability (calibrated, out-of-fold):",
            render_reliability(
                self.calibration.predictions["y_true"],
                self.calibration.predictions["calibrated"],
            ),
        ]

        decomp = decompose_brier(
            self.calibration.predictions["y_true"],
            self.calibration.predictions["calibrated"],
        )
        if decomp:
            parts += [
                "",
                f"Brier decomposition: reliability {decomp['reliability']:.5f} "
                f"(lower better) | resolution {decomp['resolution']:.5f} "
                f"(higher better) | uncertainty {decomp['uncertainty']:.5f}",
            ]
            if decomp["resolution"] < decomp["reliability"]:
                parts.append(
                    "  Resolution below reliability: the forecast carries less "
                    "information than it has calibration error. Post-hoc "
                    "calibration cannot fix that — it needs a better signal."
                )
        return "\n".join(parts)


def build_ensemble(
    base_results: Mapping[str, OOFResult],
    context: pd.DataFrame | None = None,
    context_cols: Sequence[str] = (),
    splitter: PurgedWalkForwardSplit | None = None,
    calibration_method: str = "auto",
    calibration_blocks: int = 4,
) -> EnsembleResult:
    """Run the full stack-and-calibrate pipeline."""
    frames = {name: r.predictions for name, r in base_results.items()}

    base_metrics = pd.DataFrame(
        [
            {"model": name, **r.pooled_metrics}
            for name, r in base_results.items()
            if r.task == "direction"
        ]
    )

    meta = build_meta_features(frames, extra=context, extra_cols=context_cols)

    log.info("Fitting meta-model")
    stack = fit_stack(meta, splitter=splitter, base_metrics=base_metrics)

    log.info("Calibrating out-of-fold")
    calibration = calibrate_out_of_fold(
        stack.predictions, method=calibration_method, n_blocks=calibration_blocks
    )

    production = fit_production_calibrator(stack.predictions, method=calibration_method)

    before = calibration_metrics(
        calibration.predictions["y_true"], calibration.predictions["raw"]
    )
    after = calibration_metrics(
        calibration.predictions["y_true"], calibration.predictions["calibrated"]
    )
    log.info(
        "ECE %.4f -> %.4f | sharpness %.4f -> %.4f",
        before["ece_quantile"],
        after["ece_quantile"],
        before["sharpness"],
        after["sharpness"],
    )
    if after["ece_quantile"] > before["ece_quantile"]:
        log.warning(
            "Calibration made ECE worse out-of-fold (%.4f -> %.4f). The "
            "calibrator is fitting noise; try method='sigmoid' or more data.",
            before["ece_quantile"],
            after["ece_quantile"],
        )

    return EnsembleResult(stack, calibration, production, meta)
