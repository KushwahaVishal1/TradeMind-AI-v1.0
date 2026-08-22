"""Retraining governance: the state machine and the gates that guard it.

::

    HEALTHY
       |
    WARNING
       |
    DEGRADED
       |
    RETRAIN_REQUIRED
       |
    RETRAINING
       |
    VALIDATING
       |-- FAIL --> KEEP_CURRENT_MODEL
       +-- PASS --> PROMOTE_NEW_MODEL

## The four policy rules, and why each exists

**1. Drift alone never triggers retraining.**
Markets change distribution constantly; that is what markets do. A model can be
entirely healthy on shifted inputs. Retraining on drift alone means retraining
continuously, and every retrain is a fresh opportunity to overfit to whatever
the last few months happened to look like.

**2. A data-quality ERROR blocks retraining, however severe everything else is.**
This is the rule that feels wrong and is right. When performance has collapsed
*and* the ingestion layer is reporting errors, the collapse is more likely
caused by the bad data than by model decay — and retraining on bad data bakes
the corruption into the replacement. Fix the data first. Ordering the gates so
this one wins is the whole point of having a policy rather than a threshold.

**3. Insufficient new observations means wait.**
Retraining on a handful of new rows produces a model that differs from its
predecessor by noise. The candidate then wins or loses promotion by chance.

**4. A failed candidate has no path to production.**
Structural, not procedural: ``PromotionDecision`` cannot be constructed with
``promote=True`` when validation failed. There is no flag to override it.

## Diagnosis, not just detection

When retraining is warranted the policy also states *why*, because the cause
determines whether retraining is even the right response:

- **performance down + drift** — the input distribution moved and the model did
  not follow. Retraining is likely to help.
- **performance down, no drift** — the relationship between features and target
  changed while the inputs look the same. Retraining on more of the same
  features may not help; the feature set is the suspect.
- **drift, performance stable** — the world moved and the model is coping.
  Monitor, do nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

log = logging.getLogger(__name__)

MIN_NEW_OBSERVATIONS = 500


class HealthState(str, Enum):
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    DEGRADED = "DEGRADED"
    RETRAIN_REQUIRED = "RETRAIN_REQUIRED"
    RETRAINING = "RETRAINING"
    VALIDATING = "VALIDATING"
    KEEP_CURRENT_MODEL = "KEEP_CURRENT_MODEL"
    PROMOTE_NEW_MODEL = "PROMOTE_NEW_MODEL"
    BLOCKED = "BLOCKED"


class Diagnosis(str, Enum):
    NONE = "NONE"
    LIKELY_REGIME_CHANGE = "LIKELY_REGIME_CHANGE"
    LIKELY_RELATIONSHIP_CHANGE = "LIKELY_RELATIONSHIP_CHANGE"
    DRIFT_ONLY_MODEL_COPING = "DRIFT_ONLY_MODEL_COPING"
    DATA_QUALITY_SUSPECTED = "DATA_QUALITY_SUSPECTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class MonitoringSignals:
    """The inputs the policy reasons over."""

    performance_degraded: bool = False
    performance_detectable: bool = True
    calibration_drifted: bool = False
    n_features_drifting: int = 0
    n_features_total: int = 0
    data_quality_errors: int = 0
    data_quality_warnings: int = 0
    n_new_observations: int = 0

    @property
    def drift_fraction(self) -> float:
        return (
            self.n_features_drifting / self.n_features_total
            if self.n_features_total else 0.0
        )

    @property
    def has_drift(self) -> bool:
        return self.n_features_drifting > 0


@dataclass(frozen=True)
class PolicyDecision:
    """The policy's verdict, with its reasoning."""

    state: HealthState
    diagnosis: Diagnosis
    reasons: tuple[str, ...]
    should_retrain: bool

    def render(self) -> str:
        lines = [f"{self.state.value} — {self.diagnosis.value}"]
        for r in self.reasons:
            lines.append(f"  - {r}")
        lines.append(
            f"  retrain: {'YES' if self.should_retrain else 'no'}"
        )
        return "\n".join(lines)


class RetrainingPolicy:
    """Decides whether retraining is warranted, and says why."""

    def __init__(
        self,
        min_new_observations: int = MIN_NEW_OBSERVATIONS,
        drift_fraction_warning: float = 0.20,
    ) -> None:
        self.min_new_observations = min_new_observations
        self.drift_fraction_warning = drift_fraction_warning

    def evaluate(self, signals: MonitoringSignals) -> PolicyDecision:
        reasons: list[str] = []

        # --- Gate 1: data quality wins over everything ------------------
        if signals.data_quality_errors > 0:
            return PolicyDecision(
                state=HealthState.BLOCKED,
                diagnosis=Diagnosis.DATA_QUALITY_SUSPECTED,
                reasons=(
                    f"{signals.data_quality_errors} unresolved data-quality "
                    "ERROR(s). Retraining is blocked regardless of other "
                    "signals — a model trained on bad data bakes the corruption "
                    "in. Fix ingestion first.",
                ),
                should_retrain=False,
            )

        # --- Gate 2: enough new data to learn from ----------------------
        insufficient = signals.n_new_observations < self.min_new_observations
        if insufficient:
            reasons.append(
                f"Only {signals.n_new_observations} new observations "
                f"(need {self.min_new_observations}); a model fitted on this "
                "would differ from its predecessor by noise."
            )

        # --- Collect evidence -------------------------------------------
        if signals.performance_degraded:
            reasons.append("Performance degraded beyond the detection floor.")
        elif not signals.performance_detectable:
            reasons.append(
                "Performance change is below the window's minimum detectable "
                "effect — unchanged metrics are NOT evidence of health here."
            )

        if signals.calibration_drifted:
            reasons.append("Calibration has drifted beyond tolerance.")

        if signals.has_drift:
            reasons.append(
                f"{signals.n_features_drifting}/{signals.n_features_total} "
                f"features show persistent drift ({signals.drift_fraction:.0%})."
            )

        # --- Gate 3: drift alone is never enough ------------------------
        performance_problem = (
            signals.performance_degraded or signals.calibration_drifted
        )

        if signals.has_drift and not performance_problem:
            return PolicyDecision(
                state=(
                    HealthState.WARNING
                    if signals.drift_fraction >= self.drift_fraction_warning
                    else HealthState.HEALTHY
                ),
                diagnosis=Diagnosis.DRIFT_ONLY_MODEL_COPING,
                reasons=tuple(reasons + [
                    "Drift without a performance problem does not justify "
                    "retraining. The distribution moved; the model is coping."
                ]),
                should_retrain=False,
            )

        if not performance_problem:
            return PolicyDecision(
                state=HealthState.HEALTHY,
                diagnosis=Diagnosis.NONE,
                reasons=tuple(reasons) or ("All monitored signals nominal.",),
                should_retrain=False,
            )

        # --- Performance problem: diagnose the cause --------------------
        diagnosis = (
            Diagnosis.LIKELY_REGIME_CHANGE if signals.has_drift
            else Diagnosis.LIKELY_RELATIONSHIP_CHANGE
        )
        reasons.append(
            "Performance fell alongside input drift: the distribution moved "
            "and the model did not follow. Retraining is likely to help."
            if signals.has_drift else
            "Performance fell with stable inputs: the feature-target "
            "relationship changed. Retraining on the same features may not "
            "help — investigate the feature set."
        )

        if insufficient:
            return PolicyDecision(
                state=HealthState.DEGRADED,
                diagnosis=diagnosis,
                reasons=tuple(reasons + [
                    "Waiting for more observations before retraining."
                ]),
                should_retrain=False,
            )

        return PolicyDecision(
            state=HealthState.RETRAIN_REQUIRED,
            diagnosis=diagnosis,
            reasons=tuple(reasons),
            should_retrain=True,
        )


# =====================================================================
# Candidate validation and promotion
# =====================================================================

@dataclass(frozen=True)
class ValidationOutcome:
    """Result of validating a retrained candidate."""

    passed: bool
    candidate_version: str
    incumbent_version: str
    checks: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def failed_checks(self) -> list[str]:
        return [k for k, v in self.checks.items() if not v]


class PromotionBlocked(RuntimeError):
    """An attempt to promote a candidate that did not pass validation."""


@dataclass(frozen=True)
class PromotionDecision:
    """Whether a candidate replaces the incumbent.

    ``promote=True`` alongside a failed validation is rejected in
    ``__post_init__``. There is no override flag: a failed candidate has no path
    to production, and making that a runtime construction error rather than a
    code-review convention is the difference between a policy and a hope.
    """

    promote: bool
    outcome: ValidationOutcome
    decided_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    reason: str = ""

    def __post_init__(self) -> None:
        if self.promote and not self.outcome.passed:
            raise PromotionBlocked(
                f"Candidate {self.outcome.candidate_version} failed validation "
                f"({', '.join(self.outcome.failed_checks) or 'unspecified'}) and "
                "cannot be promoted. There is no override."
            )

    @property
    def final_state(self) -> HealthState:
        return (
            HealthState.PROMOTE_NEW_MODEL if self.promote
            else HealthState.KEEP_CURRENT_MODEL
        )


def validate_candidate(
    candidate_version: str,
    incumbent_version: str,
    candidate_metrics: dict[str, float],
    incumbent_metrics: dict[str, float],
    min_auc_improvement: float = 0.0,
    max_ece_regression: float = 0.02,
) -> ValidationOutcome:
    """Check a candidate against the incumbent on multiple axes.

    Promotion is never decided on a single metric. A candidate with a better
    AUC and much worse calibration would change trade frequency through the
    decision engine's thresholds while looking like an improvement.
    """
    checks: dict[str, bool] = {}
    notes: list[str] = []

    cand_auc = candidate_metrics.get("roc_auc", float("nan"))
    inc_auc = incumbent_metrics.get("roc_auc", float("nan"))
    checks["auc_not_worse"] = bool(cand_auc >= inc_auc + min_auc_improvement)
    if not checks["auc_not_worse"]:
        notes.append(f"AUC {cand_auc:.4f} vs incumbent {inc_auc:.4f}.")

    cand_ece = candidate_metrics.get("ece_quantile", 0.0)
    inc_ece = incumbent_metrics.get("ece_quantile", 0.0)
    checks["calibration_not_worse"] = bool(cand_ece <= inc_ece + max_ece_regression)
    if not checks["calibration_not_worse"]:
        notes.append(f"ECE {cand_ece:.4f} vs incumbent {inc_ece:.4f}.")

    # A candidate that beats the incumbent while predicting nothing is a
    # well-calibrated constant, not an improvement.
    sharpness = candidate_metrics.get("sharpness", float("nan"))
    checks["produces_varied_predictions"] = bool(
        sharpness > 0.001 if sharpness == sharpness else False
    )
    if not checks["produces_varied_predictions"]:
        notes.append("Candidate predictions are nearly constant.")

    checks["sufficient_evaluation_data"] = bool(
        candidate_metrics.get("n", 0) >= MIN_NEW_OBSERVATIONS
    )

    passed = all(checks.values())
    return ValidationOutcome(
        passed=passed,
        candidate_version=candidate_version,
        incumbent_version=incumbent_version,
        checks=checks,
        metrics={"candidate_auc": cand_auc, "incumbent_auc": inc_auc,
                 "candidate_ece": cand_ece, "incumbent_ece": inc_ece},
        notes=tuple(notes),
    )


def decide_promotion(outcome: ValidationOutcome) -> PromotionDecision:
    """Turn a validation outcome into a promotion decision. Never overrides."""
    if outcome.passed:
        return PromotionDecision(
            promote=True, outcome=outcome,
            reason=f"{outcome.candidate_version} passed every validation check.",
        )
    return PromotionDecision(
        promote=False, outcome=outcome,
        reason=(
            f"{outcome.candidate_version} failed: "
            f"{', '.join(outcome.failed_checks)}. Incumbent "
            f"{outcome.incumbent_version} retained."
        ),
    )
