"""Aggregate model health across every monitored dimension."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .alerts import AlertCollector
from .retraining import HealthState, MonitoringSignals, PolicyDecision


@dataclass
class HealthReport:
    """One snapshot of system health, and the policy's verdict on it."""

    as_of: str
    signals: MonitoringSignals
    decision: PolicyDecision
    alerts: AlertCollector
    performance: dict = field(default_factory=dict)
    calibration: list = field(default_factory=list)
    drift: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def state(self) -> HealthState:
        return self.decision.state

    def render(self) -> str:
        lines = [f"=== health report {self.as_of} ===", "", self.decision.render()]

        if self.performance:
            lines += ["", f"performance: {self.performance.get('explanation', '')}"]
            for w in self.performance.get("windows", []):
                lines.append(f"  {w}")

        if self.calibration:
            lines.append("")
            lines.append("calibration:")
            for w in self.calibration:
                lines.append(f"  {w}")

        if not self.drift.empty:
            from .feature_drift import summarise_drift

            s = summarise_drift(self.drift)
            lines += [
                "",
                f"drift: {s.get('n_significant', 0):.0f} significant, "
                f"{s.get('n_moderate', 0):.0f} moderate of "
                f"{s.get('n_measurable', 0):.0f} measurable features "
                f"(max PSI {s.get('max_psi', float('nan')):.3f})",
                f"  KS at p<0.05 naive: {s.get('n_ks_naive', 0):.0f} features | "
                f"after FDR control: {s.get('n_ks_fdr', 0):.0f}",
            ]

        if self.alerts.alerts:
            lines += ["", "alerts:", self.alerts.render()]

        return "\n".join(lines)
