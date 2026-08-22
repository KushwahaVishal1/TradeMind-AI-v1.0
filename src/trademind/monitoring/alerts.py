"""Alerts.

Deliberately austere. An alerting system that fires often is an alerting system
nobody reads, and the drift arithmetic in ``feature_drift.py`` shows how easily
that happens: naive per-feature testing would produce roughly 47 alerts a month
from noise alone.

So: severity is assigned from effect size rather than significance, INFO is not
alertable, and duplicate alerts within a cooldown window are suppressed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum


class Severity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


ALERTABLE = (Severity.WARNING, Severity.ERROR, Severity.CRITICAL)


@dataclass(frozen=True)
class Alert:
    severity: Severity
    code: str
    message: str
    context: dict = field(default_factory=dict)
    raised_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )

    @property
    def alertable(self) -> bool:
        return self.severity in ALERTABLE

    def __str__(self) -> str:
        return f"[{self.severity.value}] {self.code}: {self.message}"


@dataclass
class AlertCollector:
    """Gathers alerts and suppresses repeats within a cooldown."""

    cooldown_days: int = 3
    alerts: list[Alert] = field(default_factory=list)
    _last_seen: dict[str, date] = field(default_factory=dict)

    def add(
        self, severity: Severity, code: str, message: str, when: date | None = None, **context
    ) -> Alert | None:
        """Record an alert unless the same code fired recently."""
        alert = Alert(severity, code, message, context)
        when = when or date.today()

        previous = self._last_seen.get(code)
        if previous is not None and (when - previous).days < self.cooldown_days:
            return None

        self._last_seen[code] = when
        self.alerts.append(alert)
        return alert

    @property
    def alertable(self) -> list[Alert]:
        return [a for a in self.alerts if a.alertable]

    @property
    def worst(self) -> Severity:
        order = [Severity.INFO, Severity.WARNING, Severity.ERROR, Severity.CRITICAL]
        present = [a.severity for a in self.alerts]
        return max(present, key=order.index) if present else Severity.INFO

    def render(self) -> str:
        if not self.alerts:
            return "no alerts"
        return "\n".join(str(a) for a in self.alerts)
