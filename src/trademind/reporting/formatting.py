"""Formatting for display.

One rule runs through this module: **a number is never shown without whatever
qualifies it.** An AUC appears with its standard error, a Sharpe with its
sample length, a drift count with the naive count it replaced.

The reason is specific to this project. Almost every honest finding here is a
finding about *uncertainty* — the edge is smaller than the cost, the degradation
is smaller than the detection floor, the winner is within selection noise. A
dashboard that prints bare numbers deletes all of that and leaves a page of
figures that look like results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Verdict(str, Enum):
    """How a headline number should be read."""

    GOOD = "GOOD"
    NEUTRAL = "NEUTRAL"
    CONCERN = "CONCERN"
    BAD = "BAD"
    UNKNOWN = "UNKNOWN"

    @property
    def symbol(self) -> str:
        return {
            "GOOD": "✓", "NEUTRAL": "–", "CONCERN": "!",
            "BAD": "✗", "UNKNOWN": "?",
        }[self.value]


@dataclass(frozen=True)
class Metric:
    """A number and everything needed to read it correctly."""

    label: str
    value: float | None
    unit: str = ""              # "", "%", "bps", "x"
    stderr: float | None = None
    baseline: float | None = None
    baseline_label: str = "baseline"
    verdict: Verdict = Verdict.NEUTRAL
    note: str = ""
    decimals: int = 4

    @property
    def available(self) -> bool:
        return self.value is not None and math.isfinite(self.value)

    def format_value(self) -> str:
        if not self.available:
            return "n/a"
        v = self.value
        if self.unit == "%":
            return f"{v:+.2%}" if v < 0 or self.label.lower().startswith(
                ("return", "cagr", "drawdown", "lift")
            ) else f"{v:.2%}"
        if self.unit == "bps":
            return f"{v * 10000:+.1f} bps"
        if self.unit == "x":
            return f"{v:.1f}x"
        return f"{v:.{self.decimals}f}"

    def format_full(self) -> str:
        """The value plus its uncertainty and baseline, when they exist."""
        if not self.available:
            return "n/a"

        parts = [self.format_value()]
        if self.stderr is not None and math.isfinite(self.stderr):
            parts.append(f"± {self.stderr:.{self.decimals}f}")
        if self.baseline is not None and math.isfinite(self.baseline):
            parts.append(
                f"(vs {self.baseline_label} "
                f"{self.baseline:.{self.decimals}f})"
            )
        return " ".join(parts)

    @property
    def within_noise(self) -> bool:
        """Is the gap to the baseline smaller than the sampling error?

        Uses the standard error of a difference, which is sqrt(2) larger than
        the standard error of either estimate.
        """
        if not (self.available and self.stderr and self.baseline is not None):
            return False
        return abs(self.value - self.baseline) < 2 * self.stderr * math.sqrt(2)


@dataclass(frozen=True)
class Panel:
    """A titled group of metrics with an overall verdict and a headline."""

    title: str
    verdict: Verdict
    headline: str
    metrics: tuple[Metric, ...] = ()
    caveats: tuple[str, ...] = ()

    def render(self) -> str:
        lines = [f"{self.verdict.symbol} {self.title}: {self.headline}"]
        for m in self.metrics:
            lines.append(f"    {m.label:<28} {m.format_full()}")
            if m.note:
                lines.append(f"      {m.note}")
        for c in self.caveats:
            lines.append(f"    ! {c}")
        return "\n".join(lines)


def format_sample_caveat(n_periods: int, kind: str = "sessions") -> str:
    """A standard warning about short samples."""
    years = n_periods / 252 if kind == "sessions" else n_periods
    if years >= 5:
        return ""
    if years >= 3:
        return (
            f"{years:.1f} years of data. Ratios are indicative; expect wide "
            "confidence intervals."
        )
    return (
        f"Only {years:.1f} years of data. Sharpe and similar ratios are not "
        "reliably estimable at this length."
    )


def pluralise(n: int, singular: str, plural: str | None = None) -> str:
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"
