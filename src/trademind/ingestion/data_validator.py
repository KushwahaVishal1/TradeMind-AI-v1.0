"""Data-quality validation.

Design rule from the roadmap, taken literally: **report, never silently fix.**

An auto-repairing ingestion layer is worse than a broken one. If a bad bar is
quietly interpolated, the model trains on a number the market never printed and
nothing downstream can tell. So every check here emits a finding; the caller
decides what to do, and ERROR-severity findings block the pipeline.

Severity contract:

``ERROR``
    The data is internally inconsistent or unusable. Blocks ingestion and, per
    the Phase 8 policy, blocks retraining even when every other signal is fine.

``WARNING``
    Plausible but suspicious. Ingestion proceeds; the finding is persisted.

``INFO``
    Expected under a known limitation — most often the approximate calendar.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .base import PRICE_COLUMNS_SPLIT
from .calendar import TradingCalendar

# A single-session move beyond this is either a real shock, an unrecorded
# split, or bad data. 20% on a large-cap NSE name is rare enough to be worth
# a human look and common enough that ERROR would be wrong.
JUMP_WARN_THRESHOLD = 0.20
# Beyond this, an unrecorded corporate action is far more likely than a real move.
JUMP_ERROR_THRESHOLD = 0.50


@dataclass(frozen=True)
class Issue:
    severity: str
    code: str
    detail: str
    symbol: str | None = None
    issue_date: date | None = None

    def __str__(self) -> str:
        where = f"{self.symbol or '-'} {self.issue_date or '-'}"
        return f"[{self.severity}] {self.code} ({where}): {self.detail}"


@dataclass
class ValidationReport:
    symbol: str
    issues: list[Issue]

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "ERROR"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "WARNING"]

    @property
    def ok(self) -> bool:
        """True when nothing blocks ingestion. Warnings do not block."""
        return not self.errors

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for i in self.issues:
            counts[i.severity] = counts.get(i.severity, 0) + 1
        if not counts:
            return f"{self.symbol}: clean"
        parts = [f"{n} {sev.lower()}" for sev, n in sorted(counts.items())]
        return f"{self.symbol}: {', '.join(parts)}"


class DataValidator:
    """Runs every check against one symbol's bars."""

    def __init__(self, calendar: TradingCalendar | None = None) -> None:
        self.calendar = calendar or TradingCalendar()

    def validate(self, df: pd.DataFrame, symbol: str) -> ValidationReport:
        issues: list[Issue] = []

        if df.empty:
            return ValidationReport(
                symbol, [Issue("ERROR", "NO_DATA", "Provider returned zero rows", symbol)]
            )

        issues += self._check_chronology(df, symbol)
        issues += self._check_duplicates(df, symbol)
        issues += self._check_prices_positive(df, symbol)
        issues += self._check_ohlc_consistency(df, symbol)
        issues += self._check_volume(df, symbol)
        issues += self._check_jumps(df, symbol)
        issues += self._check_calendar(df, symbol)
        issues += self._check_corporate_actions(df, symbol)
        issues += self._check_staleness(df, symbol)

        return ValidationReport(symbol, issues)

    # -- individual checks ------------------------------------------------

    @staticmethod
    def _check_chronology(df: pd.DataFrame, symbol: str) -> list[Issue]:
        if df["date"].is_monotonic_increasing:
            return []
        return [Issue("ERROR", "NOT_SORTED", "Dates are not ascending", symbol)]

    @staticmethod
    def _check_duplicates(df: pd.DataFrame, symbol: str) -> list[Issue]:
        dupes = df[df["date"].duplicated(keep=False)]
        if dupes.empty:
            return []
        days = sorted({pd.Timestamp(d).date() for d in dupes["date"]})
        return [
            Issue(
                "ERROR", "DUPLICATE_DATES",
                f"{len(days)} duplicated session(s), e.g. {days[:5]}", symbol,
            )
        ]

    @staticmethod
    def _check_prices_positive(df: pd.DataFrame, symbol: str) -> list[Issue]:
        out = []
        for col in PRICE_COLUMNS_SPLIT:
            bad = df[(df[col] <= 0) | df[col].isna()]
            if not bad.empty:
                out.append(
                    Issue(
                        "ERROR", "INVALID_PRICE",
                        f"{len(bad)} non-positive or missing value(s) in {col}",
                        symbol, pd.Timestamp(bad["date"].iloc[0]).date(),
                    )
                )
        return out

    @staticmethod
    def _check_ohlc_consistency(df: pd.DataFrame, symbol: str) -> list[Issue]:
        """High must bound the bar; low must floor it."""
        out = []
        hi, lo = df["high_split"], df["low_split"]
        op, cl = df["open_split"], df["close_split"]

        violations = {
            "HIGH_BELOW_LOW": hi < lo,
            "HIGH_BELOW_OPEN": hi < op,
            "HIGH_BELOW_CLOSE": hi < cl,
            "LOW_ABOVE_OPEN": lo > op,
            "LOW_ABOVE_CLOSE": lo > cl,
        }
        for code, mask in violations.items():
            if mask.any():
                first = df.loc[mask, "date"].iloc[0]
                out.append(
                    Issue("ERROR", code, f"{int(mask.sum())} bar(s) violate OHLC bounds",
                          symbol, pd.Timestamp(first).date())
                )
        return out

    @staticmethod
    def _check_volume(df: pd.DataFrame, symbol: str) -> list[Issue]:
        out = []
        if (df["volume"] < 0).any():
            out.append(Issue("ERROR", "NEGATIVE_VOLUME",
                             f"{int((df['volume'] < 0).sum())} bar(s)", symbol))
        zero = df["volume"] == 0
        if zero.any():
            # A zero-volume session on a liquid name means a halt or a data gap.
            # Suspicious, not fatal.
            out.append(
                Issue("WARNING", "ZERO_VOLUME",
                      f"{int(zero.sum())} zero-volume session(s), e.g. "
                      f"{pd.Timestamp(df.loc[zero, 'date'].iloc[0]).date()}", symbol)
            )
        return out

    @staticmethod
    def _check_jumps(df: pd.DataFrame, symbol: str) -> list[Issue]:
        """Large single-session moves, ignoring days with a recorded split.

        A jump on a day with no recorded corporate action is the signature of a
        split the provider missed, which would corrupt both the raw
        reconstruction and every return feature.
        """
        out = []
        if len(df) < 2:
            return out

        ret = df["close_split"].pct_change()
        has_split = df["split_ratio"].ne(1.0)
        suspect = ret.abs() > JUMP_WARN_THRESHOLD
        suspect &= ~has_split & ~has_split.shift(-1, fill_value=False)

        for idx in df.index[suspect.fillna(False)]:
            move = float(ret.loc[idx])
            sev = "ERROR" if abs(move) > JUMP_ERROR_THRESHOLD else "WARNING"
            out.append(
                Issue(
                    sev, "ABNORMAL_JUMP",
                    f"{move:+.1%} move with no recorded corporate action",
                    symbol, pd.Timestamp(df.loc[idx, "date"]).date(),
                )
            )
        return out

    def _check_calendar(self, df: pd.DataFrame, symbol: str) -> list[Issue]:
        start = pd.Timestamp(df["date"].iloc[0]).date()
        end = pd.Timestamp(df["date"].iloc[-1]).date()
        out = []

        missing = self.calendar.missing_sessions(df["date"], start, end)
        if missing:
            # Under the weekday fallback, festival holidays look like gaps.
            sev = "INFO" if self.calendar.is_approximate else "WARNING"
            out.append(
                Issue(sev, "MISSING_SESSIONS",
                      f"{len(missing)} expected session(s) absent, e.g. "
                      f"{missing[:5]}"
                      + (" (approximate calendar; festival holidays expected here)"
                         if self.calendar.is_approximate else ""),
                      symbol)
            )

        if not self.calendar.is_approximate:
            extra = self.calendar.unexpected_sessions(df["date"], start, end)
            if extra:
                out.append(
                    Issue("WARNING", "UNEXPECTED_SESSIONS",
                          f"{len(extra)} bar(s) on non-trading days, e.g. {extra[:5]}",
                          symbol)
                )
        return out

    @staticmethod
    def _check_corporate_actions(df: pd.DataFrame, symbol: str) -> list[Issue]:
        """Reconcile recorded splits against the price move on the ex-date.

        On a 2:1 ex-date the split-adjusted series should be continuous, because
        the provider has already adjusted history. A large move on an ex-date
        means the adjustment and the action disagree.
        """
        out = []
        splits = df.index[df["split_ratio"].ne(1.0)]
        if len(df) < 2:
            return out

        ret = df["close_split"].pct_change()
        for idx in splits:
            ratio = float(df.loc[idx, "split_ratio"])
            move = ret.loc[idx]
            when = pd.Timestamp(df.loc[idx, "date"]).date()

            if pd.isna(move):
                continue
            if abs(move) > JUMP_WARN_THRESHOLD:
                out.append(
                    Issue("ERROR", "SPLIT_RECONCILIATION",
                          f"{ratio}:1 split but split-adjusted series moved "
                          f"{move:+.1%}; adjustment and action disagree",
                          symbol, when)
                )
            else:
                out.append(
                    Issue("INFO", "SPLIT_APPLIED",
                          f"{ratio}:1 split reconciled (move {move:+.2%})",
                          symbol, when)
                )

        divs = df[df["dividend"] > 0]
        if not divs.empty:
            yields = divs["dividend"] / divs["close_split"]
            extreme = divs[yields > 0.25]
            if not extreme.empty:
                out.append(
                    Issue("WARNING", "LARGE_DIVIDEND",
                          f"{len(extreme)} dividend(s) exceeding 25% of price; "
                          "likely a special dividend or a data error",
                          symbol, pd.Timestamp(extreme["date"].iloc[0]).date())
                )
        return out

    @staticmethod
    def _check_staleness(df: pd.DataFrame, symbol: str) -> list[Issue]:
        """Runs of an identical close, which usually means a stale feed."""
        if len(df) < 5:
            return []
        close = df["close_split"].to_numpy(dtype=float)
        same = np.r_[False, np.isclose(close[1:], close[:-1])]

        longest = run = 0
        for flag in same:
            run = run + 1 if flag else 0
            longest = max(longest, run)

        if longest >= 4:
            return [
                Issue("WARNING", "STALE_PRICES",
                      f"{longest + 1} consecutive sessions with an identical close",
                      symbol)
            ]
        return []
