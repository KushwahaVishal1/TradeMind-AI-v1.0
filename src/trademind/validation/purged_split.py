"""Purged, embargoed walk-forward splitting.

## Why this cannot be sklearn's ``TimeSeriesSplit``

Two reasons, either one fatal.

**It splits on row position.** This is a *panel*: 15 symbols share every date, so
consecutive rows are different stocks on the same day, not consecutive days. A
positional split cuts through the middle of a trading session, putting RELIANCE
on 2023-06-15 in train and TCS on 2023-06-15 in validation. Same day, same market
move, both sides of the boundary. Every split here operates on the sorted unique
*date* index and assigns whole sessions.

**It has no gap.** ``TimeSeriesSplit`` places validation immediately after
training, so the last training row's label overlaps the first validation rows.
See ``embargo.py``.

## What this yields

Expanding-window walk-forward by default: each fold trains on everything from
the start of development data up to the purge boundary, then validates on the
next block. Rolling windows are available via ``train_size`` for testing
whether the model decays on old data.

::

    fold 0  |TTTTTTTT|P|VVVV|E|........................|
    fold 1  |TTTTTTTTTTTTTTT|P|VVVV|E|.................|
    fold 2  |TTTTTTTTTTTTTTTTTTTTTT|P|VVVV|E|..........|

    T train   P purge   V validation   E embargo
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .embargo import GapConfig, ValidationConfigError, embargo_mask, purge_mask

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    """One train/validation split, described in both dates and row indices."""

    index: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    n_purged: int
    n_embargoed: int

    def __str__(self) -> str:
        return (
            f"fold {self.index}: train {self.train_start.date()}..{self.train_end.date()} "
            f"({len(self.train_idx):,} rows) | val {self.val_start.date()}.."
            f"{self.val_end.date()} ({len(self.val_idx):,} rows) | "
            f"purged {self.n_purged}, embargoed {self.n_embargoed}"
        )

    def describe(self) -> dict:
        return {
            "fold": self.index,
            "train_start": self.train_start.date(),
            "train_end": self.train_end.date(),
            "val_start": self.val_start.date(),
            "val_end": self.val_end.date(),
            "n_train": len(self.train_idx),
            "n_val": len(self.val_idx),
            "n_purged": self.n_purged,
            "n_embargoed": self.n_embargoed,
        }


class PurgedWalkForwardSplit:
    """Walk-forward splitter with purge and embargo, keyed on dates.

    Parameters
    ----------
    n_splits
        Number of folds.
    test_sessions
        Length of each validation block, in trading sessions.
    gaps
        Purge/embargo configuration. Validated against the label horizon.
    train_sessions
        ``None`` for an expanding window (default), or a fixed number of
        sessions for a rolling window.
    min_train_sessions
        Refuse to emit a fold with less training history than this. Prevents
        the first fold from training on a sample too thin to mean anything.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_sessions: int = 126,
        gaps: GapConfig | None = None,
        train_sessions: int | None = None,
        min_train_sessions: int = 252,
    ) -> None:
        if n_splits < 1:
            raise ValidationConfigError("n_splits must be >= 1")
        if test_sessions < 1:
            raise ValidationConfigError("test_sessions must be >= 1")

        self.n_splits = n_splits
        self.test_sessions = test_sessions
        self.gaps = gaps or GapConfig()
        self.train_sessions = train_sessions
        self.min_train_sessions = min_train_sessions

    # -- core ------------------------------------------------------------

    def split(self, df: pd.DataFrame, date_col: str = "date") -> Iterator[Fold]:
        """Yield folds for a date-sorted panel."""
        if date_col not in df.columns:
            raise ValidationConfigError(f"No '{date_col}' column to split on")

        dates = pd.to_datetime(df[date_col])
        sessions = pd.Index(sorted(dates.unique()))
        n = len(sessions)

        required = self.min_train_sessions + self.gaps.purge + self.test_sessions
        if n < required:
            raise ValidationConfigError(
                f"Need at least {required} sessions for one fold "
                f"(min_train {self.min_train_sessions} + purge {self.gaps.purge} "
                f"+ test {self.test_sessions}); the data has {n}."
            )

        # Lay validation blocks end-to-end, finishing at the last session, so
        # the most recent data is always validated on.
        val_starts = []
        for k in range(self.n_splits):
            end_pos = n - k * self.test_sessions
            start_pos = end_pos - self.test_sessions
            train_end_pos = start_pos - self.gaps.purge
            if train_end_pos < self.min_train_sessions:
                break
            val_starts.append(start_pos)
        val_starts.reverse()

        if not val_starts:
            raise ValidationConfigError(
                f"No fold satisfies min_train_sessions={self.min_train_sessions} "
                f"with {n} sessions and {self.n_splits} splits. Reduce n_splits, "
                "test_sessions, or min_train_sessions."
            )
        if len(val_starts) < self.n_splits:
            log.warning(
                "Requested %d folds but only %d fit the available history",
                self.n_splits,
                len(val_starts),
            )

        arr = dates.to_numpy()
        positions = np.arange(len(df))

        for i, start_pos in enumerate(val_starts):
            val_start = sessions[start_pos]
            val_end = sessions[min(n - 1, start_pos + self.test_sessions - 1)]

            in_val = (arr >= np.datetime64(val_start)) & (arr <= np.datetime64(val_end))

            # Everything strictly before validation is a training candidate.
            candidate = arr < np.datetime64(val_start)

            if self.train_sessions is not None:
                first_pos = max(0, start_pos - self.gaps.purge - self.train_sessions)
                candidate &= arr >= np.datetime64(sessions[first_pos])

            purged = purge_mask(dates, val_start, self.gaps.purge)
            embargoed = embargo_mask(dates, val_end, self.gaps.embargo, sessions)

            in_train = candidate & ~purged
            n_purged = int((candidate & purged).sum())

            yield Fold(
                index=i,
                train_idx=positions[in_train],
                val_idx=positions[in_val],
                train_start=pd.Timestamp(arr[in_train].min()),
                train_end=pd.Timestamp(arr[in_train].max()),
                val_start=pd.Timestamp(val_start),
                val_end=pd.Timestamp(val_end),
                n_purged=n_purged,
                n_embargoed=int(embargoed.sum()),
            )

    def get_n_splits(self, df: pd.DataFrame | None = None) -> int:
        return self.n_splits if df is None else len(list(self.split(df)))

    def summary(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame([f.describe() for f in self.split(df)])


def assert_fold_is_clean(
    fold: Fold, df: pd.DataFrame, gaps: GapConfig, date_col: str = "date"
) -> None:
    """Verify a fold's separation. Raises on any violation.

    Called by the validation runner on every fold rather than trusted from the
    splitter's own logic, because a splitter bug is silent by nature: the
    metrics simply come out better than they should.
    """
    dates = pd.to_datetime(df[date_col])
    train_dates = dates.iloc[fold.train_idx]
    val_dates = dates.iloc[fold.val_idx]

    if train_dates.empty or val_dates.empty:
        raise ValidationConfigError(f"Fold {fold.index} has an empty side")

    overlap = set(train_dates.unique()) & set(val_dates.unique())
    if overlap:
        raise ValidationConfigError(
            f"Fold {fold.index}: {len(overlap)} date(s) appear in both train and "
            f"validation, e.g. {sorted(overlap)[:3]}"
        )

    if train_dates.max() >= val_dates.min():
        raise ValidationConfigError(
            f"Fold {fold.index}: training extends to {train_dates.max().date()}, "
            f"at or beyond validation start {val_dates.min().date()}"
        )

    sessions = pd.Index(sorted(dates.unique()))
    gap = (
        sessions.searchsorted(val_dates.min(), side="left")
        - sessions.searchsorted(train_dates.max(), side="left")
        - 1
    )
    if gap < gaps.purge:
        raise ValidationConfigError(
            f"Fold {fold.index}: only {gap} session(s) between train end and "
            f"validation start; purge requires {gaps.purge}. Labels from the "
            "last training rows overlap the validation window."
        )
