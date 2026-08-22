"""Purge and embargo: the two gaps that separate training from validation.

They solve different problems and are frequently confused.

## Purge — removes training rows whose *label* reaches into validation

A feature row at date *t* is labelled with an outcome that has not happened yet.
Under this project's timing contract the label for *t* spans ``open[t+1]`` to
``open[t+2]``, so the row at *t* encodes information about *t+2*.

If validation starts at *v* and a training row sits at ``v - 1``, that row's
label is partly determined by prices inside the validation window. The model
learns from an outcome it is about to be scored on::

    train ... t-2  t-1   t  | v   v+1  v+2 ...  validation
                       └────┴──┴───┘
                       label of t leaks into v and v+1

Required purge is therefore ``horizon + 1``, not ``horizon``. The extra day is
the execution offset — the gap between the decision at *t* and the entry at
*t+1*. A purge sized only to the horizon leaves exactly one leaking row at every
fold boundary, which is enough to inflate validation scores and impossible to
spot from the metrics alone.

## Embargo — removes training rows immediately *after* validation

Purge handles the label reaching forward. Embargo handles serial correlation
reaching backward.

Financial features are strongly autocorrelated: a 20-day volatility on the day
after validation ends shares 19 of its 20 observations with the validation
window. It is not a leaked label — it is a near-duplicate of data the model was
scored on, which makes any later fold that trains on it optimistic.

Embargo matters in walk-forward because each fold's validation window becomes
the next fold's training data. It should be at least as long as the longest
rolling window whose contamination you care about; a few days is conventional,
and the full 252-day warm-up would be so aggressive it would leave nothing to
train on.

::

    |<--- train --->|<-purge->|<-- validation -->|<-embargo->|<- future train ->|
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class ValidationConfigError(ValueError):
    """Purge/embargo settings that would permit leakage."""


def required_purge(horizon: int, execution_offset: int = 1) -> int:
    """Minimum purge for a given label horizon.

    ``execution_offset`` is the gap between the decision and the entry fill —
    1 under this project's contract (decide at close of t, enter at open of
    t+1). Pass 0 only for a close-to-close research label, which is not what
    this system trades on.
    """
    if horizon < 1:
        raise ValidationConfigError(f"horizon must be >= 1, got {horizon}")
    if execution_offset < 0:
        raise ValidationConfigError("execution_offset must be >= 0")
    return horizon + execution_offset


@dataclass(frozen=True)
class GapConfig:
    """Purge and embargo, validated against the label definition."""

    horizon: int = 1
    purge: int = 2
    embargo: int = 3
    execution_offset: int = 1

    def __post_init__(self) -> None:
        needed = required_purge(self.horizon, self.execution_offset)
        if self.purge < needed:
            raise ValidationConfigError(
                f"purge={self.purge} is too small for horizon={self.horizon} "
                f"with execution_offset={self.execution_offset}; needs >= {needed}. "
                "A purge sized only to the horizon leaves one leaking row at "
                "every fold boundary."
            )
        if self.embargo < 0:
            raise ValidationConfigError("embargo must be >= 0")

    @classmethod
    def from_config(cls, cfg) -> GapConfig:
        return cls(
            horizon=cfg.get("features.target_horizon_days"),
            purge=cfg.get("features.purge_days"),
            embargo=cfg.get("features.embargo_days"),
        )


def purge_mask(dates: pd.Series, val_start: pd.Timestamp, purge: int) -> np.ndarray:
    """True for training dates that must be dropped ahead of ``val_start``.

    Measured in *sessions*, not calendar days, so a weekend does not silently
    absorb the gap. ``dates`` must be the sorted unique session index.
    """
    if purge <= 0:
        return np.zeros(len(dates), dtype=bool)

    sessions = pd.Index(sorted(pd.Series(dates).unique()))
    pos = sessions.searchsorted(val_start, side="left")
    cutoff_pos = max(0, pos - purge)
    cutoff = sessions[cutoff_pos] if cutoff_pos < len(sessions) else sessions[-1]

    return (pd.Series(dates).to_numpy() >= np.datetime64(cutoff)) & (
        pd.Series(dates).to_numpy() < np.datetime64(val_start)
    )


def embargo_mask(
    dates: pd.Series, val_end: pd.Timestamp, embargo: int, sessions: pd.Index
) -> np.ndarray:
    """True for dates inside the embargo window following ``val_end``."""
    if embargo <= 0:
        return np.zeros(len(dates), dtype=bool)

    pos = sessions.searchsorted(val_end, side="right")
    end_pos = min(len(sessions) - 1, pos + embargo - 1)
    if pos > end_pos:
        return np.zeros(len(dates), dtype=bool)

    embargo_end = sessions[end_pos]
    arr = pd.Series(dates).to_numpy()
    return (arr > np.datetime64(val_end)) & (arr <= np.datetime64(embargo_end))
