"""Labels, and the timing contract that defines them.

## The problem with the roadmap's label

The roadmap specifies two things that cannot both be true::

    target_return_1d[t] = adj_close[t+1] / adj_close[t] - 1
    "Signals at t execute at the next trading session."

If the decision is made after the close of *t* and the order fills during
session *t+1*, then the move from ``close[t]`` to ``close[t+1]`` is already
partly or wholly in the past by the time the position exists. Training on that
label teaches the model to forecast a return the strategy cannot capture, and
the backtest then quietly fails to reproduce the model's apparent skill.

This is the same class of error as a leaky feature — the label reaches back
across the execution boundary — and it is easy to miss because nothing about it
looks like look-ahead. Every individual number is real.

## The contract this project uses

::

    session t          session t+1              session t+2
    ---------          -----------              -----------
    close: features         open: ENTRY              open: EXIT
           computed              (fill here)              (fill here)
           decision made

    tradeable_return_1d[t] = adj_open[t+2] / adj_open[t+1] - 1

Features are computed from data through the close of *t*, which is legitimately
available — the decision is made after the bell. The entry fills at the open of
*t+1*, the earliest price no part of the decision could have seen. A one-day
holding period exits at the open of *t+2*.

The cost is one extra day of label lag: ``tradeable_return_1d[t]`` needs data
through *t+2*, so the two most recent rows can never be labelled. The benefit is
that the backtester can actually earn what the model is trained to predict.

## Both labels are produced

``tradeable_return_1d``
    Open-to-open across the executable window. **The training target.**

``research_return_1d``
    Close-to-close, the roadmap's original definition. Retained as a diagnostic
    only, because it is the conventional academic label and comparing the two
    quantifies exactly how much apparent skill lives in the untradeable gap.

Any model trained on ``research_return_1d`` and evaluated by the backtester will
disagree with itself. ``pipeline.py`` defaults to the tradeable label and the
choice is recorded in the feature version.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADEABLE_LABEL = "tradeable_return_1d"
RESEARCH_LABEL = "research_return_1d"
DIRECTION_LABEL = "tradeable_direction_1d"

LABEL_COLUMNS = (TRADEABLE_LABEL, RESEARCH_LABEL, DIRECTION_LABEL)


def add_adjusted_open(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ``adj_open`` using the same factor that produced ``adj_close``.

    The corporate-action layer only adjusts the close. Entry and exit happen at
    the open, so the open needs the identical treatment — using a mix of
    adjusted close and unadjusted open would inject a spurious overnight return
    at every dividend.
    """
    out = df.copy()
    factor = np.where(
        out["close_split"].to_numpy() > 0,
        out["adj_close"].to_numpy() / out["close_split"].to_numpy(),
        1.0,
    )
    out["adj_factor"] = factor
    out["adj_open"] = out["open_split"].to_numpy() * factor
    return out


def tradeable_forward_return(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Return earned by entering at open[t+1] and exiting at open[t+1+horizon].

    Uses ``shift(-n)``, which is the one place in this codebase where looking
    forward is correct: a label *is* future information. The guarantee is that
    no *feature* does this, which ``tests/test_leakage.py`` enforces globally.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    entry = df["adj_open"].shift(-1)
    exit_ = df["adj_open"].shift(-(1 + horizon))
    return exit_ / entry - 1.0


def research_forward_return(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Close-to-close forward return. Diagnostic only — see the module docstring."""
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    return df["adj_close"].shift(-horizon) / df["adj_close"] - 1.0


def add_labels(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Attach both labels plus the binary direction target.

    Rows too recent to be labelled keep NaN rather than being dropped. That is
    deliberate: the most recent row is exactly the one the live daily job needs
    a prediction for, and dropping it here would make the training and inference
    paths diverge. ``pipeline.drop_unlabelled`` removes them for training only.
    """
    out = add_adjusted_open(df) if "adj_open" not in df.columns else df.copy()

    out[TRADEABLE_LABEL] = tradeable_forward_return(out, horizon)
    out[RESEARCH_LABEL] = research_forward_return(out, horizon)

    direction = np.where(
        out[TRADEABLE_LABEL].isna(),
        np.nan,
        (out[TRADEABLE_LABEL] > 0).astype(float),
    )
    out[DIRECTION_LABEL] = direction

    return out


def label_lag_days(horizon: int = 1) -> int:
    """Sessions of future data a label needs. Rows this recent stay unlabelled."""
    return horizon + 1
