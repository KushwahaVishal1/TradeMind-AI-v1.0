"""Feature pipeline: raw bars in, model-ready panel out.

Order of operations, and why:

1. ``add_adjusted_open`` — the adjustment factor everything else needs.
2. Per-symbol feature blocks — technical, returns, volatility, volume, regime.
3. Labels — the only place ``shift(-n)`` is permitted.
4. Warm-up trim — drop rows whose longest-window feature never filled.
5. Cross-sectional ranks — needs the full panel, so it comes after concatenation.

Warm-up trimming matters more than it looks. A 200-day moving average is NaN for
199 rows; if those rows survive, any model that tolerates NaN silently trains on
a different feature set for early dates than for late ones, and the early period
looks anomalous for reasons that have nothing to do with the market.
"""

from __future__ import annotations

import logging

import pandas as pd

from . import regime, returns, technical, volatility, volume
from .labels import (
    DIRECTION_LABEL,
    LABEL_COLUMNS,
    TRADEABLE_LABEL,
    add_adjusted_open,
    add_labels,
)

log = logging.getLogger(__name__)

# Longest lookback any feature uses. Rows before this are structurally
# incomplete and are dropped rather than imputed.
WARMUP_SESSIONS = 252

# Columns carried through for joining and backtesting, never fed to a model.
PASSTHROUGH = (
    "date",
    "symbol",
    "open_raw",
    "high_raw",
    "low_raw",
    "close_raw",
    "adj_close",
    "adj_open",
    "close_split",
    "volume",
    "split_ratio",
    "dividend",
)


def build_symbol_features(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Full feature set for a single symbol's bar history."""
    if df.empty:
        return pd.DataFrame()

    if not df["date"].is_monotonic_increasing:
        raise ValueError("Feature construction requires date-sorted bars")

    base = add_adjusted_open(df).reset_index(drop=True)

    blocks = [
        technical.build(base),
        returns.build(base),
        volatility.build(base),
        volume.build(base),
    ]
    features = pd.concat(blocks, axis=1)

    # Regime needs a volatility series that already exists.
    features = pd.concat(
        [features, regime.build(base, volatility=features["volatility_20"])], axis=1
    )

    keep = [c for c in PASSTHROUGH if c in base.columns]
    out = pd.concat([base[keep], features], axis=1)

    return add_labels(out, horizon=horizon)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model input columns: everything that is neither passthrough nor label."""
    excluded = set(PASSTHROUGH) | set(LABEL_COLUMNS) | {"adj_factor"}
    return [c for c in df.columns if c not in excluded]


def trim_warmup(df: pd.DataFrame, sessions: int = WARMUP_SESSIONS) -> pd.DataFrame:
    """Drop the leading rows where long-window features have not filled."""
    if len(df) <= sessions:
        return df.iloc[0:0].copy()
    return df.iloc[sessions:].reset_index(drop=True)


def drop_unlabelled(df: pd.DataFrame, label: str = TRADEABLE_LABEL) -> pd.DataFrame:
    """Remove rows with no label. **Training only.**

    The most recent rows are unlabelled because their outcome has not happened
    yet — and those are exactly the rows the live daily job predicts on. Calling
    this on the inference path would drop today.
    """
    return df[df[label].notna()].reset_index(drop=True)


def build_panel(
    frames: dict[str, pd.DataFrame],
    horizon: int = 1,
    cross_sectional: bool = True,
) -> pd.DataFrame:
    """Build the multi-symbol panel that the modelling phases consume."""
    built = []
    for symbol, bars in frames.items():
        try:
            feats = trim_warmup(build_symbol_features(bars, horizon))
            if feats.empty:
                log.warning("%s: insufficient history after warm-up trim", symbol)
                continue
            built.append(feats)
        except Exception as exc:
            log.error("%s: feature build failed: %s", symbol, exc)

    if not built:
        return pd.DataFrame()

    panel = pd.concat(built, ignore_index=True)
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)

    if cross_sectional and panel["symbol"].nunique() > 1:
        panel = regime.add_cross_sectional(
            panel, ["return_5d", "return_20d", "volatility_20", "relative_volume_20"]
        )

    n_feat = len(feature_columns(panel))
    log.info(
        "Panel: %d rows | %d symbols | %d features | %s..%s",
        len(panel),
        panel["symbol"].nunique(),
        n_feat,
        panel["date"].min().date(),
        panel["date"].max().date(),
    )
    return panel


def coverage_report(panel: pd.DataFrame) -> pd.DataFrame:
    """Missing-rate per feature.

    A feature that is 40% NaN after warm-up trimming is broken, not sparse.
    Phase 8's feature-drift monitor tracks the same statistic against this
    baseline.
    """
    cols = feature_columns(panel)
    return (
        pd.DataFrame(
            {
                "feature": cols,
                "missing_rate": [panel[c].isna().mean() for c in cols],
                "n_unique": [panel[c].nunique() for c in cols],
            }
        )
        .sort_values("missing_rate", ascending=False)
        .reset_index(drop=True)
    )


__all__ = [
    "DIRECTION_LABEL",
    "TRADEABLE_LABEL",
    "WARMUP_SESSIONS",
    "build_panel",
    "build_symbol_features",
    "coverage_report",
    "drop_unlabelled",
    "feature_columns",
    "trim_warmup",
]
