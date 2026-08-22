"""Threshold derivation, and the cost feasibility question that precedes it.

## The finding that comes first

Before optimising anything, check whether the strategy can clear its own costs.
With the configured cost model::

    commission 3 bps + spread 5 bps + slippage 5 bps = 13 bps per side
    round trip                                        = 26 bps = 0.0026

A one-day holding period means **every signal is a full round trip**. There is
no amortisation. So each trade must produce more than 26 bps of expected return
before it breaks even.

What can a daily signal produce? Going long the top decile of a ranked
cross-section, expected edge is approximately::

    edge  ~  IC  x  sigma_daily  x  E[z | top decile]
          ~  IC  x  0.015        x  1.755

Which gives:

===========  ==================  ===============
IC           expected edge       needed
===========  ==================  ===============
0.02          5.3 bps            26 bps
0.03          7.9 bps            26 bps
0.05         13.2 bps            26 bps
0.10         26.3 bps            26 bps
===========  ==================  ===============

**Breakeven IC is roughly 0.099.** A daily information coefficient near 0.10 is
not a realistic target on liquid large-caps; 0.02-0.05 is what a good signal
produces. The measured IC in Phase 4 was 0.027 with a fold standard deviation of
0.055 — indistinguishable from zero.

The conclusion is structural, not a tuning problem: **a one-day holding period
at these costs cannot be profitable at any threshold.** No threshold search will
fix it, because the search is choosing among trades that all lose money on
costs.

``cost_feasibility()`` computes this for the configured costs and reports it
before any optimisation runs. The honest responses are a longer holding period
(costs amortise over more days of edge), lower costs, or accepting that the
system is a decision-support research platform rather than a profitable
strategy. This project takes the third position explicitly.

## The roadmap's min_expected_return was below the cost floor

It specified ``min_expected_return = 0.0005`` — 5 bps, against a 26 bps round
trip. Every trade passing that gate loses 21 bps on costs alone before the
market moves. The gate is now *derived* from the cost model rather than set by
hand, and ``config.yaml`` leaves it null until it is.

## Thresholds must be derived out-of-fold

Searching thresholds on the same predictions used to report performance is
selection bias — the same error as in-sample calibration, one level up. With
enough candidate thresholds the best one is mostly luck, and the reported
performance is that luck. ``optimise_thresholds_out_of_fold`` fits on earlier
blocks and evaluates on later ones.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# E[z | z above the 90th percentile] for a standard normal.
TOP_DECILE_Z = 1.755


@dataclass(frozen=True)
class CostModel:
    """Transaction costs in basis points, per side."""

    commission_bps: float = 3.0
    spread_bps: float = 5.0
    slippage_bps: float = 5.0

    @property
    def per_side(self) -> float:
        return (self.commission_bps + self.spread_bps + self.slippage_bps) / 10000.0

    @property
    def round_trip(self) -> float:
        return 2.0 * self.per_side

    @classmethod
    def from_config(cls, cfg) -> CostModel:
        return cls(
            commission_bps=cfg.get("backtest.commission_bps"),
            spread_bps=cfg.get("backtest.spread_bps"),
            slippage_bps=cfg.get("backtest.slippage_bps"),
        )


@dataclass(frozen=True)
class FeasibilityReport:
    """Whether the strategy can clear its own costs, before any tuning."""

    round_trip_cost: float
    holding_days: int
    daily_volatility: float
    breakeven_ic: float
    observed_ic: float | None
    expected_edge: float | None
    feasible: bool

    def render(self) -> str:
        lines = [
            "cost feasibility",
            f"  round-trip cost      {self.round_trip_cost * 10000:6.1f} bps",
            f"  holding period       {self.holding_days} session(s)",
            f"  daily volatility     {self.daily_volatility * 100:6.2f}%",
            f"  breakeven IC         {self.breakeven_ic:6.3f}",
        ]
        if self.observed_ic is not None:
            lines += [
                f"  observed IC          {self.observed_ic:6.3f}",
                f"  expected edge        {self.expected_edge * 10000:6.1f} bps",
            ]
        lines.append(
            "  VERDICT: costs can be cleared"
            if self.feasible
            else "  VERDICT: NOT FEASIBLE — expected edge is below the round-trip "
            "cost. No threshold fixes this; the holding period or the cost "
            "model has to change."
        )
        return "\n".join(lines)


def cost_feasibility(
    costs: CostModel,
    holding_days: int = 1,
    daily_volatility: float = 0.015,
    observed_ic: float | None = None,
    selection_z: float = TOP_DECILE_Z,
) -> FeasibilityReport:
    """Compare the achievable edge against the cost floor.

    ``holding_days`` amortises the round trip: holding for 5 sessions spreads
    one round trip over 5 days of edge, which is the main lever available.
    """
    if holding_days < 1:
        raise ValueError("holding_days must be >= 1")

    rt = costs.round_trip
    # Edge accumulates over the holding period; cost is paid once.
    edge_per_ic = daily_volatility * selection_z * np.sqrt(holding_days)
    breakeven_ic = rt / edge_per_ic if edge_per_ic > 0 else float("inf")

    expected_edge = observed_ic * edge_per_ic if observed_ic is not None else None
    feasible = expected_edge is not None and expected_edge > rt

    return FeasibilityReport(
        round_trip_cost=rt,
        holding_days=holding_days,
        daily_volatility=daily_volatility,
        breakeven_ic=float(breakeven_ic),
        observed_ic=observed_ic,
        expected_edge=expected_edge,
        feasible=bool(feasible),
    )


def minimum_expected_return(costs: CostModel, margin: float = 1.5) -> float:
    """The cost floor a forecast must clear before a trade is worth making.

    ``margin`` above 1.0 requires the expected return to exceed costs by a
    buffer rather than merely match them, because the forecast is itself
    uncertain and a trade that breaks even in expectation loses money about half
    the time after costs.

    Derived, never tuned. Tuning this against returns would optimise the cost
    model itself, which is a fact about the broker, not a free parameter.
    """
    if margin < 1.0:
        raise ValueError("margin must be >= 1.0; below that trades lose in expectation")
    return costs.round_trip * margin


# =====================================================================
# Threshold search
# =====================================================================


@dataclass(frozen=True)
class Thresholds:
    """The decision boundaries. Derived on development data, then locked."""

    buy: float
    sell: float
    min_expected_return: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.sell <= self.buy <= 1.0:
            raise ValueError(
                f"Need 0 <= sell ({self.sell}) <= buy ({self.buy}) <= 1. "
                "Overlapping thresholds make the HOLD band undefined."
            )
        if self.min_expected_return < 0:
            raise ValueError("min_expected_return must be >= 0")

    def as_dict(self) -> dict:
        return asdict(self)


def _net_return(y_true: np.ndarray, traded: np.ndarray, cost: float) -> float:
    """Mean per-observation return after paying the round trip on each trade."""
    if traded.sum() == 0:
        return 0.0
    return float((y_true[traded].sum() - cost * traded.sum()) / len(y_true))


def evaluate_threshold(
    probabilities: np.ndarray,
    returns: np.ndarray,
    buy: float,
    min_expected: float,
    expected_returns: np.ndarray | None,
    cost: float,
) -> dict[str, float]:
    """Score one candidate threshold on realised, cost-adjusted returns.

    The objective is net return, not accuracy. Optimising accuracy produces a
    threshold that trades constantly on marginal edges, because accuracy has no
    concept of the 26 bps each trade costs.
    """
    traded = probabilities >= buy
    if expected_returns is not None:
        traded &= expected_returns >= min_expected

    n_trades = int(traded.sum())
    gross = float(returns[traded].mean()) if n_trades else 0.0

    return {
        "buy": buy,
        "min_expected_return": min_expected,
        "n_trades": n_trades,
        "trade_rate": n_trades / len(returns) if len(returns) else 0.0,
        "gross_return_per_trade": gross,
        "net_return_per_trade": gross - cost if n_trades else 0.0,
        "net_return_total": _net_return(returns, traded, cost),
        "hit_rate": float((returns[traded] > 0).mean()) if n_trades else 0.0,
    }


def optimise_thresholds_out_of_fold(
    signals: pd.DataFrame,
    costs: CostModel,
    prob_col: str = "calibrated",
    return_col: str = "y_true",
    expected_return_col: str | None = None,
    n_blocks: int = 4,
    min_fit_rows: int = 500,
    buy_grid: np.ndarray | None = None,
    margin: float = 1.5,
) -> tuple[Thresholds, pd.DataFrame]:
    """Choose thresholds on earlier blocks, score them on later ones.

    Returns the thresholds selected on *all* development data — the artifact
    that ships — plus a frame showing how each block's selection performed
    out-of-sample. The second is the honest estimate; the first is what runs.

    A large gap between in-block and out-of-block performance means the
    threshold is fitting noise, which on a weak signal is the normal outcome.
    """
    data = signals.dropna(subset=[prob_col, return_col]).sort_values("date")
    data = data.reset_index(drop=True)
    if len(data) < min_fit_rows * 2:
        raise ValueError(
            f"Need at least {min_fit_rows * 2} rows to derive thresholds "
            f"out-of-fold; got {len(data)}."
        )

    if buy_grid is None:
        buy_grid = np.round(np.arange(0.45, 0.81, 0.01), 4)

    cost = costs.round_trip
    floor = minimum_expected_return(costs, margin)

    dates = pd.Index(sorted(data["date"].unique()))
    blocks = np.array_split(np.arange(len(dates)), n_blocks)
    rows = []

    for k, block in enumerate(blocks):
        if k == 0 or len(block) == 0:
            continue
        prior = data[data["date"] < dates[block[0]]]
        if len(prior) < min_fit_rows:
            continue

        chosen = _search(
            prior, buy_grid, floor, prob_col, return_col, expected_return_col, cost
        )

        holdout = data[data["date"].isin(set(dates[block]))]
        scored = evaluate_threshold(
            holdout[prob_col].to_numpy(),
            holdout[return_col].to_numpy(),
            chosen["buy"],
            floor,
            holdout[expected_return_col].to_numpy() if expected_return_col else None,
            cost,
        )
        rows.append(
            {
                "block": k,
                "selected_buy": chosen["buy"],
                "in_block_net": chosen["net_return_total"],
                "out_of_block_net": scored["net_return_total"],
                "out_of_block_trades": scored["n_trades"],
            }
        )

    stability = pd.DataFrame(rows)
    if not stability.empty:
        spread = stability["selected_buy"].std(ddof=1) if len(stability) > 1 else 0.0
        log.info(
            "Threshold stability: selected buy varies %.3f across blocks",
            spread if pd.notna(spread) else 0.0,
        )
        if pd.notna(spread) and spread > 0.05:
            log.warning(
                "Selected buy threshold swings by %.3f between blocks. The "
                "threshold is fitting noise, not finding a boundary.",
                spread,
            )

    final = _search(data, buy_grid, floor, prob_col, return_col, expected_return_col, cost)

    thresholds = Thresholds(
        buy=float(final["buy"]),
        # V1 exits whenever the buy condition lapses, so the sell boundary sits
        # just below it. A separate sell threshold would create a dead band in
        # which a deteriorating position is neither held with conviction nor
        # exited, and there is no evidence here to place that band.
        sell=float(max(0.0, final["buy"] - 0.02)),
        min_expected_return=floor,
    )
    log.info(
        "Derived thresholds: buy=%.3f sell=%.3f min_expected_return=%.5f "
        "(cost floor %.5f x margin %.1f)",
        thresholds.buy,
        thresholds.sell,
        thresholds.min_expected_return,
        costs.round_trip,
        margin,
    )
    return thresholds, stability


def _search(
    data: pd.DataFrame,
    buy_grid: np.ndarray,
    floor: float,
    prob_col: str,
    return_col: str,
    expected_return_col: str | None,
    cost: float,
) -> dict[str, float]:
    """Grid search for the buy threshold maximising net total return."""
    probs = data[prob_col].to_numpy()
    rets = data[return_col].to_numpy()
    exp = data[expected_return_col].to_numpy() if expected_return_col else None

    scored = [evaluate_threshold(probs, rets, float(b), floor, exp, cost) for b in buy_grid]
    best = max(scored, key=lambda r: r["net_return_total"])

    if best["n_trades"] == 0:
        # Every candidate produced zero trades: the signal never clears the
        # cost floor. Report the most permissive threshold rather than an
        # arbitrary one, and let the feasibility report explain why.
        log.warning(
            "No threshold produced a profitable trade set; the signal does not "
            "clear the cost floor at any boundary."
        )
        return scored[0]
    return best
