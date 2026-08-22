"""Decision-quality evaluation.

Measures the decisions, not the model. A model with a decent AUC can still
produce a losing decision layer if the thresholds trade too often, and the two
failure modes look nothing alike in the diagnostics.

Every return figure here is **net of costs**. A gross number on a daily strategy
is not informative -- it is the cost that decides the outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import Signal


def evaluate_decisions(
    decisions: pd.DataFrame,
    outcomes: pd.DataFrame,
    round_trip_cost: float,
    signal_col: str = "signal",
    outcome_col: str = "y_true",
) -> dict[str, float]:
    """Score a set of decisions against what the market actually did."""
    merged = (
        decisions.merge(
            outcomes,
            left_on=["symbol", "decision_date"],
            right_on=["symbol", "date"],
            how="inner",
        )
        if "date" in outcomes.columns
        else decisions.join(outcomes)
    )

    if merged.empty:
        return {}

    buys = merged[merged[signal_col] == Signal.BUY.value]
    n_total = len(merged)
    n_buys = len(buys)

    gross = float(buys[outcome_col].mean()) if n_buys else 0.0
    net = gross - round_trip_cost if n_buys else 0.0

    out = {
        "n_decisions": float(n_total),
        "n_buys": float(n_buys),
        "buy_rate": n_buys / n_total if n_total else 0.0,
        "n_holds": float((merged[signal_col] == Signal.HOLD.value).sum()),
        "n_sells": float((merged[signal_col] == Signal.SELL.value).sum()),
        "gross_return_per_trade": gross,
        "round_trip_cost": round_trip_cost,
        "net_return_per_trade": net,
        "total_net_return": net * n_buys if n_buys else 0.0,
        "hit_rate": float((buys[outcome_col] > 0).mean()) if n_buys else 0.0,
        # The share of gross edge consumed by costs. Above 1.0 means the
        # strategy pays more in costs than it earns.
        "cost_drag_ratio": (round_trip_cost / gross if n_buys and gross > 0 else float("inf")),
    }

    if "rejected" in merged.columns:
        out["rejection_rate"] = float(merged["rejected"].mean())

    if n_buys > 1:
        per_trade = buys[outcome_col].to_numpy() - round_trip_cost
        sd = float(np.std(per_trade, ddof=1))
        out["net_sharpe_per_trade"] = (
            float(np.mean(per_trade) / sd * np.sqrt(252)) if sd > 0 else 0.0
        )
    return out


def render_evaluation(metrics: dict[str, float]) -> str:
    if not metrics:
        return "no decisions to evaluate"

    lines = [
        f"decisions {int(metrics['n_decisions']):,} | "
        f"buys {int(metrics['n_buys']):,} ({metrics['buy_rate']:.1%}) | "
        f"holds {int(metrics['n_holds']):,} | sells {int(metrics['n_sells']):,}",
        f"gross/trade {metrics['gross_return_per_trade'] * 10000:+.1f} bps | "
        f"cost {metrics['round_trip_cost'] * 10000:.1f} bps | "
        f"net/trade {metrics['net_return_per_trade'] * 10000:+.1f} bps",
        f"hit rate {metrics['hit_rate']:.1%}",
    ]
    drag = metrics.get("cost_drag_ratio", float("inf"))
    if not np.isfinite(drag) or drag > 1.0:
        lines.append(
            "  costs exceed gross edge: this decision set loses money by "
            "construction, independent of model quality."
        )
    return "\n".join(lines)
