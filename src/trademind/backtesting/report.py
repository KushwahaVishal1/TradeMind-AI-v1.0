"""Performance metrics and reporting.

Every figure is net of costs. Two things are reported that most backtest
summaries omit, and both are the numbers that decide whether a result is real:

``cost_drag``
    Total costs as a fraction of initial capital, and the gross return the
    strategy would have needed to break even. On a daily strategy this often
    exceeds the entire return, which is the finding, not a footnote.

``turnover``
    Annualised. A Sharpe of 1.2 at 40x annual turnover and a Sharpe of 1.2 at
    2x turnover are not the same result — the first is far more sensitive to
    every cost assumption, and its confidence interval under a worse slippage
    estimate is enormous.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """Deepest peak-to-trough decline, and its length in sessions."""
    if len(equity) < 2:
        return 0.0, 0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    int(dd.idxmin()) if len(dd) else 0
    depth = float(dd.min())

    in_dd = dd < 0
    longest = current = 0
    for flag in in_dd:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return depth, longest


def sharpe_ratio(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe, zero risk-free rate.

    Unreliable on short samples: over one year of daily data the standard error
    is roughly 1.0, so a Sharpe of 1.4 is not distinguishable from 0.4. Read it
    alongside the sample length, never alone.
    """
    if len(returns) < 2:
        return 0.0
    sd = float(returns.std(ddof=1))
    return float(returns.mean() / sd * np.sqrt(periods)) if sd > 0 else 0.0


def sortino_ratio(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Like Sharpe, but penalising only downside deviation."""
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) < 2:
        return 0.0
    dd = float(downside.std(ddof=1))
    return float(returns.mean() / dd * np.sqrt(periods)) if dd > 0 else 0.0


def sharpe_stderr(n_periods: int, periods: int = TRADING_DAYS) -> float:
    """Approximate standard error of an annualised Sharpe estimate."""
    return float(np.sqrt(periods / n_periods)) if n_periods > 0 else float("nan")


def performance_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    initial_capital: float | None = None,
) -> dict[str, float]:
    """Full performance summary from an equity curve."""
    if equity_curve.empty:
        return {}

    equity = equity_curve["equity"].reset_index(drop=True)
    returns = _returns(equity)
    n = len(equity)

    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    total_return = end / start - 1.0 if start else 0.0
    years = n / TRADING_DAYS

    depth, dd_len = max_drawdown(equity)
    sharpe = sharpe_ratio(returns)

    out = {
        "n_sessions": float(n),
        "years": years,
        "initial_equity": start,
        "final_equity": end,
        "total_return": total_return,
        "cagr": (end / start) ** (1 / years) - 1.0 if years > 0 and start > 0 else 0.0,
        "annual_volatility": float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
        if len(returns) > 1
        else 0.0,
        "sharpe": sharpe,
        "sharpe_stderr": sharpe_stderr(n),
        "sortino": sortino_ratio(returns),
        "max_drawdown": depth,
        "max_drawdown_sessions": float(dd_len),
        "win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
        "mean_exposure": float(equity_curve["exposure"].mean())
        if "exposure" in equity_curve
        else 0.0,
    }

    if "total_costs" in equity_curve.columns:
        total_costs = float(equity_curve["total_costs"].iloc[-1])
        capital = initial_capital or start
        out["total_costs"] = total_costs
        out["cost_drag"] = total_costs / capital if capital else 0.0
        # What the strategy needed to earn before costs just to break even.
        out["gross_return_required"] = total_return + out["cost_drag"]

    if trades is not None and not trades.empty:
        n_round_trips = int((trades["side"] == "SELL").sum())
        out["n_fills"] = float(len(trades))
        out["n_round_trips"] = float(n_round_trips)
        out["total_notional"] = float(trades["notional"].sum())

        capital = initial_capital or start
        if capital and years > 0:
            out["annual_turnover"] = out["total_notional"] / capital / years

        if n_round_trips:
            wins = trades[(trades["side"] == "SELL") & (trades["realised_pnl"] > 0)]
            out["trade_win_rate"] = len(wins) / n_round_trips
            out["mean_cost_per_fill"] = float(trades["total_cost"].mean())

        for component in ("commission", "spread", "slippage"):
            if component in trades.columns:
                out[f"total_{component}"] = float(trades[component].sum())

    return out


def compare_to_benchmarks(
    strategy: dict[str, float], benchmarks: dict[str, dict[str, float]]
) -> pd.DataFrame:
    """Strategy against each benchmark on the identical window."""
    rows = [{"name": "TradeMind", **strategy}]
    for name, metrics in benchmarks.items():
        rows.append({"name": name, **metrics})

    cols = [
        "name",
        "total_return",
        "cagr",
        "sharpe",
        "sharpe_stderr",
        "max_drawdown",
        "annual_volatility",
        "cost_drag",
        "annual_turnover",
    ]
    frame = pd.DataFrame(rows)
    return frame[[c for c in cols if c in frame.columns]]


def render_report(
    metrics: dict[str, float],
    benchmarks: dict[str, dict[str, float]] | None = None,
) -> str:
    """Human-readable summary that leads with the caveats, not the Sharpe."""
    if not metrics:
        return "no results"

    lines = [
        f"period          {metrics['n_sessions']:.0f} sessions ({metrics['years']:.2f} years)",
        f"total return    {metrics['total_return']:+.2%}",
        f"CAGR            {metrics['cagr']:+.2%}",
        f"volatility      {metrics['annual_volatility']:.2%}",
        f"Sharpe          {metrics['sharpe']:+.2f} +/- {metrics['sharpe_stderr']:.2f}",
        f"Sortino         {metrics['sortino']:+.2f}",
        f"max drawdown    {metrics['max_drawdown']:.2%} "
        f"({metrics['max_drawdown_sessions']:.0f} sessions)",
        f"mean exposure   {metrics['mean_exposure']:.1%}",
    ]

    if "cost_drag" in metrics:
        lines += [
            "",
            f"total costs     {metrics['total_costs']:,.0f} "
            f"({metrics['cost_drag']:.2%} of capital)",
            f"gross needed    {metrics['gross_return_required']:+.2%} "
            "just to break even after costs",
        ]
        for component in ("commission", "spread", "slippage"):
            key = f"total_{component}"
            if key in metrics:
                lines.append(f"  {component:<13} {metrics[key]:,.0f}")

    if "n_round_trips" in metrics:
        lines += [
            "",
            f"round trips     {metrics['n_round_trips']:.0f} "
            f"({metrics.get('n_fills', 0):.0f} fills)",
            f"annual turnover {metrics.get('annual_turnover', 0):.1f}x",
        ]

    # Sharpe is uninterpretable on a short sample; say so rather than implying
    # otherwise by printing it plainly.
    if metrics["years"] < 3:
        lines.append(
            f"\nNOTE: {metrics['years']:.1f} years is too short to estimate "
            f"Sharpe reliably (stderr {metrics['sharpe_stderr']:.2f}). Treat "
            "the ratio as indicative only."
        )

    if benchmarks:
        lines += ["", "vs benchmarks (identical window, identical costs):"]
        lines.append(compare_to_benchmarks(metrics, benchmarks).to_string(index=False))

    return "\n".join(lines)
