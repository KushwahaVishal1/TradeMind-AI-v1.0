"""Benchmarks, computed on the identical window as the strategy.

Two of them, and the pair is deliberate:

``buy_and_hold``
    Equal-weight the universe on day one and never trade. The passive
    alternative -- what you get for doing nothing.

``equal_weight_rebalanced``
    Equal weight, rebalanced monthly, paying the same costs. Isolates whether
    any strategy return comes from *selection* or merely from rebalancing.

Both pay the same transaction costs. A cost-free benchmark against a
cost-charged strategy is not a comparison, and it is the single easiest way to
make a losing strategy look competitive.
"""

from __future__ import annotations

import math
from datetime import date

import pandas as pd

from .config import BacktestConfig, compute_costs
from .portfolio import Portfolio


def _open_prices(bars: dict[str, pd.DataFrame], when: date) -> dict[str, float]:
    out = {}
    for symbol, df in bars.items():
        row = df[df["date"] == pd.Timestamp(when)]
        if not row.empty:
            price = float(row["open_raw"].iloc[0])
            if price > 0:
                out[symbol] = price
    return out


def _close_prices(bars: dict[str, pd.DataFrame], when: date) -> dict[str, float]:
    out = {}
    for symbol, df in bars.items():
        row = df[df["date"] == pd.Timestamp(when)]
        if not row.empty:
            price = float(row["close_raw"].iloc[0])
            if price > 0:
                out[symbol] = price
    return out


def _apply_actions(portfolio: Portfolio, bars: dict[str, pd.DataFrame], when: date) -> None:
    for symbol, df in bars.items():
        row = df[df["date"] == pd.Timestamp(when)]
        if row.empty:
            continue
        ratio = float(row["split_ratio"].iloc[0])
        if ratio != 1.0:
            portfolio.apply_split(symbol, ratio, when)
        dividend = float(row["dividend"].iloc[0])
        if dividend > 0:
            portfolio.apply_dividend(symbol, dividend, when)


def buy_and_hold(
    bars: dict[str, pd.DataFrame],
    sessions: list[date],
    config: BacktestConfig,
) -> pd.DataFrame:
    """Equal-weight on the first session, then hold."""
    portfolio = Portfolio(config.initial_capital)
    curve = []

    for i, when in enumerate(sessions):
        _apply_actions(portfolio, bars, when)

        if i == 1:  # buy on the second session, matching the t+1 rule
            prices = _open_prices(bars, when)
            if prices:
                weight = 1.0 / len(prices)
                for symbol, price in prices.items():
                    shares = math.floor((weight * config.initial_capital * 0.999) / price)
                    if shares > 0:
                        costs = compute_costs(shares * price, config.costs)
                        if shares * price + costs.total <= portfolio.cash:
                            portfolio.buy(symbol, shares, price, costs, when)

        closes = _close_prices(bars, when)
        if closes:
            portfolio.reconcile(closes, when)
            curve.append(portfolio.snapshot(when, closes))

    return pd.DataFrame(curve)


def equal_weight_rebalanced(
    bars: dict[str, pd.DataFrame],
    sessions: list[date],
    config: BacktestConfig,
    rebalance_every: int = 21,
) -> pd.DataFrame:
    """Equal weight, rebalanced periodically, paying the same costs."""
    portfolio = Portfolio(config.initial_capital)
    curve = []

    for i, when in enumerate(sessions):
        _apply_actions(portfolio, bars, when)

        if i > 0 and i % rebalance_every == 1:
            prices = _open_prices(bars, when)
            if prices:
                for symbol in list(portfolio.open_symbols):
                    price = prices.get(symbol)
                    holding = portfolio.position(symbol)
                    if price and not holding.is_flat:
                        costs = compute_costs(holding.shares * price, config.costs)
                        portfolio.sell(symbol, holding.shares, price, costs, when)

                equity = portfolio.cash
                weight = 1.0 / len(prices)
                for symbol, price in prices.items():
                    shares = math.floor((weight * equity * 0.999) / price)
                    if shares > 0:
                        costs = compute_costs(shares * price, config.costs)
                        if shares * price + costs.total <= portfolio.cash:
                            portfolio.buy(symbol, shares, price, costs, when)

        closes = _close_prices(bars, when)
        if closes:
            portfolio.reconcile(closes, when)
            curve.append(portfolio.snapshot(when, closes))

    return pd.DataFrame(curve)
