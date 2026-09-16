"""Performance: the equity curve, with its costs and caveats attached."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from components.render import empty_state, render_panel

from trademind.reporting import build_performance_panel


def render(state, cfg, store) -> None:
    st.title("Performance")
    st.caption(
        "Historical development backtest. "
        "Daily predictions do not update this equity curve."
    )

    metrics = state.get("performance")
    if not metrics:
        empty_state("No backtest results yet.", "python main.py backtest")
        return

    render_panel(build_performance_panel(metrics, state.get("benchmark")))

    curve = state.get("equity_curve")
    if curve is not None and not curve.empty:
        # A running session can still receive CSV strings from an older loader.
        # Normalize at the rendering boundary as well, without mutating state.
        curve = curve.copy()
        curve["date"] = pd.to_datetime(curve["date"])
        curve = curve.sort_values("date")
        st.caption(
            f"Backtest period: {curve['date'].min():%d %b %Y} to "
            f"{curve['date'].max():%d %b %Y} | {len(curve):,} sessions"
        )
        if "n_positions" in curve and curve["n_positions"].eq(0).all():
            st.info(
                "No positions were opened in this backtest. Equity stayed in cash; "
                "the flat line does not measure prediction accuracy."
            )
        st.subheader("Equity")
        st.line_chart(curve, x="date", y="equity", x_label="Date", y_label="Equity (INR)")

        st.subheader("Exposure")
        st.caption(
            "Fraction of portfolio equity invested each session. "
            "Zero means the portfolio is entirely in cash."
        )
        st.area_chart(curve.set_index("date")["exposure"])

    trades = state.get("trades")
    if trades is not None and not trades.empty:
        st.subheader("Cost breakdown")
        st.caption(
            "Itemised so the question 'which assumption is doing the work?' "
            "has an answer. Spread and slippage are the least certain."
        )
        totals = {
            component: float(trades[component].sum())
            for component in ("commission", "spread", "slippage")
            if component in trades.columns
        }
        columns = st.columns(len(totals) or 1)
        for col, (name, value) in zip(columns, totals.items(), strict=False):
            col.metric(name.title(), f"{value:,.0f}")

        st.subheader("Trades")
        st.dataframe(trades.tail(100), width="stretch")
