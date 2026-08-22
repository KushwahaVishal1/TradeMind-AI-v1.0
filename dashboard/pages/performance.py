"""Performance: the equity curve, with its costs and caveats attached."""

from __future__ import annotations

import streamlit as st
from components.render import empty_state, render_panel

from trademind.reporting import build_performance_panel


def render(state, cfg, store) -> None:
    st.title("Performance")

    metrics = state.get("performance")
    if not metrics:
        empty_state("No backtest results yet.", "python main.py backtest")
        return

    render_panel(build_performance_panel(metrics, state.get("benchmark")))

    curve = state.get("equity_curve")
    if curve is not None and not curve.empty:
        st.subheader("Equity")
        st.line_chart(curve.set_index("date")["equity"])

        st.subheader("Exposure")
        st.caption(
            "Time spent invested. Low exposure inflates Sharpe by keeping "
            "volatility down while the strategy sits in cash."
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
        st.dataframe(trades.tail(100), use_container_width=True)
