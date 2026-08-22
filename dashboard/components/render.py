"""Shared Streamlit rendering for the reporting primitives."""

from __future__ import annotations

import streamlit as st

from trademind.reporting import Panel, Verdict

COLOURS = {
    Verdict.GOOD: "#1a7f37",
    Verdict.NEUTRAL: "#57606a",
    Verdict.CONCERN: "#9a6700",
    Verdict.BAD: "#cf222e",
    Verdict.UNKNOWN: "#57606a",
}


def render_panel(panel: Panel) -> None:
    """One panel: verdict, headline, metrics, caveats."""
    colour = COLOURS[panel.verdict]
    st.markdown(
        f"<h4 style='color:{colour};margin-bottom:0'>"
        f"{panel.verdict.symbol} {panel.title}</h4>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**{panel.headline}**")

    available = [m for m in panel.metrics if m.available]
    if available:
        columns = st.columns(min(4, len(available)))
        for col, metric in zip(columns * 10, available, strict=False):
            with col:
                st.metric(metric.label, metric.format_value())
                if metric.stderr is not None:
                    st.caption(f"± {metric.stderr:.4f}")
                if metric.note:
                    st.caption(metric.note)

    for caveat in panel.caveats:
        st.caption(f":warning: {caveat}")
    st.divider()


def empty_state(message: str, command: str = "") -> None:
    st.info(message + (f"\n\n```\n{command}\n```" if command else ""))
