"""Drift: distribution shift, and what the monitor can actually see."""

from __future__ import annotations

import streamlit as st
from components.render import empty_state, render_panel

from trademind.reporting import build_detectability_panel, build_drift_panel


def render(state, cfg, store) -> None:
    st.title("Drift and monitoring sensitivity")

    render_panel(build_drift_panel(state.get("drift_summary")))

    resolved = state.get("resolved")
    if resolved is not None and not resolved.empty:
        from trademind.monitoring import rolling_windows

        render_panel(build_detectability_panel(rolling_windows(resolved)))
    else:
        empty_state(
            "No resolved predictions yet, so monitoring sensitivity cannot be assessed.",
            "python main.py daily",
        )

    drift = state.get("drift")
    if drift is not None and not drift.empty:
        st.subheader("Per-feature")
        st.dataframe(
            drift[
                [
                    "feature",
                    "psi",
                    "severity",
                    "ks_pvalue",
                    "ks_significant_fdr",
                    "missing_rate_current",
                ]
            ],
            use_container_width=True,
        )
        st.caption(
            "PSI is the primary signal because it measures how much a "
            "distribution moved. A p-value measures only whether it moved at "
            "all, and with thousands of observations that detects shifts far "
            "too small to matter."
        )
