"""Model: skill against baselines, and calibration."""

from __future__ import annotations

import streamlit as st
from components.render import empty_state, render_panel

from trademind.reporting import build_skill_panel


def render(state, cfg, store) -> None:
    st.title("Model")

    if state.get("auc") is None:
        empty_state("No evaluated model yet.", "python main.py train")
        return

    render_panel(
        build_skill_panel(
            state.get("auc"),
            state.get("auc_stderr"),
            state.get("majority_accuracy"),
            state.get("accuracy"),
        )
    )

    resolved = state.get("resolved")
    if resolved is not None and not resolved.empty:
        from trademind.ensemble import reliability_table

        st.subheader("Calibration")
        st.caption(
            "Does an 80% prediction win 80% of the time? Read the `count` "
            "column: a large gap in a bin holding twelve rows is noise."
        )
        table = reliability_table(
            resolved["actual_direction"], resolved["calibrated_probability"]
        )
        if not table.empty:
            st.dataframe(table, use_container_width=True)
            st.line_chart(table.set_index("mean_predicted")[["observed_frequency"]])

    st.subheader("Expected range")
    st.caption(
        "Honest, leak-free ROC-AUC on next-day NSE large-cap direction sits "
        "around 0.51-0.54. Above 0.58 is treated as a leakage signal, not a "
        "result — the leakage suite gets pointed at it before anything is "
        "celebrated."
    )
