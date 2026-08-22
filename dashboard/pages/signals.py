"""Signals: today's decisions, rejections included.

Showing only BUYs would hide the most informative rows. A rejection with reason
``BELOW_COST_FLOOR`` means the model was confident and the trade still was not
worth making -- that is the system working, and it is invisible on a page that
lists only trades.
"""

from __future__ import annotations

import streamlit as st
from components.render import empty_state

from trademind.reporting import build_signals_table, signal_counts


def render(state, cfg, store) -> None:
    st.title("Signals")

    predictions = state.get("predictions")
    if predictions is None or predictions.empty:
        empty_state("No predictions stored yet.", "python main.py daily")
        return

    counts = signal_counts(predictions)
    columns = st.columns(len(counts) or 1)
    for col, (label, value) in zip(columns, counts.items()):
        col.metric(label, value)

    if counts.get("REJECTED"):
        st.caption(
            "Rejections preserve the existing position rather than "
            "liquidating. A `BELOW_COST_FLOOR` rejection means the model was "
            "confident but the expected move did not clear the round-trip cost."
        )

    st.subheader("Latest decisions")
    st.dataframe(build_signals_table(predictions), use_container_width=True)

    st.subheader("Lineage")
    st.caption(
        "Every row carries the model, feature, decision and threshold versions "
        "that produced it, plus the training window. A prediction whose "
        "`training_end` is not strictly before its `prediction_date` is "
        "rejected at write time."
    )
    lineage = [
        c
        for c in (
            "symbol",
            "prediction_date",
            "execution_date",
            "model_version",
            "feature_version",
            "decision_version",
            "threshold_version",
            "training_start",
            "training_end",
            "run_id",
        )
        if c in predictions.columns
    ]
    st.dataframe(predictions[lineage].head(50), use_container_width=True)
