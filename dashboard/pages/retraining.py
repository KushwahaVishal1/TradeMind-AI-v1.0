"""Retraining: policy state, and the run history behind it."""

from __future__ import annotations

import streamlit as st
from components.render import empty_state


def render(state, cfg, store) -> None:
    st.title("Retraining governance")

    st.subheader("Policy")
    st.markdown(
        """
```
HEALTHY -> WARNING -> DEGRADED -> RETRAIN_REQUIRED
        -> RETRAINING -> VALIDATING -> PROMOTE | KEEP_CURRENT
```

1. **Drift alone never retrains.** Markets change distribution constantly; a
   model can be healthy on shifted inputs.
2. **A data-quality ERROR blocks retraining**, however severe everything else
   looks. When performance collapses *and* ingestion is erroring, the data is
   the likelier cause, and retraining on it bakes the corruption in.
3. **Insufficient new observations means wait.** A model fitted on a handful of
   rows differs from its predecessor by noise.
4. **A failed candidate has no path to production.** Enforced at the promotion
   decision and again at the registry. No override exists.
"""
    )

    report = state.get("health_report")
    if report is not None:
        st.subheader("Current state")
        st.code(report.render())
    else:
        st.caption("No health report in this session. Run `python main.py monitor`.")

    runs = state.get("runs")
    if runs is not None and not runs.empty:
        st.subheader("Run history")
        st.dataframe(
            runs[
                [
                    "run_id",
                    "mode",
                    "started_at",
                    "finished_at",
                    "status",
                    "git_commit",
                    "config_hash",
                ]
            ],
            use_container_width=True,
        )
    else:
        empty_state("No pipeline runs recorded.", "python main.py daily")

    st.subheader("Model registry")
    try:
        from trademind.experiments import ModelLifecycle

        summary = ModelLifecycle(store.conn).summary()
        if summary.empty:
            st.caption("No models registered yet.")
        else:
            st.dataframe(summary, use_container_width=True)
    except Exception as exc:
        st.caption(f"Registry unavailable: {exc}")
