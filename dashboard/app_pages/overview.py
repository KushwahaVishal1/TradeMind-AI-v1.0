"""Overview: the honest summary.

Panel order is set in ``reporting.panels.build_overview`` and is deliberate --
cost feasibility first, equity curve last. A dashboard that opens with a rising
curve invites the viewer to stop reading.
"""

from __future__ import annotations

import streamlit as st
from components.render import render_panel

from trademind.reporting import build_overview


def render(state, cfg, store) -> None:
    st.title("Overview")

    if not state:
        st.info(
            "Nothing computed yet. Start with:\n\n"
            "```\npython main.py init\npython main.py ingest\n"
            "python main.py features\n```"
        )
        return

    for panel in build_overview(state):
        render_panel(panel)

    with st.expander("What this dashboard does not show"):
        st.markdown(
            """
- **Survivorship bias.** The universe is today's large-caps, so names that
  were liquid in 2015 and have since delisted are silently excluded. Every
  backtest figure is optimistic by an unmeasured amount.
- **Reconstructed execution prices.** As-traded prices are derived from a free
  provider's split history. A missed split corrupts them for all prior dates.
- **A single final test.** Development results appear here. The locked
  final-test window is evaluated once, in the final report.
- **Live trading.** Nothing here has traded real money.
"""
        )
