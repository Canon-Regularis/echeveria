"""The Streamlit entry point. ``phytovision dashboard`` runs ``streamlit run`` on this file.

Streamlit executes it top to bottom on each rerun, so the ``__main__`` guard calls ``render``. The
analysed pipeline mirrors the API: ``Pipeline.default()`` unless ``PHYTOVISION_CONFIG`` or
``PHYTOVISION_MODEL_PATH`` is set.
"""

from __future__ import annotations

from phytovision.dashboard.analyze_tab import render_analyze_tab
from phytovision.dashboard.temporal_tab import render_temporal_tab
from phytovision.dashboard.theme import TERMINAL_CSS
from phytovision.serving import engine_from_env


def render() -> None:  # pragma: no cover: exercised only inside a running Streamlit server
    """Draw the two-tab terminal. Streamlit calls this on each rerun."""
    import streamlit as st

    st.set_page_config(page_title="phytovision", page_icon="🌱", layout="wide")
    st.markdown(TERMINAL_CSS, unsafe_allow_html=True)
    st.markdown("### PHYTOVISION // EXPLAINABLE WATER-STRESS TERMINAL")

    engine, conformal = engine_from_env()
    analyze_tab, temporal_tab = st.tabs(["ANALYZE", "TEMPORAL"])
    with analyze_tab:
        render_analyze_tab(engine, conformal)
    with temporal_tab:
        render_temporal_tab(engine)


if __name__ == "__main__":  # pragma: no cover: the streamlit runner sets __name__ to __main__
    render()
