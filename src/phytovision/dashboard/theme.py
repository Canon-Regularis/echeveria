"""The dashboard's visual theme: the terminal stylesheet and the shared plotly layout."""

from __future__ import annotations

TERMINAL_CSS = """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1500px; }
h1, h2, h3 { text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="stMetricLabel"] { text-transform: uppercase; letter-spacing: 0.04em; opacity: 0.75; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
</style>
"""

DARK_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "margin": {"l": 8, "r": 8, "t": 8, "b": 8},
}
