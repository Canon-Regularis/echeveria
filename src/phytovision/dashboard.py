"""Streamlit dashboard for interactive water-stress analysis.

Needs the 'dashboard' extra: pip install -e ".[dashboard]". Launch it with
``phytovision dashboard``, which runs ``streamlit run`` on this module. The analysed pipeline
mirrors the API: ``Pipeline.default()`` unless ``PHYTOVISION_CONFIG`` or ``PHYTOVISION_MODEL_PATH``
is set.

Streamlit and plotly are imported lazily inside ``render`` so the helpers below stay importable,
and therefore testable, with only the base dependencies installed.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image as PILImage
from PIL import UnidentifiedImageError

from phytovision.exceptions import InvalidImageError, PhytoVisionError
from phytovision.serving import engine_from_env
from phytovision.types import AnalysisReport, Image


def decode_image(data: bytes) -> Image:
    """Decode uploaded bytes into an RGB array, raising a clean domain error on junk input."""
    try:
        return np.asarray(PILImage.open(io.BytesIO(data)).convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError(f"invalid image: {exc}") from exc


def reason_rows(report: AnalysisReport) -> list[dict[str, object]]:
    """Flatten the explanation into display rows, strongest driver first."""
    return [
        {
            "feature": reason.feature,
            "value": round(reason.value, 4),
            "effect on stress": reason.direction,
            "contribution": round(reason.contribution, 4),
            "why": reason.description,
        }
        for reason in report.explanation.reasons
    ]


def contribution_series(report: AnalysisReport) -> tuple[list[str], list[float]]:
    """Feature names and signed contributions for a bar chart, largest magnitude first."""
    ranked = sorted(
        report.explanation.reasons, key=lambda reason: abs(reason.contribution), reverse=True
    )
    return [reason.feature for reason in ranked], [reason.contribution for reason in ranked]


def render() -> None:  # pragma: no cover - exercised only inside a running Streamlit server
    """The Streamlit entry point. Streamlit executes this module top to bottom on each rerun."""
    import plotly.graph_objects as go
    import streamlit as st

    st.set_page_config(page_title="phytovision", page_icon="🌱", layout="wide")
    st.title("phytovision - explainable water-stress analysis")
    st.caption("Upload a plant image. The verdict, the overlay, and the reasons behind it appear.")

    engine, conformal = engine_from_env()
    upload = st.file_uploader("Plant image", type=["png", "jpg", "jpeg", "bmp", "tiff"])
    if upload is None:
        st.info("Waiting for an image.")
        return

    try:
        image = decode_image(upload.getvalue())
        report = engine.analyze(image)
    except PhytoVisionError as exc:
        # decode_image raises InvalidImageError; analyze can raise SegmentationError and friends.
        # Catch the shared base so a bad upload shows a message, matching the API's behaviour.
        st.error(str(exc))
        return

    from phytovision.visualize import render_overlay

    stress = report.stress
    left, right = st.columns(2)
    left.image(image, caption="input", use_container_width=True)
    right.image(render_overlay(image, report), caption="overlay", use_container_width=True)

    st.subheader(f"Verdict: {stress.label.upper()}")
    score, confidence, regions = st.columns(3)
    score.metric("stress score", f"{stress.score:.2f}")
    confidence.metric("confidence", f"{stress.confidence:.2f}")
    regions.metric(f"regions ({report.regions.kind})", len(report.regions))

    if conformal is not None:
        label_set = conformal.predict_set(report.plant_features)
        coverage = round((1.0 - label_set.alpha) * 100)
        members = ", ".join(label_set.labels) or "(empty)"
        st.write(f"Conformal set ({coverage}% coverage): {members}")

    for name, output in report.head_outputs.items():
        st.write(f"Head '{name}': {output}")

    features, contributions = contribution_series(report)
    if features:
        st.subheader("What drove the verdict")
        figure = go.Figure(go.Bar(x=contributions, y=features, orientation="h"))
        figure.update_layout(yaxis={"autorange": "reversed"}, height=32 * len(features) + 120)
        st.plotly_chart(figure, use_container_width=True)
        st.dataframe(reason_rows(report), use_container_width=True)


if __name__ == "__main__":  # pragma: no cover - the streamlit runner sets __name__ to __main__
    render()
