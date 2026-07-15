"""HTTP API for the pipeline, built with FastAPI. Needs the 'api' extra: pip install -e ".[api]".

Run it with ``uvicorn phytovision.api:app`` or ``phytovision serve``. The served pipeline is
``Pipeline.default()`` unless the ``PHYTOVISION_CONFIG`` or ``PHYTOVISION_MODEL_PATH`` environment
variables are set (``phytovision serve --config/--model-path`` sets them). A model saved with
``train --calibrate`` adds a conformal label set to the ``/analyze`` response.
``/analyze?disease=true`` attaches the placeholder disease head, and ``/trend`` fits a stress trend
over a batch of tagged images.
"""

from __future__ import annotations

import io

import numpy as np
from fastapi import FastAPI, Form, HTTPException, Response, UploadFile
from PIL import Image as PILImage
from PIL import UnidentifiedImageError
from PIL.Image import DecompressionBombError

from phytovision.exceptions import PhytoVisionError
from phytovision.models.conformal import SplitConformalClassifier
from phytovision.pipeline import Pipeline
from phytovision.serving import attach_heads, engine_from_env
from phytovision.temporal import FeatureHistory, plant_trends
from phytovision.types import AnalysisReport, Image
from phytovision.visualize import render_overlay


def create_app(
    pipeline: Pipeline | None = None, conformal: SplitConformalClassifier | None = None
) -> FastAPI:
    """Build the API. Pass a pipeline and/or conformal wrapper to override the env defaults."""
    engine, conformal = engine_from_env(pipeline, conformal)
    app = FastAPI(title="phytovision")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/analyze")
    async def analyze(file: UploadFile, disease: bool = False) -> dict[str, object]:
        """Analyze one image. ``disease=true`` attaches the disease head, an unvalidated placeholder
        (not a diagnostic); its probabilities carry a ``disclaimer`` in the response."""
        report = _run(attach_heads(engine, disease=disease), await file.read())
        payload = report.summary()
        if conformal is not None:
            label_set = conformal.predict_set(report.plant_features)
            payload["conformal"] = {"labels": list(label_set.labels), "alpha": label_set.alpha}
        if report.head_outputs:  # summary() lists head names only; expose the values here
            payload["head_outputs"] = report.head_outputs
            if "disease" in report.head_outputs:
                payload["disclaimer"] = "disease is an unvalidated placeholder, not a diagnostic"
        return payload

    @app.post("/overlay")
    async def overlay(file: UploadFile) -> Response:
        image = _decode(await file.read())
        report = _run(engine, image)
        buffer = io.BytesIO()
        render_overlay(image, report).save(buffer, format="PNG")
        return Response(content=buffer.getvalue(), media_type="image/png")

    @app.post("/trend")
    async def trend(
        files: list[UploadFile],
        plant_id: list[str] = Form(...),
        timestamp: list[str] = Form(...),
    ) -> dict[str, object]:
        if not files or not len(files) == len(plant_id) == len(timestamp):
            raise HTTPException(
                status_code=400,
                detail="files, plant_id, and timestamp must be non-empty and the same length",
            )
        history = FeatureHistory()
        for upload, pid, when in zip(files, plant_id, timestamp, strict=True):
            history.record(pid, when, _run(engine, await upload.read()))
        return _trend_payload(history)

    return app


def _run(engine: Pipeline, data: bytes | Image) -> AnalysisReport:
    image = data if isinstance(data, np.ndarray) else _decode(data)
    try:
        return engine.analyze(image)
    except PhytoVisionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _decode(data: bytes) -> Image:
    try:
        return np.asarray(PILImage.open(io.BytesIO(data)).convert("RGB"))
    except (UnidentifiedImageError, DecompressionBombError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc


def _trend_payload(history: FeatureHistory) -> dict[str, object]:
    """Serialize per-plant stress trends and their time-ordered score series to plain JSON."""
    plants: dict[str, object] = {}
    for plant_id, trend in plant_trends(history).items():
        plants[plant_id] = {
            "direction": trend.direction,
            "slope": round(trend.slope, 6),
            "n": trend.n,
            "start_score": round(trend.start_score, 4),
            "end_score": round(trend.end_score, 4),
            "series": [
                {"timestamp": obs.timestamp, "score": round(obs.stress_score, 4)}
                for obs in history.series_for(plant_id)
            ],
        }
    return {"plants": plants}


app = create_app()
