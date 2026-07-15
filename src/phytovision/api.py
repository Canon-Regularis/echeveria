"""HTTP API for the pipeline, built with FastAPI. Needs the 'api' extra: pip install -e ".[api]".

Run it with ``uvicorn phytovision.api:app`` or ``phytovision serve``. The served pipeline is
``Pipeline.default()`` unless the ``PHYTOVISION_CONFIG`` or ``PHYTOVISION_MODEL_PATH`` environment
variables are set (``phytovision serve --config/--model-path`` sets them). A model saved with
``train --calibrate`` adds a conformal label set to the ``/analyze`` response.
"""

from __future__ import annotations

import io

import numpy as np
from fastapi import FastAPI, HTTPException, Response, UploadFile
from PIL import Image as PILImage
from PIL import UnidentifiedImageError

from phytovision.exceptions import PhytoVisionError
from phytovision.models.conformal import SplitConformalClassifier
from phytovision.pipeline import Pipeline
from phytovision.serving import engine_from_env
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
    async def analyze(file: UploadFile) -> dict[str, object]:
        report = _run(engine, await file.read())
        payload = report.summary()
        if conformal is not None:
            label_set = conformal.predict_set(report.plant_features)
            payload["conformal"] = {"labels": list(label_set.labels), "alpha": label_set.alpha}
        return payload

    @app.post("/overlay")
    async def overlay(file: UploadFile) -> Response:
        image = _decode(await file.read())
        report = _run(engine, image)
        buffer = io.BytesIO()
        render_overlay(image, report).save(buffer, format="PNG")
        return Response(content=buffer.getvalue(), media_type="image/png")

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
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc


app = create_app()
