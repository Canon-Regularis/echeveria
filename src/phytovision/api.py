"""HTTP API for the pipeline, built with FastAPI. Needs the 'api' extra: pip install -e ".[api]".

Run it with ``uvicorn phytovision.api:app`` or ``phytovision serve``.
"""

from __future__ import annotations

import io

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image as PILImage
from PIL import UnidentifiedImageError

from phytovision.exceptions import PhytoVisionError
from phytovision.pipeline import Pipeline


def create_app(pipeline: Pipeline | None = None) -> FastAPI:
    """Build the API. Pass a pipeline to override the default wiring."""
    engine = pipeline or Pipeline.default()
    app = FastAPI(title="phytovision")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/analyze")
    async def analyze(file: UploadFile) -> dict[str, object]:
        raw = await file.read()
        try:
            image = np.asarray(PILImage.open(io.BytesIO(raw)).convert("RGB"))
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc
        try:
            return engine.analyze(image).summary()
        except PhytoVisionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
