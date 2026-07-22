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

from phytovision.api_payloads import trend_payload
from phytovision.exceptions import InvalidImageError, PhytoVisionError
from phytovision.io import decode_rgb_bytes
from phytovision.models.conformal import SplitConformalClassifier
from phytovision.pipeline import Pipeline
from phytovision.registries import FORECASTERS, SURVIVAL_MODELS
from phytovision.serving import attach_heads, engine_from_env
from phytovision.temporal import FeatureHistory
from phytovision.types import AnalysisReport, Image
from phytovision.visualize import render_overlay, render_saliency_overlay


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
    async def analyze(
        file: UploadFile,
        disease: bool = False,
        drought_stage: bool = False,
        physiology: bool = False,
    ) -> dict[str, object]:
        """Analyze one image. ``disease``, ``drought_stage``, and ``physiology`` query flags attach
        optional heads that are unvalidated placeholders or proxies, not diagnostics; their outputs
        carry a ``disclaimer``."""
        report = _run(
            attach_heads(
                engine, disease=disease, drought_stage=drought_stage, physiology=physiology
            ),
            await file.read(),
        )
        payload = report.summary()
        if conformal is not None:
            label_set = conformal.predict_set(report.plant_features)
            payload["conformal"] = {"labels": list(label_set.labels), "alpha": label_set.alpha}
        notes: list[str] = []
        if report.head_outputs:  # summary() lists head names only; expose the values here
            payload["head_outputs"] = report.head_outputs
            if "disease" in report.head_outputs:
                notes.append("disease is an unvalidated placeholder, not a diagnostic")
            if "drought_stage" in report.head_outputs:
                notes.append(
                    "drought_stage is a literature-motivated rule set, not a diagnosis; its "
                    "physiology proxies are crude RGB indices, not measurements"
                )
            if "physiology" in report.head_outputs:
                notes.append(
                    "physiology reports crude RGB proxies for water potential, stomatal "
                    "conductance, and transpiration, not measured physiology"
                )
        if not report.quality.usable:
            notes.append("input quality is low, so the score may be unreliable")
        if notes:
            payload["disclaimer"] = "; ".join(notes)
        return payload

    @app.post("/overlay")
    async def overlay(file: UploadFile) -> Response:
        image = _decode(await file.read())
        report = _run(engine, image)
        buffer = io.BytesIO()
        render_overlay(image, report).save(buffer, format="PNG")
        return Response(content=buffer.getvalue(), media_type="image/png")

    @app.post("/saliency")
    async def saliency(file: UploadFile) -> Response:
        """A pigment saliency overlay: where colour drivers moved the score. It is an RGB proxy."""
        image = _decode(await file.read())
        report = _run(engine, image)
        buffer = io.BytesIO()
        render_saliency_overlay(image, report, engine.model).save(buffer, format="PNG")
        return Response(content=buffer.getvalue(), media_type="image/png")

    @app.post("/trend")
    async def trend(
        files: list[UploadFile],
        plant_id: list[str] = Form(...),
        timestamp: list[str] = Form(...),
        forecaster: str = "linear-trend",
        survival_model: str = "weibull-aft",
    ) -> dict[str, object]:
        """Fit a stress trend over tagged images. ``forecaster`` picks the trajectory model behind
        each plant's ``forecast`` block; ``survival_model`` picks the time-to-wilt model behind the
        ``survival`` blocks. Every estimate is a synthetic-trained RGB proxy, not a prognosis."""
        if not files or not len(files) == len(plant_id) == len(timestamp):
            raise HTTPException(
                status_code=400,
                detail="files, plant_id, and timestamp must be non-empty and the same length",
            )
        try:
            chosen = FORECASTERS.create(forecaster)
        except KeyError as exc:
            # KeyError.__str__ repr-quotes its message, so read the raw text to keep it clean.
            detail = str(exc.args[0]) if exc.args else str(exc)
            raise HTTPException(status_code=400, detail=detail) from exc
        if survival_model and survival_model not in SURVIVAL_MODELS:
            available = SURVIVAL_MODELS.names()
            raise HTTPException(
                status_code=400,
                detail=f"unknown survival model {survival_model!r}; available: {available}",
            )
        history = FeatureHistory()
        for upload, pid, when in zip(files, plant_id, timestamp, strict=True):
            history.record(pid, when, _run(engine, await upload.read()))
        try:
            return trend_payload(history, chosen, survival_model or None)
        except ImportError as exc:  # a forecaster whose optional extra is not installed
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def _run(engine: Pipeline, data: bytes | Image) -> AnalysisReport:
    image = data if isinstance(data, np.ndarray) else _decode(data)
    try:
        return engine.analyze(image)
    except PhytoVisionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _decode(data: bytes) -> Image:
    try:
        return decode_rgb_bytes(data)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


app = create_app()
