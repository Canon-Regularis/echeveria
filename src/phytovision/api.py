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
from phytovision.exceptions import InvalidImageError, ModelNotFittedError, PhytoVisionError
from phytovision.io import decode_rgb_bytes
from phytovision.models.conformal import SplitConformalClassifier
from phytovision.pipeline import Pipeline
from phytovision.registries import FORECASTERS, SURVIVAL_MODELS
from phytovision.serving import attach_heads, engine_from_env
from phytovision.temporal import FeatureHistory
from phytovision.types import AnalysisReport, Image
from phytovision.visualize import render_overlay, render_saliency_overlay

# A generous ceiling for one plant photo: large enough for a real high-resolution phone image, and
# small enough that one oversized upload cannot be materialized in memory or handed to the decoder.
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
# A ceiling on one /trend batch, so a single request cannot enqueue an unbounded number of analyses.
_MAX_TREND_FILES = 500


def create_app(
    pipeline: Pipeline | None = None, conformal: SplitConformalClassifier | None = None
) -> FastAPI:
    """Build the API. Pass a pipeline and/or conformal wrapper to override the env defaults."""
    engine, conformal = engine_from_env(pipeline, conformal)
    # An uncalibrated wrapper would raise ModelNotFittedError on every /analyze request; fail at
    # construction instead, so the misconfiguration surfaces once when the app is built.
    if conformal is not None and conformal.qhat is None:
        raise ModelNotFittedError("conformal wrapper is not calibrated; call calibrate() first")
    app = FastAPI(title="phytovision")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/analyze")
    def analyze(
        file: UploadFile,
        disease: bool = False,
        drought_stage: bool = False,
        physiology: bool = False,
    ) -> dict[str, object]:
        """Analyze one image. ``disease``, ``drought_stage``, and ``physiology`` query flags attach
        optional heads that are unvalidated placeholders or proxies, not diagnostics; their outputs
        carry a ``disclaimer``.

        A plain ``def`` handler (not ``async def``) so Starlette runs the CPU-bound analysis in its
        worker threadpool rather than on the event loop, keeping ``/health`` and other requests
        responsive while one image is being scored."""
        report = _run(
            attach_heads(
                engine, disease=disease, drought_stage=drought_stage, physiology=physiology
            ),
            _read_capped(file),
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
    def overlay(file: UploadFile) -> Response:
        image = _decode(_read_capped(file))
        report = _run(engine, image)
        buffer = io.BytesIO()
        render_overlay(image, report).save(buffer, format="PNG")
        return Response(content=buffer.getvalue(), media_type="image/png")

    @app.post("/saliency")
    def saliency(file: UploadFile) -> Response:
        """A pigment saliency overlay: where colour drivers moved the score. It is an RGB proxy."""
        image = _decode(_read_capped(file))
        report = _run(engine, image)
        buffer = io.BytesIO()
        render_saliency_overlay(image, report, engine.model).save(buffer, format="PNG")
        return Response(content=buffer.getvalue(), media_type="image/png")

    @app.post("/trend")
    def trend(
        files: list[UploadFile],
        plant_id: list[str] = Form(...),
        timestamp: list[str] = Form(...),
        forecaster: str = "linear-trend",
        survival_model: str = "weibull-aft",
    ) -> dict[str, object]:
        """Fit a stress trend over tagged images. ``forecaster`` picks the trajectory model behind
        each plant's ``forecast`` block; ``survival_model`` picks the time-to-wilt model behind the
        ``survival`` blocks. Every estimate is a synthetic-trained RGB proxy, not a prognosis.

        A plain ``def`` handler for the same reason as ``/analyze``: the per-image analysis runs in
        the threadpool, off the event loop."""
        if not files or not len(files) == len(plant_id) == len(timestamp):
            raise HTTPException(
                status_code=400,
                detail="files, plant_id, and timestamp must be non-empty and the same length",
            )
        if len(files) > _MAX_TREND_FILES:
            raise HTTPException(
                status_code=413,
                detail=f"a /trend batch is limited to {_MAX_TREND_FILES} files",
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
            history.record(pid, when, _run(engine, _read_capped(upload)))
        try:
            return trend_payload(history, chosen, survival_model or None)
        except ImportError as exc:  # a forecaster whose optional extra is not installed
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def _read_capped(upload: UploadFile) -> bytes:
    """Read an upload synchronously (the handlers are sync, so this runs in the threadpool), with a
    hard byte cap. Starlette has already spooled the raw multipart body, so this does not bound
    transfer; reading one byte past the limit and checking the length rejects an oversized body
    before it becomes one large in-memory bytes object or reaches the decoder, and also catches a
    client that lies about ``size``."""
    if upload.size is not None and upload.size > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"upload exceeds {_MAX_UPLOAD_BYTES} bytes")
    data = upload.file.read(_MAX_UPLOAD_BYTES + 1)
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"upload exceeds {_MAX_UPLOAD_BYTES} bytes")
    return data


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
