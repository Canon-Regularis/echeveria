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
from typing import TYPE_CHECKING

import numpy as np
from fastapi import FastAPI, Form, HTTPException, Response, UploadFile

from phytovision.exceptions import InvalidImageError, PhytoVisionError
from phytovision.io import decode_rgb_bytes
from phytovision.models.conformal import SplitConformalClassifier
from phytovision.models.survival import fit_cohort_survival
from phytovision.pipeline import Pipeline
from phytovision.registries import FORECASTERS, SURVIVAL_MODELS
from phytovision.serving import attach_heads, engine_from_env
from phytovision.temporal import (
    FeatureHistory,
    plant_early_warnings,
    plant_forecasts,
    plant_trends,
)
from phytovision.types import AnalysisReport, Image
from phytovision.visualize import render_overlay, render_saliency_overlay

if TYPE_CHECKING:
    from phytovision.models.base import TrajectoryForecaster
    from phytovision.models.survival import SurvivalFit


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
        file: UploadFile, disease: bool = False, drought_stage: bool = False
    ) -> dict[str, object]:
        """Analyze one image. ``disease=true`` and ``drought_stage=true`` attach optional heads that
        are unvalidated placeholders, not diagnostics; their outputs carry a ``disclaimer``."""
        report = _run(
            attach_heads(engine, disease=disease, drought_stage=drought_stage), await file.read()
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
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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
            return _trend_payload(history, chosen, survival_model or None)
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


def _trend_payload(
    history: FeatureHistory,
    forecaster: TrajectoryForecaster | None = None,
    survival_model: str | None = "weibull-aft",
) -> dict[str, object]:
    """Serialize per-plant trends, series, the early warning, the forecast, and survival to JSON."""
    warnings = plant_early_warnings(history)
    forecasts = plant_forecasts(history, forecaster=forecaster)
    survival_fit, survival_note = _survival_fit(history, survival_model)
    per_plant = survival_fit.per_plant if survival_fit is not None else {}
    plants: dict[str, object] = {}
    for plant_id, trend in plant_trends(history).items():
        warning = warnings[plant_id]
        forecast = forecasts[plant_id]
        entry: dict[str, object] = {
            "direction": trend.direction,
            "slope": round(trend.slope, 6),
            "n": trend.n,
            "start_score": round(trend.start_score, 4),
            "end_score": round(trend.end_score, 4),
            "early_warning": {
                "flagged": warning.flagged,
                "pigment_slope": round(warning.pigment_slope, 6),
                "note": warning.note,
            },
            "forecast": {
                "method": forecast.method,
                "projected_scores": {
                    str(horizon): round(score, 4)
                    for horizon, score in forecast.projected_scores.items()
                },
                "lower": {str(h): round(v, 4) for h, v in forecast.lower.items()},
                "upper": {str(h): round(v, 4) for h, v in forecast.upper.items()},
                "interval_level": forecast.interval_level,
                "steps_to_stressed": forecast.steps_to_stressed,
                "confidence": round(forecast.confidence, 4),
            },
            "series": [
                {"timestamp": obs.timestamp, "score": round(obs.stress_score, 4)}
                for obs in history.series_for(plant_id)
            ],
        }
        if plant_id in per_plant:
            plant_survival = per_plant[plant_id]
            entry["survival"] = {
                "basis": plant_survival.basis,
                "median": plant_survival.median,
                "lower": plant_survival.lower,
                "upper": plant_survival.upper,
            }
        plants[plant_id] = entry

    disclaimer = (
        "early_warning and forecast are RGB-proxy extrapolations, not predictions; the "
        "prediction interval is an uncertainty estimate, not a measurement"
    )
    payload: dict[str, object] = {
        "plants": plants,
        "survival": _survival_summary(survival_fit),
        "disclaimer": disclaimer + _SURVIVAL_DISCLAIMER_SUFFIX if survival_fit else disclaimer,
    }
    if survival_note is not None:
        payload["survival_note"] = survival_note
    return payload


_SURVIVAL_DISCLAIMER_SUFFIX = (
    "; the survival estimate is a synthetic-trained RGB-proxy indication of time-to-wilt, not a "
    "validated prognosis"
)


def _survival_fit(
    history: FeatureHistory, survival_model: str | None
) -> tuple[SurvivalFit | None, str | None]:
    """Fit the cohort survival, or degrade to no survival when the stats extra is absent."""
    if not survival_model:
        return None, None
    try:
        return fit_cohort_survival(history, survival_model), None
    except ImportError as exc:  # survival is additive: keep the forecast, note the omission
        return None, str(exc)


def _survival_summary(fit: SurvivalFit | None) -> dict[str, object] | None:
    if fit is None:
        return None
    return {
        "model": fit.model_name,
        "cohort_median": fit.cohort_median,
        "cohort_median_ci": list(fit.cohort_median_ci),
        "concordance_index": fit.concordance_index,
        "concordance_note": "in-sample, optimistic; see the survival benchmark for held-out scores",
        "curve": [
            {"t": t, "survival": s, "lower": lo, "upper": hi}
            for t, s, lo, hi in zip(
                fit.curve.times, fit.curve.survival, fit.curve.lower, fit.curve.upper, strict=True
            )
        ],
        "disclaimer": fit.disclaimer,
    }


app = create_app()
