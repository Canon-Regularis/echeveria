"""The FastAPI /analyze endpoint (F10). Skipped if fastapi/httpx are absent."""

from __future__ import annotations

import io

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from phytovision.api import create_app


def _png_bytes(image: np.ndarray) -> bytes:
    from PIL import Image as PILImage

    buffer = io.BytesIO()
    PILImage.fromarray((image * 255).astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health() -> None:
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_returns_summary(healthy_image) -> None:
    client = TestClient(create_app())
    files = {"file": ("plant.png", _png_bytes(healthy_image), "image/png")}
    response = client.post("/analyze", files=files)
    assert response.status_code == 200
    assert set(response.json()["stress"]) == {"score", "confidence", "label", "model"}


def test_analyze_rejects_non_image() -> None:
    client = TestClient(create_app())
    files = {"file": ("bad.png", b"not an image", "image/png")}
    response = client.post("/analyze", files=files)
    assert response.status_code == 400


def test_analyze_rejects_a_decompression_bomb(healthy_image, monkeypatch) -> None:
    import PIL.Image

    # A decompression-bomb image must be a clean 400, not an uncaught 500.
    monkeypatch.setattr(PIL.Image, "MAX_IMAGE_PIXELS", 4)
    client = TestClient(create_app())
    files = {"file": ("plant.png", _png_bytes(healthy_image), "image/png")}
    assert client.post("/analyze", files=files).status_code == 400


def test_overlay_endpoint_returns_png(healthy_image) -> None:
    client = TestClient(create_app())
    files = {"file": ("plant.png", _png_bytes(healthy_image), "image/png")}
    response = client.post("/overlay", files=files)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_saliency_endpoint_returns_png(healthy_image) -> None:
    client = TestClient(create_app())
    files = {"file": ("plant.png", _png_bytes(healthy_image), "image/png")}
    response = client.post("/saliency", files=files)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def _calibrated_conformal():
    from phytovision.models.conformal import SplitConformalClassifier
    from phytovision.models.stress.heuristic import HeuristicStressModel
    from phytovision.types import PlantFeatures

    calib = [PlantFeatures(values={"colour.gcc_mean": v}, region_count=1) for v in (0.2, 0.35, 0.5)]
    return SplitConformalClassifier(HeuristicStressModel(), alpha=0.1).calibrate(calib, [0, 0, 1])


def test_analyze_includes_conformal_when_wrapped(healthy_image) -> None:
    client = TestClient(create_app(conformal=_calibrated_conformal()))
    files = {"file": ("plant.png", _png_bytes(healthy_image), "image/png")}
    payload = client.post("/analyze", files=files).json()
    assert "labels" in payload["conformal"]
    assert payload["conformal"]["alpha"] == 0.1


def test_env_model_path_wires_the_served_model(healthy_image, tmp_path, monkeypatch) -> None:
    pytest.importorskip("joblib")
    path = tmp_path / "calibrated.joblib"
    _calibrated_conformal().save(path)
    monkeypatch.setenv("PHYTOVISION_MODEL_PATH", str(path))

    client = TestClient(create_app())  # pipeline=None, so it reads the environment
    files = {"file": ("plant.png", _png_bytes(healthy_image), "image/png")}
    payload = client.post("/analyze", files=files).json()
    assert "conformal" in payload


def test_analyze_disease_true_exposes_head_values(stressed_image) -> None:
    client = TestClient(create_app())
    files = {"file": ("plant.png", _png_bytes(stressed_image), "image/png")}
    payload = client.post("/analyze", files=files, params={"disease": "true"}).json()
    disease = payload["head_outputs"]["disease"]
    assert set(disease) == {"healthy", "lesion-like"}
    assert all(0.0 <= value <= 1.0 for value in disease.values())  # real probabilities, not labels
    assert sum(disease.values()) == pytest.approx(1.0)
    assert "disclaimer" in payload  # the placeholder is labelled for API clients


def test_analyze_omits_head_outputs_by_default(healthy_image) -> None:
    client = TestClient(create_app())
    files = {"file": ("plant.png", _png_bytes(healthy_image), "image/png")}
    payload = client.post("/analyze", files=files).json()
    assert "head_outputs" not in payload
    assert "disclaimer" not in payload


def test_analyze_drought_stage_exposes_the_head(healthy_image) -> None:
    client = TestClient(create_app())
    files = {"file": ("plant.png", _png_bytes(healthy_image), "image/png")}
    payload = client.post("/analyze", files=files, params={"drought_stage": "true"}).json()
    stage = payload["head_outputs"]["drought_stage"]
    assert stage["stage"] in {"well-watered", "early-stress", "moderate", "severe"}
    assert set(stage["markers"]) == {"pigment", "turgor_loss", "necrosis"}
    assert "disclaimer" in payload  # the placeholder is labelled for API clients


def test_trend_sorts_by_timestamp_not_upload_order(healthy_image, stressed_image) -> None:
    client = TestClient(create_app())
    # Upload in reverse chronological order, so a correct response proves the timestamp sort rather
    # than echoing upload order: the stressed image is tagged later, the healthy image earlier.
    files = [
        ("files", ("b.png", _png_bytes(stressed_image), "image/png")),
        ("files", ("a.png", _png_bytes(healthy_image), "image/png")),
    ]
    data = {"plant_id": ["p1", "p1"], "timestamp": ["2026-03-02", "2026-03-01"]}
    response = client.post("/trend", files=files, data=data)
    assert response.status_code == 200
    plant = response.json()["plants"]["p1"]
    assert plant["n"] == 2
    series = plant["series"]
    assert [point["timestamp"] for point in series] == ["2026-03-01", "2026-03-02"]  # chronological
    assert series[0]["score"] < series[1]["score"]  # healthy (earlier) below stressed (later)
    assert plant["direction"] == "rising"
    assert set(plant["early_warning"]) == {"flagged", "pigment_slope", "note"}
    assert isinstance(plant["early_warning"]["flagged"], bool)
    assert set(plant["forecast"]) == {"projected_scores", "steps_to_stressed", "confidence"}
    assert "disclaimer" in response.json()  # early_warning and forecast are labelled RGB proxies


def test_trend_rejects_mismatched_lengths(healthy_image) -> None:
    client = TestClient(create_app())
    files = [
        ("files", ("a.png", _png_bytes(healthy_image), "image/png")),
        ("files", ("b.png", _png_bytes(healthy_image), "image/png")),
    ]
    data = {"plant_id": ["p1"], "timestamp": ["2026-03-01"]}  # one tag for two files
    response = client.post("/trend", files=files, data=data)
    assert response.status_code == 400
