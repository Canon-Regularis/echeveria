"""The FastAPI /analyze endpoint (F10). Skipped if fastapi/httpx are absent."""

from __future__ import annotations

import io

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from phytovision.api import create_app  # noqa: E402


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


def test_overlay_endpoint_returns_png(healthy_image) -> None:
    client = TestClient(create_app())
    files = {"file": ("plant.png", _png_bytes(healthy_image), "image/png")}
    response = client.post("/overlay", files=files)
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
