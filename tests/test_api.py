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
