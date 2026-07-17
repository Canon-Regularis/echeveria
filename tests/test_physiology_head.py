"""The standalone physiology head: surface the RGB physiology proxies without drought staging."""

from __future__ import annotations

import io
import json

import numpy as np
from fastapi.testclient import TestClient

from phytovision.api import create_app
from phytovision.cli import main
from phytovision.models.base import Head
from phytovision.models.drought.rule_based import physiology_basis, physiology_proxies
from phytovision.models.physiology.head import PhysiologyHead
from phytovision.pipeline import Pipeline
from phytovision.serving import attach_heads
from phytovision.types import PlantFeatures

_PROXY_KEYS = {"water_potential_proxy", "stomatal_conductance_proxy", "transpiration_proxy"}


def _png_bytes(image: np.ndarray) -> bytes:
    from PIL import Image as PILImage

    buffer = io.BytesIO()
    PILImage.fromarray((image * 255).astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_head_name_and_protocol() -> None:
    head = PhysiologyHead()
    assert head.name == "physiology"
    assert isinstance(head, Head)  # satisfies the runtime-checkable Head protocol


def test_head_output_matches_the_proxy_functions() -> None:
    values = {"colour.gcc_mean": 0.30, "geometry.solidity": 0.5, "colour.yellow_fraction": 0.3}
    output = PhysiologyHead().run(PlantFeatures.from_values(values))
    assert set(output) == _PROXY_KEYS | {"basis"}
    for key, value in physiology_proxies(values).items():
        assert output[key] == value  # the head reports the proxy functions verbatim
    assert output["basis"] == physiology_basis()


def test_attach_and_analyze_populates_the_head(healthy_image) -> None:
    report = attach_heads(Pipeline.default(), physiology=True).analyze(healthy_image)
    physiology = report.head_outputs["physiology"]
    assert isinstance(physiology, dict)
    assert set(physiology) == _PROXY_KEYS | {"basis"}
    assert all(0.0 <= physiology[key] <= 1.0 for key in _PROXY_KEYS)


def test_not_attached_by_default(healthy_image) -> None:
    report = Pipeline.default().analyze(healthy_image)
    assert "physiology" not in report.head_outputs


def test_cli_physiology_flag_surfaces_the_head(image_path, capsys) -> None:
    assert main(["analyze", str(image_path), "--physiology", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["head_outputs"]["physiology"]) == _PROXY_KEYS | {"basis"}


def test_api_physiology_true_exposes_head_and_disclaimer(healthy_image) -> None:
    client = TestClient(create_app())
    files = {"file": ("plant.png", _png_bytes(healthy_image), "image/png")}
    payload = client.post("/analyze", files=files, params={"physiology": "true"}).json()
    physiology = payload["head_outputs"]["physiology"]
    assert set(physiology) == _PROXY_KEYS | {"basis"}
    assert "disclaimer" in payload
    assert "physiology" in payload["disclaimer"]
