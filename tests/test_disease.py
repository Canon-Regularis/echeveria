"""Disease head (F28): a placeholder DiseaseModel wrapped as a post-model Head."""

from __future__ import annotations

import pytest

from phytovision.models.base import Head
from phytovision.models.disease.head import DiseaseHead
from phytovision.models.disease.heuristic import HeuristicDiseaseModel
from phytovision.pipeline import Pipeline
from phytovision.registries import DISEASE_MODELS
from phytovision.types import PlantFeatures


def _predict(brown: float, glcm_contrast: float) -> dict[str, float]:
    features = PlantFeatures(
        values={"colour.brown_fraction": brown, "texture.glcm_contrast": glcm_contrast},
        region_count=1,
    )
    return HeuristicDiseaseModel().predict(features)


def test_heuristic_disease_returns_a_normalized_two_class_distribution() -> None:
    healthy = _predict(brown=0.0, glcm_contrast=0.0)
    spotty = _predict(brown=0.6, glcm_contrast=4.0)
    assert set(healthy) == set(spotty) == {"healthy", "lesion-like"}
    assert healthy["healthy"] + healthy["lesion-like"] == pytest.approx(1.0)
    assert spotty["healthy"] + spotty["lesion-like"] == pytest.approx(1.0)
    assert spotty["lesion-like"] > healthy["lesion-like"]


def test_browning_alone_raises_lesion_risk() -> None:
    # Hold contrast fixed so this isolates the brown term; a dropped brown weight would fail here.
    assert _predict(brown=0.6, glcm_contrast=0.0)["lesion-like"] > _predict(0.2, 0.0)["lesion-like"]


def test_surface_contrast_alone_raises_lesion_risk() -> None:
    # Hold browning fixed so this isolates the speckle/contrast term.
    assert _predict(brown=0.0, glcm_contrast=4.0)["lesion-like"] > _predict(0.0, 1.0)["lesion-like"]


def test_risk_saturates_at_one() -> None:
    # brown=1 and contrast>=5 push 0.6*1 + 0.4*1 to 1.0, exercising the _clip01 ceiling.
    saturated = _predict(brown=1.0, glcm_contrast=6.0)
    assert saturated["lesion-like"] == pytest.approx(1.0)
    assert saturated["healthy"] == pytest.approx(0.0)


def test_disease_head_satisfies_the_head_protocol() -> None:
    head = DiseaseHead(HeuristicDiseaseModel())
    assert isinstance(head, Head)
    assert head.name == "disease"
    result = head.run(PlantFeatures(values={"colour.brown_fraction": 0.3}, region_count=1))
    assert "lesion-like" in result


def test_disease_registered_and_attachable(healthy_image) -> None:
    assert "heuristic" in DISEASE_MODELS.names()
    pipeline = Pipeline.default().add_head(DiseaseHead(DISEASE_MODELS.create("heuristic")))
    report = pipeline.analyze(healthy_image)
    assert "disease" in report.head_outputs
    assert set(report.head_outputs["disease"]) == {"healthy", "lesion-like"}
