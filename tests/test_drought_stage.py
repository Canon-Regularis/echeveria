"""Drought-stage head: a rule-based ordinal classifier over the drought progression.

The rules are literature-motivated priors (the Sedum drought study), not a fitted model."""

from __future__ import annotations

from phytovision.dashboard import drought_markers
from phytovision.models.base import Head
from phytovision.models.drought.head import DroughtStageHead
from phytovision.models.drought.rule_based import RuleBasedDroughtStage
from phytovision.pipeline import Pipeline
from phytovision.registries import DROUGHT_STAGE_MODELS
from phytovision.serving import attach_heads
from phytovision.types import PlantFeatures


def _stage(**values: float) -> dict[str, object]:
    return RuleBasedDroughtStage().stage(PlantFeatures(values=dict(values), region_count=1))


def test_well_watered_when_no_markers() -> None:
    result = _stage(**{"colour.gcc_mean": 0.40, "geometry.solidity": 0.95})
    assert result["stage"] == "well-watered"


def test_early_stress_is_pigment_before_turgor_loss() -> None:
    # Greenness down and some yellowing, but shape/turgor intact and no browning.
    result = _stage(
        **{"colour.gcc_mean": 0.34, "colour.yellow_fraction": 0.10, "geometry.solidity": 0.95}
    )
    assert result["stage"] == "early-stress"


def test_moderate_when_pigment_and_turgor_both_decline() -> None:
    result = _stage(
        **{
            "colour.gcc_mean": 0.28,
            "colour.yellow_fraction": 0.30,
            "geometry.solidity": 0.60,
            "morphology.concavity": 0.20,
        }
    )
    assert result["stage"] == "moderate"


def test_moderate_from_pigment_alone() -> None:
    # Pigment high, turgor and necrosis low: the pigment-only escalation must still reach moderate.
    assert (
        _stage(**{"colour.gcc_mean": 0.28, "colour.yellow_fraction": 0.30})["stage"] == "moderate"
    )


def test_severe_when_necrosis_is_high() -> None:
    assert _stage(**{"colour.brown_fraction": 0.30})["stage"] == "severe"


def test_severe_from_turgor_collapse() -> None:
    # Shape collapse with no browning: the turgor>=0.60 severe path.
    result = _stage(**{"geometry.solidity": 0.40, "morphology.concavity": 0.50})
    assert result["stage"] == "severe"


def test_basis_names_the_actual_driver() -> None:
    # Moderate reached via browning alone: the basis must name browning, not pigment/turgor.
    result = _stage(**{"colour.brown_fraction": 0.15})
    assert result["stage"] == "moderate"
    assert result["basis"] == "browning"


def test_markers_are_bounded_and_reported() -> None:
    result = _stage(**{"colour.gcc_mean": 0.30, "colour.brown_fraction": 0.10})
    markers = result["markers"]
    assert set(markers) == {"pigment", "turgor_loss", "necrosis"}
    assert all(0.0 <= value <= 1.0 for value in markers.values())
    assert "basis" in result


def test_head_satisfies_the_protocol_and_is_registered() -> None:
    assert "rule-based" in DROUGHT_STAGE_MODELS.names()
    head = DroughtStageHead(DROUGHT_STAGE_MODELS.create("rule-based"))
    assert isinstance(head, Head)
    assert head.name == "drought_stage"
    result = head.run(PlantFeatures(values={"colour.gcc_mean": 0.4}, region_count=1))
    assert result["stage"] in {"well-watered", "early-stress", "moderate", "severe"}


def test_attach_and_run_on_a_pipeline(healthy_image, stressed_image) -> None:
    pipeline = attach_heads(Pipeline.default(), drought_stage=True)
    healthy = pipeline.analyze(healthy_image).head_outputs["drought_stage"]
    stressed = pipeline.analyze(stressed_image).head_outputs["drought_stage"]
    assert healthy["stage"] == "well-watered"
    assert stressed["stage"] != "well-watered"


def test_stage_output_carries_an_additive_physiology_block() -> None:
    result = _stage(**{"geometry.solidity": 0.45, "colour.yellow_fraction": 0.4})
    physiology = result["physiology"]
    assert set(physiology) == {
        "water_potential_proxy",
        "stomatal_conductance_proxy",
        "transpiration_proxy",
    }
    assert all(0.0 <= value <= 1.0 for value in physiology.values())
    assert result["physiology_basis"] and "not measurements" in result["physiology_basis"]
    # The existing keys are untouched, so the CLI/API/dashboard marker assertions still hold.
    assert set(result["markers"]) == {"pigment", "turgor_loss", "necrosis"}
    assert result["stage"] in {"well-watered", "early-stress", "moderate", "severe"}


def test_head_physiology_moves_in_the_documented_direction(healthy_image, stressed_image) -> None:
    pipeline = attach_heads(Pipeline.default(), drought_stage=True)
    healthy = pipeline.analyze(healthy_image).head_outputs["drought_stage"]["physiology"]
    stressed = pipeline.analyze(stressed_image).head_outputs["drought_stage"]["physiology"]
    assert stressed["water_potential_proxy"] > healthy["water_potential_proxy"]  # deficit rises
    assert stressed["stomatal_conductance_proxy"] < healthy["stomatal_conductance_proxy"]
    assert stressed["transpiration_proxy"] < healthy["transpiration_proxy"]


def test_dashboard_drought_markers_helper(healthy_image) -> None:
    report = attach_heads(Pipeline.default(), drought_stage=True).analyze(healthy_image)
    names, scores = drought_markers(report)
    assert set(names) == {"pigment", "turgor_loss", "necrosis"}
    assert all(0.0 <= value <= 1.0 for value in scores)
    # A report without the head yields nothing to plot.
    assert drought_markers(Pipeline.default().analyze(healthy_image)) == ([], [])
