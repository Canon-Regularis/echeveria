"""Explanations cite the drought-physiology mechanism behind each feature."""

from __future__ import annotations

from phytovision.explainability._reasons import build_reasons
from phytovision.explainability.physiology import physiology_note
from phytovision.models.stress.heuristic import HeuristicStressModel
from phytovision.types import PlantFeatures


def test_physiology_note_maps_known_markers_and_ignores_others() -> None:
    assert physiology_note("colour.yellow_fraction") == "chlorophyll degradation"
    assert physiology_note("colour.red_fraction") == "anthocyanin accumulation"
    assert physiology_note("geometry.solidity") == "turgor loss / leaf deformation"
    assert physiology_note("geometry.area_px") is None  # unmapped keys carry no note


def test_reason_descriptions_are_grounded_in_physiology() -> None:
    model = HeuristicStressModel()
    features = PlantFeatures(
        values={"colour.yellow_fraction": 0.4, "colour.gcc_mean": 0.3}, region_count=1
    )
    reasons = build_reasons(model, features, model.contributions(features), top_k=6)

    by_feature = {reason.feature: reason.description for reason in reasons}
    assert "chlorophyll degradation" in by_feature["colour.yellow_fraction"]
    assert by_feature["colour.yellow_fraction"].startswith("yellowing raises the estimate")


def test_unmapped_feature_description_is_unchanged() -> None:
    # A feature with no physiology note keeps the plain template (geometry.area_px is unmapped).
    model = HeuristicStressModel()
    empty = PlantFeatures(values={}, region_count=1)
    reasons = build_reasons(model, empty, {"geometry.area_px": 1.0}, top_k=6)
    assert reasons[0].description == "geometry.area_px raises the estimate"


def test_physiology_proxy_notes_render_and_ground_the_basis() -> None:
    from phytovision.models.drought.rule_based import physiology_basis

    keys = (
        "physiology.water_potential_proxy",
        "physiology.stomatal_conductance_proxy",
        "physiology.transpiration_proxy",
    )
    notes = [physiology_note(key) for key in keys]
    assert all(note is not None and "proxy" in note for note in notes)
    assert physiology_note("physiology.unmapped_proxy") is None

    basis = physiology_basis()
    assert all(note in basis for note in notes if note is not None)  # notes travel with the values
    assert "not measurements" in basis
