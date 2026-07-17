"""The dry-down simulator: determinism, event timing, censoring, and manifest round-trips."""

from __future__ import annotations

import csv

import numpy as np

from phytovision.cli import main
from phytovision.datasets.manifest import CsvManifestLoader
from phytovision.models.base import STRESSED_THRESHOLD, bucket_label
from phytovision.models.stress.heuristic import HeuristicStressModel
from phytovision.simulation import (
    SYNTHETIC_SOURCE,
    DryDownParams,
    cohort_history,
    load_history,
    simulate_cohort,
    simulate_plant,
    write_events,
    write_manifest,
)
from phytovision.simulation.drydown import feature_keys
from phytovision.types import PlantFeatures


def _plant(rate: float, **overrides: object) -> object:
    params = DryDownParams(process_noise=0.0, observation_noise=0.0, feature_noise=0.0, **overrides)
    return simulate_plant("p", params, np.random.default_rng(0), decline_rate=rate)


def test_same_seed_reproduces_the_cohort_exactly() -> None:
    first = simulate_cohort(5, DryDownParams(n_steps=12), seed=7)
    second = simulate_cohort(5, DryDownParams(n_steps=12), seed=7)
    assert [p.plant_id for p in first.series] == [p.plant_id for p in second.series]
    for a, b in zip(first.series, second.series, strict=True):
        assert a.latent == b.latent
        assert [o.stress_score for o in a.observations] == [o.stress_score for o in b.observations]


def test_a_faster_decline_wilts_earlier() -> None:
    slow = _plant(0.05, n_steps=40)
    fast = _plant(0.30, n_steps=40)
    assert not slow.censored and not fast.censored
    assert fast.duration < slow.duration


def test_a_slow_decline_over_a_short_window_is_censored() -> None:
    plant = _plant(0.01, n_steps=6)
    assert plant.censored
    assert plant.event_step is None
    assert plant.duration == 5  # the last observed step
    assert max(plant.latent) < STRESSED_THRESHOLD


def test_the_event_fires_at_the_stressed_cut() -> None:
    plant = _plant(0.15, n_steps=40)
    assert plant.event_step is not None
    assert plant.latent[plant.event_step] >= STRESSED_THRESHOLD
    assert plant.latent[plant.event_step - 1] < STRESSED_THRESHOLD  # the first crossing


def test_synthetic_features_use_real_namespaces_and_are_finite() -> None:
    plant = _plant(0.12, n_steps=10)
    for observation in plant.observations:
        assert set(observation.features) == set(feature_keys())
        assert all(np.isfinite(v) for v in observation.features.values())
        # PlantFeatures enforces finiteness, so this construction must not raise.
        PlantFeatures.from_values(observation.features)


def test_the_heuristic_score_rises_as_the_plant_dries() -> None:
    # The synthetic features are built to drive a rising stress score, so a model reading them
    # agrees with the latent state. This keeps the feature vectors and the score consistent.
    plant = _plant(0.15, n_steps=30)
    model = HeuristicStressModel()
    early = model.predict(PlantFeatures.from_values(plant.observations[2].features)).score
    late = model.predict(PlantFeatures.from_values(plant.observations[-1].features)).score
    assert late > early


def test_cohort_history_groups_observations_by_plant() -> None:
    cohort = simulate_cohort(4, DryDownParams(n_steps=8), seed=1)
    history = cohort_history(cohort)
    assert len(history.plant_ids) == 4
    assert len(history) == 4 * 8
    for plant_id in history.plant_ids:
        assert len(history.series_for(plant_id)) == 8


def test_manifest_round_trips_through_the_csv_loader(tmp_path) -> None:
    cohort = simulate_cohort(3, DryDownParams(n_steps=6), seed=2)
    manifest = write_manifest(cohort, tmp_path / "cohort.csv")

    loader = CsvManifestLoader(manifest)
    samples = list(loader)
    assert len(samples) == 3 * 6
    assert {s.source for s in samples} == {SYNTHETIC_SOURCE}
    assert all(s.plant_id and s.timestamp and s.target is not None for s in samples)
    assert {s.label for s in samples} <= {"healthy", "mild", "stressed"}


def test_load_history_rebuilds_scores_and_features_without_images(tmp_path) -> None:
    cohort = simulate_cohort(3, DryDownParams(n_steps=7), seed=3)
    manifest = write_manifest(cohort, tmp_path / "cohort.csv")

    history = load_history(manifest)
    assert len(history.plant_ids) == 3
    rebuilt = history.series_for("plant_000")
    original = cohort.series[0].observations
    assert len(rebuilt) == len(original)
    for got, want in zip(rebuilt, original, strict=True):
        assert got.stress_score == round(want.stress_score, 6)
        assert set(got.features) == set(feature_keys())


def test_manifest_label_matches_the_scored_bucket(tmp_path) -> None:
    cohort = simulate_cohort(2, DryDownParams(n_steps=5), seed=4)
    manifest = write_manifest(cohort, tmp_path / "cohort.csv")
    rows = list(csv.DictReader(manifest.open(encoding="utf-8-sig")))
    for row in rows:
        assert row["label"] == bucket_label(float(row["stress_score"]))


def test_events_table_records_duration_and_censoring(tmp_path) -> None:
    cohort = simulate_cohort(4, DryDownParams(n_steps=30, base_decline_rate=0.15), seed=5)
    events = write_events(cohort, tmp_path / "events.csv")
    rows = list(csv.DictReader(events.open()))
    assert len(rows) == 4
    for row, plant in zip(rows, cohort.series, strict=True):
        assert row["plant_id"] == plant.plant_id
        assert int(row["duration"]) == plant.duration
        assert int(row["event_observed"]) == int(not plant.censored)


def test_cli_simulate_writes_a_manifest_and_events_table(tmp_path, capsys) -> None:
    out = tmp_path / "cohort.csv"
    rc = main(["simulate", "--out", str(out), "--plants", "6", "--steps", "10", "--seed", "0"])
    assert rc == 0
    events = tmp_path / "cohort_events.csv"
    assert out.exists() and events.exists()
    assert "synthetic" in capsys.readouterr().out.lower()

    rows = list(csv.DictReader(out.open(encoding="utf-8-sig")))
    assert len(rows) == 6 * 10
    assert {row["source"] for row in rows} == {SYNTHETIC_SOURCE}
    assert len(list(csv.DictReader(events.open()))) == 6


def test_cli_simulate_is_reproducible(tmp_path) -> None:
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    main(["simulate", "--out", str(first), "--plants", "4", "--steps", "8", "--seed", "3"])
    main(["simulate", "--out", str(second), "--plants", "4", "--steps", "8", "--seed", "3"])
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_cli_simulate_rejects_an_empty_cohort(tmp_path, capsys) -> None:
    rc = main(["simulate", "--out", str(tmp_path / "x.csv"), "--plants", "0"])
    assert rc == 2
    assert capsys.readouterr().err.startswith("error:")
