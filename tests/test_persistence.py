"""Type-tagged model persistence (F15): every model round-trips and carries provenance."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")

import joblib

from phytovision.exceptions import ConfigError
from phytovision.models.conformal import SplitConformalClassifier
from phytovision.models.persistence import (
    build_manifest,
    load_model,
    read_envelope,
    save_model,
)
from phytovision.models.stress.ensemble import EnsembleStressModel
from phytovision.models.stress.gradient_boosted import GradientBoostedStressModel
from phytovision.models.stress.heuristic import HeuristicStressModel
from phytovision.types import PlantFeatures

_KEYS = ["colour.gcc_mean", "colour.yellow_fraction", "texture.entropy"]


def _training_data(seed: int = 0, per_class: int = 60):
    rng = np.random.default_rng(seed)
    dicts: list[dict[str, float]] = []
    labels: list[int] = []
    for _ in range(per_class):
        dicts.append(
            {
                "colour.gcc_mean": float(rng.normal(0.40, 0.02)),
                "colour.yellow_fraction": float(abs(rng.normal(0.03, 0.02))),
                "texture.entropy": float(rng.normal(2.5, 0.3)),
            }
        )
        labels.append(0)
    for _ in range(per_class):
        dicts.append(
            {
                "colour.gcc_mean": float(rng.normal(0.30, 0.02)),
                "colour.yellow_fraction": float(abs(rng.normal(0.40, 0.05))),
                "texture.entropy": float(rng.normal(4.5, 0.3)),
            }
        )
        labels.append(1)
    return dicts, labels


def _features(**values: float) -> PlantFeatures:
    return PlantFeatures(values=dict(values), region_count=1)


_STRESSED = _features(
    **{"colour.gcc_mean": 0.30, "colour.yellow_fraction": 0.45, "texture.entropy": 4.6}
)


def _fitted_gbm() -> GradientBoostedStressModel:
    dicts, labels = _training_data()
    return GradientBoostedStressModel(feature_keys=_KEYS).fit(dicts, labels)


def test_gradient_boosted_round_trips(tmp_path) -> None:
    model = _fitted_gbm()
    before = model.predict(_STRESSED).score
    path = tmp_path / "gbm.joblib"
    save_model(model, path)
    reloaded = load_model(path)
    assert isinstance(reloaded, GradientBoostedStressModel)
    assert reloaded.predict(_STRESSED).score == before


def test_heuristic_round_trips(tmp_path) -> None:
    model = HeuristicStressModel(bias=0.2, healthy_threshold=0.25, stressed_threshold=0.7)
    path = tmp_path / "heuristic.joblib"
    save_model(model, path)
    reloaded = load_model(path)
    assert isinstance(reloaded, HeuristicStressModel)
    assert reloaded.bias == 0.2
    assert reloaded.healthy_threshold == 0.25
    assert reloaded.predict(_STRESSED).score == model.predict(_STRESSED).score


def test_ensemble_round_trips_with_trained_member(tmp_path) -> None:
    ensemble = EnsembleStressModel([HeuristicStressModel(), _fitted_gbm()], weights=[1.0, 2.0])
    before = ensemble.predict(_STRESSED).score
    path = tmp_path / "ensemble.joblib"
    save_model(ensemble, path)
    reloaded = load_model(path)
    assert isinstance(reloaded, EnsembleStressModel)
    assert reloaded.weights == ensemble.weights
    assert reloaded.predict(_STRESSED).score == pytest.approx(before)


def test_conformal_round_trips(tmp_path) -> None:
    model = _fitted_gbm()
    calib = [_STRESSED, _features(**{"colour.gcc_mean": 0.41, "colour.yellow_fraction": 0.01})]
    clf = SplitConformalClassifier(model, alpha=0.2).calibrate(calib, [1, 0])
    path = tmp_path / "conformal.joblib"
    clf.save(path)
    reloaded = SplitConformalClassifier.load(path)
    assert reloaded.qhat == clf.qhat
    assert reloaded.alpha == clf.alpha
    assert reloaded.predict_set(_STRESSED).labels == clf.predict_set(_STRESSED).labels


def test_manifest_records_provenance() -> None:
    manifest = build_manifest(feature_keys=_KEYS, sources=["a", "b", "a", None], seed=7)
    assert manifest["feature_keys"] == _KEYS
    assert manifest["sources"] == ["a", "b"]  # sorted, unique, None dropped
    assert manifest["seed"] == 7
    assert "created_at" in manifest
    assert "numpy" in manifest["versions"]


def test_manifest_is_saved_and_read_back(tmp_path) -> None:
    model = _fitted_gbm()
    path = tmp_path / "gbm.joblib"
    save_model(model, path, manifest=build_manifest(feature_keys=_KEYS, seed=3))
    envelope = read_envelope(path)
    assert envelope["model_type"] == "gradient-boosted"
    assert envelope["manifest"]["seed"] == 3


def test_legacy_gradient_boosted_file_still_loads(tmp_path) -> None:
    model = _fitted_gbm()
    path = tmp_path / "legacy.joblib"
    joblib.dump(model.state(), path)  # the pre-envelope raw dict shape
    reloaded = load_model(path)
    assert isinstance(reloaded, GradientBoostedStressModel)
    assert reloaded.predict(_STRESSED).score == model.predict(_STRESSED).score


def test_saving_an_unpersistable_model_raises(tmp_path) -> None:
    from phytovision.models.base import StressModel
    from phytovision.types import StressAssessment

    class _NoState(StressModel):  # no state()/MODEL_TYPE, so it is not persistable
        name = "nostate"

        def predict(self, features: PlantFeatures) -> StressAssessment:
            return StressAssessment(0.5, 0.5, "x", self.name)

    with pytest.raises(ConfigError, match="cannot be saved"):
        save_model(_NoState(), tmp_path / "x.joblib")


def test_unrecognized_file_raises(tmp_path) -> None:
    path = tmp_path / "junk.joblib"
    joblib.dump({"nonsense": 1}, path)
    with pytest.raises(ConfigError, match="unrecognized model file"):
        read_envelope(path)


def test_corrupt_file_raises_clean_error(tmp_path) -> None:
    path = tmp_path / "corrupt.joblib"
    path.write_bytes(b"this is not a joblib file at all")
    with pytest.raises(ConfigError, match="could not read model file"):
        read_envelope(path)


def test_malformed_but_loadable_envelope_raises_clean_error(tmp_path) -> None:
    # A loadable envelope whose state is not a mapping would crash dict(...) with a raw TypeError;
    # it must surface as a ConfigError like every other bad-file path.
    path = tmp_path / "malformed.joblib"
    joblib.dump({"model_type": "heuristic", "state": 5}, path)
    with pytest.raises(ConfigError, match="malformed model file"):
        read_envelope(path)
