"""Shared serving plumbing: config parsing and env-driven engine resolution used by the API and
the dashboard."""

from __future__ import annotations

import json

import pytest

from phytovision.exceptions import ConfigError, PhytoVisionError
from phytovision.models.conformal import SplitConformalClassifier
from phytovision.models.persistence import save_model
from phytovision.models.stress.heuristic import HeuristicStressModel
from phytovision.pipeline import Pipeline
from phytovision.serving import engine_from_env, read_config
from phytovision.types import PlantFeatures


def test_read_config_parses_toml_and_json(tmp_path) -> None:
    toml_file = tmp_path / "c.toml"
    toml_file.write_text('model = "heuristic"\n')
    json_file = tmp_path / "c.json"
    json_file.write_text(json.dumps({"model": "heuristic"}))
    assert read_config(toml_file) == {"model": "heuristic"}
    assert read_config(json_file) == {"model": "heuristic"}


def test_read_config_rejects_unknown_suffix(tmp_path) -> None:
    bad = tmp_path / "c.yaml"
    bad.write_text("model: heuristic\n")
    with pytest.raises(ConfigError, match="must be .toml or .json"):
        read_config(bad)


def test_read_config_rejects_non_mapping_top_level(tmp_path) -> None:
    listy = tmp_path / "c.json"
    listy.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ConfigError, match="table/object"):
        read_config(listy)


def test_engine_from_env_prefers_an_explicit_pipeline() -> None:
    pipeline = Pipeline.default()
    engine, conformal = engine_from_env(pipeline)
    assert engine is pipeline
    assert conformal is None


def test_engine_from_env_honors_a_non_default_config(tmp_path, monkeypatch) -> None:
    # lab-chroma is not the default segmenter, so the resolved pipeline must reflect the config
    # rather than silently falling back to Pipeline.default().
    config = tmp_path / "c.json"
    config.write_text(json.dumps({"segmenter": "lab-chroma"}))
    monkeypatch.setenv("PHYTOVISION_CONFIG", str(config))
    monkeypatch.delenv("PHYTOVISION_MODEL_PATH", raising=False)
    engine, conformal = engine_from_env()
    assert type(engine.segmenter).__name__ == "LabChromaSegmenter"
    assert conformal is None


def test_engine_from_env_propagates_a_bad_config(tmp_path, monkeypatch) -> None:
    # An unknown component proves PHYTOVISION_CONFIG is actually read and parsed, not ignored.
    config = tmp_path / "c.json"
    config.write_text(json.dumps({"model": "does-not-exist"}))
    monkeypatch.setenv("PHYTOVISION_CONFIG", str(config))
    monkeypatch.delenv("PHYTOVISION_MODEL_PATH", raising=False)
    with pytest.raises(PhytoVisionError):
        engine_from_env()


def test_engine_from_env_loads_a_saved_model(tmp_path, monkeypatch, healthy_image) -> None:
    model_path = tmp_path / "m.joblib"
    save_model(HeuristicStressModel(), model_path)
    monkeypatch.delenv("PHYTOVISION_CONFIG", raising=False)
    monkeypatch.setenv("PHYTOVISION_MODEL_PATH", str(model_path))
    engine, conformal = engine_from_env()
    assert conformal is None
    # The loaded model is wired in and the pipeline still runs.
    assert engine.analyze(healthy_image).stress.model_name == HeuristicStressModel().name


def test_engine_from_env_unwraps_a_calibrated_conformal_model(tmp_path, monkeypatch) -> None:
    # A calibrated file must return a non-None conformal wrapper AND run the unwrapped inner model
    # in the pipeline. Dropping either half silently loses conformal coverage.
    calibration = [
        PlantFeatures(
            values={"colour.gcc_mean": 0.5, "colour.yellow_fraction": 0.1}, region_count=1
        ),
        PlantFeatures(
            values={"colour.gcc_mean": 0.2, "colour.yellow_fraction": 0.6}, region_count=1
        ),
    ]
    wrapper = SplitConformalClassifier(HeuristicStressModel(), alpha=0.1).calibrate(
        calibration, [0, 1]
    )
    path = tmp_path / "conformal.joblib"
    wrapper.save(path)
    monkeypatch.delenv("PHYTOVISION_CONFIG", raising=False)
    monkeypatch.setenv("PHYTOVISION_MODEL_PATH", str(path))

    engine, conformal = engine_from_env()
    assert conformal is not None
    assert conformal.alpha == 0.1
    assert not isinstance(engine.model, SplitConformalClassifier)
    assert isinstance(engine.model, HeuristicStressModel)
