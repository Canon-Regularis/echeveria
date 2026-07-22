"""The typed pipeline-config schema: validation, defaults, and a diffable resolved config."""

from __future__ import annotations

import pytest

from phytovision.config_schema import ComponentSpec, PipelineConfig
from phytovision.exceptions import ConfigError


def test_empty_config_fills_every_default() -> None:
    config = PipelineConfig.from_mapping({})
    assert config.preprocessor.name == "resize-normalize"
    assert config.segmenter.name == "exg-otsu"
    assert config.model.name == "heuristic"
    assert config.explainer.name == "feature-contribution"
    assert len(config.feature_extractors) == 4


def test_unknown_top_level_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown config key"):
        PipelineConfig.from_mapping({"modell": "heuristic"})  # a typo must fail loudly


def test_string_and_mapping_specs_normalize() -> None:
    config = PipelineConfig.from_mapping({"model": "ensemble", "segmenter": {"name": "lab-chroma"}})
    assert config.model == ComponentSpec("ensemble", {})
    assert config.segmenter.name == "lab-chroma"


def test_params_only_spec_keeps_the_default_component() -> None:
    config = PipelineConfig.from_mapping({"preprocessor": {"params": {"max_size": 256}}})
    assert config.preprocessor.name == "resize-normalize"  # default filled
    assert config.preprocessor.params == {"max_size": 256}


def test_non_list_feature_extractors_is_rejected() -> None:
    with pytest.raises(ConfigError, match="feature_extractors"):
        PipelineConfig.from_mapping({"feature_extractors": "geometry"})


def test_duplicate_feature_extractors_are_rejected() -> None:
    # A repeated extractor produces the same namespace twice and would crash every analyze(); it is
    # rejected here so the config fails at build time, not on the first image.
    with pytest.raises(ConfigError, match="duplicate feature_extractors"):
        PipelineConfig.from_mapping({"feature_extractors": ["geometry", "geometry"]})


def test_malformed_specs_are_rejected() -> None:
    with pytest.raises(ConfigError, match="string 'name'"):
        PipelineConfig.from_mapping({"model": {"name": 5}})
    with pytest.raises(ConfigError, match="invalid component spec"):
        PipelineConfig.from_mapping({"model": 5})


def test_as_dict_is_canonical_and_round_trips() -> None:
    config = PipelineConfig.from_mapping(
        {
            "model": {"name": "ensemble", "params": {"members": ["heuristic"]}},
            "feature_extractors": ["geometry", "skeleton"],
        }
    )
    resolved = config.as_dict()
    assert resolved["model"] == {"name": "ensemble", "params": {"members": ["heuristic"]}}
    assert resolved["feature_extractors"] == [{"name": "geometry"}, {"name": "skeleton"}]
    # Feeding the resolved dict back produces the identical config, so it is a stable diff target.
    assert PipelineConfig.from_mapping(resolved) == config


def test_component_spec_as_dict_omits_empty_params() -> None:
    assert ComponentSpec("x").as_dict() == {"name": "x"}
    assert ComponentSpec("x", {"a": 1}).as_dict() == {"name": "x", "params": {"a": 1}}
