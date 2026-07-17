"""Registry behaviour and config/name-driven pipeline construction."""

from __future__ import annotations

import pytest

from phytovision.exceptions import ConfigError
from phytovision.pipeline import Pipeline
from phytovision.preprocessing.basic import ResizeNormalizePreprocessor
from phytovision.registries import SEGMENTERS, STRESS_MODELS
from phytovision.registry import Registry


def test_registry_register_create_and_list() -> None:
    reg: Registry[object] = Registry("thing")

    @reg.register("a")
    class A:
        pass

    assert "a" in reg
    assert reg.names() == ["a"]
    assert isinstance(reg.create("a"), A)


def test_registry_rejects_duplicate_name() -> None:
    reg: Registry[object] = Registry("thing")
    reg.register("a")(object)
    with pytest.raises(ValueError, match="already registered"):
        reg.register("a")(object)


def test_registry_unknown_name_lists_available() -> None:
    reg: Registry[object] = Registry("thing")
    reg.register("known")(object)
    with pytest.raises(KeyError, match="known"):
        reg.create("missing")


def test_builtins_are_registered() -> None:
    assert {"heuristic", "gradient-boosted"} <= set(STRESS_MODELS.names())
    assert "exg-otsu" in SEGMENTERS


def test_default_pipeline_uses_registered_builtins() -> None:
    assert Pipeline.default().model.name == "heuristic-v1"


def test_from_names_runs_end_to_end(healthy_image) -> None:
    pipeline = Pipeline.from_names(segmenter="exg-otsu", model="heuristic")
    assert pipeline.analyze(healthy_image).stress.label in {"healthy", "mild", "stressed"}


def test_from_config_passes_params() -> None:
    pipeline = Pipeline.from_config(
        {"preprocessor": {"name": "resize-normalize", "params": {"max_size": 256}}}
    )
    assert isinstance(pipeline.preprocessor, ResizeNormalizePreprocessor)
    assert pipeline.preprocessor.max_size == 256


def test_from_config_unknown_component_raises_configerror() -> None:
    with pytest.raises(ConfigError):
        Pipeline.from_config({"model": "does-not-exist"})


def test_unknown_component_error_message_is_unquoted() -> None:
    # The wrapped KeyError message must read plainly, without the quotes KeyError.__str__ adds.
    with pytest.raises(ConfigError, match=r"^unknown ") as info:
        Pipeline.from_config({"model": "does-not-exist"})
    assert not str(info.value).startswith('"')


def test_from_config_rejects_non_list_feature_extractors() -> None:
    with pytest.raises(ConfigError, match="feature_extractors"):
        Pipeline.from_config({"feature_extractors": "geometry"})


def test_from_config_rejects_an_unknown_top_level_key() -> None:
    # A mistyped slot name is validated by the schema, so it fails loudly instead of silently.
    with pytest.raises(ConfigError, match="unknown config key"):
        Pipeline.from_config({"modell": "heuristic"})


def test_unbuildable_component_raises_configerror_not_typeerror() -> None:
    # gradient-boosted requires feature_keys, so building it by name must surface as ConfigError
    # (a PhytoVisionError), not a raw TypeError.
    with pytest.raises(ConfigError):
        Pipeline.from_names(model="gradient-boosted")


def test_from_config_params_only_keeps_default_component() -> None:
    # A spec with only params (no name) overrides the default component's parameters.
    pipeline = Pipeline.from_config({"preprocessor": {"params": {"max_size": 256}}})
    assert isinstance(pipeline.preprocessor, ResizeNormalizePreprocessor)
    assert pipeline.preprocessor.max_size == 256
