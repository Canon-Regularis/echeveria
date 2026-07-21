"""A typed, validated schema over the bare pipeline-config dict.

``config.read_config`` parses a file into a plain dict; this layer turns that dict into a validated,
diffable ``PipelineConfig``. It catches a mistyped top-level key (a silent no-op in a bare dict, and
a real reproducibility footgun) and normalizes every slot into a ``ComponentSpec`` with its defaults
filled, so a resolved config can be logged or diffed. Component names are still resolved against the
registries at build time, so this validates the shape, not the vocabulary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

from phytovision.exceptions import ConfigError
from phytovision.registries import DEFAULTS

# The single-component slots (everything except the list-valued feature_extractors).
_SLOT_KEYS: tuple[str, ...] = (
    "preprocessor",
    "segmenter",
    "region_provider",
    "aggregator",
    "model",
    "explainer",
)
_KNOWN_KEYS = frozenset({*_SLOT_KEYS, "feature_extractors"})


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """A resolved component: a registered ``name`` plus its constructor ``params``."""

    name: str
    params: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """A canonical dict for this spec, omitting an empty params block."""
        if not self.params:
            return {"name": self.name}
        return {"name": self.name, "params": dict(self.params)}


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """A validated pipeline configuration with every slot resolved to a ``ComponentSpec``."""

    preprocessor: ComponentSpec
    segmenter: ComponentSpec
    region_provider: ComponentSpec
    feature_extractors: tuple[ComponentSpec, ...]
    aggregator: ComponentSpec
    model: ComponentSpec
    explainer: ComponentSpec

    @classmethod
    def from_mapping(cls, config: Mapping[str, object] | None = None) -> PipelineConfig:
        """Validate and normalize a config mapping, filling each slot's default.

        :raises ConfigError: on an unknown top-level key, a non-list ``feature_extractors``, or a
            malformed component spec.
        """
        cfg = dict(config or {})
        unknown = sorted(set(cfg) - _KNOWN_KEYS)
        if unknown:
            valid = ", ".join(sorted(_KNOWN_KEYS))
            raise ConfigError(f"unknown config key(s): {unknown}; valid keys: {valid}")

        # Every slot default in DEFAULTS is a component name (a str); feature_extractors is the only
        # list-valued default, handled separately below.
        slots: dict[str, ComponentSpec] = {}
        for key in _SLOT_KEYS:
            name = cast(str, DEFAULTS[key])
            slots[key] = _normalize_spec(cfg.get(key, name), name)

        extractor_specs = cfg.get("feature_extractors", DEFAULTS["feature_extractors"])
        if not isinstance(extractor_specs, Sequence) or isinstance(extractor_specs, str):
            raise ConfigError("`feature_extractors` must be a list of component specs")
        extractors = tuple(_normalize_spec(spec, None) for spec in extractor_specs)

        return cls(feature_extractors=extractors, **slots)

    def as_dict(self) -> dict[str, object]:
        """A canonical, JSON-serializable dict of the resolved config, for logging or diffing."""
        resolved: dict[str, object] = {key: getattr(self, key).as_dict() for key in _SLOT_KEYS}
        resolved["feature_extractors"] = [spec.as_dict() for spec in self.feature_extractors]
        return resolved


def _normalize_spec(spec: object, default_name: str | None) -> ComponentSpec:
    """Turn a name or ``{"name", "params"}`` spec into a ``ComponentSpec``.

    A mapping spec that omits ``name`` uses ``default_name``, so a config can override only the
    parameters of a slot's default component.
    """
    if isinstance(spec, str):
        return ComponentSpec(spec, {})
    if isinstance(spec, Mapping):
        unknown = set(spec) - {"name", "params"}
        if unknown:  # a forgotten params wrapper or a typo would otherwise be silently dropped
            raise ConfigError(f"unknown key(s) in component spec {spec!r}: {sorted(unknown)}")
        name = spec.get("name", default_name)
        if not isinstance(name, str):
            raise ConfigError(f"component spec needs a string 'name': {spec!r}")
        raw_params = spec.get("params", {})
        if not isinstance(raw_params, Mapping):
            raise ConfigError(f"component spec 'params' must be a mapping: {spec!r}")
        return ComponentSpec(name, dict(raw_params))
    raise ConfigError(f"invalid component spec: {spec!r}")
