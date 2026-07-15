"""Shared plumbing for the serving surfaces (HTTP API and the Streamlit dashboard).

Both surfaces resolve the analysed pipeline the same way: ``Pipeline.default()`` unless
``PHYTOVISION_CONFIG`` or ``PHYTOVISION_MODEL_PATH`` is set. Keeping that in one place means the API
and the dashboard cannot drift apart. This module needs only the base dependencies.
"""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path

from phytovision.exceptions import ConfigError
from phytovision.models.conformal import SplitConformalClassifier
from phytovision.models.disease.head import DiseaseHead
from phytovision.models.drought.head import DroughtStageHead
from phytovision.models.persistence import load_saved
from phytovision.pipeline import Pipeline
from phytovision.registries import DISEASE_MODELS, DROUGHT_STAGE_MODELS


def read_config(path: str | os.PathLike[str]) -> dict[str, object]:
    """Parse a pipeline config file (.toml or .json) into a plain dict."""
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    suffix = file.suffix.lower()
    if suffix == ".toml":
        data = tomllib.loads(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ConfigError(f"config must be .toml or .json: {file}")
    if not isinstance(data, dict):
        raise ConfigError(f"config must be a table/object at the top level: {file}")
    return data


def engine_from_env(
    pipeline: Pipeline | None = None, conformal: SplitConformalClassifier | None = None
) -> tuple[Pipeline, SplitConformalClassifier | None]:
    """Resolve the pipeline (and optional conformal wrapper) to serve.

    An explicit ``pipeline`` wins. Otherwise read ``PHYTOVISION_CONFIG`` and
    ``PHYTOVISION_MODEL_PATH`` from the environment, as ``serve`` and ``dashboard`` set them.
    """
    if pipeline is not None:
        return pipeline, conformal

    config = os.environ.get("PHYTOVISION_CONFIG")
    engine = Pipeline.from_config(read_config(config)) if config else Pipeline.default()

    model_path = os.environ.get("PHYTOVISION_MODEL_PATH")
    if model_path:
        loaded = load_saved(model_path)
        if isinstance(loaded, SplitConformalClassifier):
            return engine.with_model(loaded.model), loaded
        return engine.with_model(loaded), conformal
    return engine, conformal


def attach_heads(
    pipeline: Pipeline, *, disease: bool = False, drought_stage: bool = False
) -> Pipeline:
    """Return a copy of ``pipeline`` with the requested optional heads attached.

    One place for the CLI, API, and dashboard to opt into the secondary heads, so they cannot wire
    them up three different ways. Both shipped heads are unvalidated placeholders / priors, not
    diagnostics.
    """
    if disease:
        pipeline = pipeline.add_head(DiseaseHead(DISEASE_MODELS.create("heuristic")))
    if drought_stage:
        pipeline = pipeline.add_head(DroughtStageHead(DROUGHT_STAGE_MODELS.create("rule-based")))
    return pipeline
