"""Shared plumbing for the serving surfaces (HTTP API and the Streamlit dashboard).

Both surfaces resolve the analysed pipeline the same way: ``Pipeline.default()`` unless
``PHYTOVISION_CONFIG`` or ``PHYTOVISION_MODEL_PATH`` is set. Keeping that in one place means the API
and the dashboard cannot drift apart. This module needs only the base dependencies.
"""

from __future__ import annotations

import os
from pathlib import Path

from phytovision.config import read_config
from phytovision.models.conformal import SplitConformalClassifier
from phytovision.models.disease.head import DiseaseHead
from phytovision.models.drought.head import DroughtStageHead
from phytovision.models.persistence import load_saved
from phytovision.models.physiology.head import PhysiologyHead
from phytovision.pipeline import Pipeline
from phytovision.registries import DISEASE_MODELS, DROUGHT_STAGE_MODELS

# The environment variables that point a served pipeline at a config and/or a saved model. The
# ``serve`` and ``dashboard`` launchers write them; ``engine_from_env`` reads them.
CONFIG_ENV = "PHYTOVISION_CONFIG"
MODEL_PATH_ENV = "PHYTOVISION_MODEL_PATH"

__all__ = [
    "CONFIG_ENV",
    "MODEL_PATH_ENV",
    "attach_heads",
    "engine_from_env",
    "read_config",
    "serving_env",
    "validate_serving_selection",
]


def engine_from_env(
    pipeline: Pipeline | None = None, conformal: SplitConformalClassifier | None = None
) -> tuple[Pipeline, SplitConformalClassifier | None]:
    """Resolve the pipeline (and optional conformal wrapper) to serve.

    An explicit ``pipeline`` wins. Otherwise read the config and model-path environment variables,
    as ``serve`` and ``dashboard`` set them.
    """
    if pipeline is not None:
        return pipeline, conformal

    config = os.environ.get(CONFIG_ENV)
    engine = Pipeline.from_config(read_config(config)) if config else Pipeline.default()

    model_path = os.environ.get(MODEL_PATH_ENV)
    if model_path:
        loaded = load_saved(model_path)
        if isinstance(loaded, SplitConformalClassifier):
            return engine.with_model(loaded.model), loaded
        return engine.with_model(loaded), conformal
    return engine, conformal


def validate_serving_selection(config: str | None, model_path: str | None) -> None:
    """Read the config and model paths a launcher was given, so a bad path fails before a server
    starts, rather than as a traceback from inside the launched process.

    :raises OSError, ImportError, PhytoVisionError: if a given path cannot be read or loaded.
    """
    if config:
        read_config(config)
    if model_path:
        load_saved(model_path)


def serving_env(config: str | None, model_path: str | None) -> dict[str, str]:
    """The environment variables that point a served pipeline at a config and/or a saved model."""
    env: dict[str, str] = {}
    if config:
        env[CONFIG_ENV] = str(Path(config))
    if model_path:
        env[MODEL_PATH_ENV] = str(Path(model_path))
    return env


def attach_heads(
    pipeline: Pipeline,
    *,
    disease: bool = False,
    drought_stage: bool = False,
    physiology: bool = False,
) -> Pipeline:
    """Return a copy of ``pipeline`` with the requested optional heads attached.

    One place for the CLI, API, and dashboard to opt into the secondary heads, so they cannot wire
    them up three different ways. Every shipped head is an unvalidated placeholder or prior, not a
    diagnostic: the physiology head reports crude RGB proxies, not measured physiology.
    """
    if disease:
        pipeline = pipeline.add_head(DiseaseHead(DISEASE_MODELS.create("heuristic")))
    if drought_stage:
        pipeline = pipeline.add_head(DroughtStageHead(DROUGHT_STAGE_MODELS.create("rule-based")))
    if physiology:
        pipeline = pipeline.add_head(PhysiologyHead())
    return pipeline
