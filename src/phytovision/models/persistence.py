"""Type-tagged persistence for stress models and their wrappers.

A persistable model has a stable ``MODEL_TYPE`` and a ``state`` / ``from_state`` pair. This wraps
that state in a joblib envelope with a manifest, and reloads it by dispatching on the type tag. One
format serves the heuristic, the gradient-boosted model, ensembles, and the conformal wrapper, so
every model round-trips the same way and carries its own provenance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from phytovision.exceptions import ConfigError
from phytovision.models.base import StressModel

if TYPE_CHECKING:
    from phytovision.models.conformal import SplitConformalClassifier

_ENVELOPE_VERSION = 1


@runtime_checkable
class Persistable(Protocol):
    """A model that can report a serializable state under a stable type tag."""

    MODEL_TYPE: ClassVar[str]

    def state(self) -> dict[str, object]: ...


def build_manifest(
    *,
    feature_keys: Sequence[str] | None = None,
    sources: Sequence[str | None] | None = None,
    seed: int | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """A provenance record: timestamp, library versions, and any training details supplied."""
    manifest: dict[str, object] = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "versions": _versions(),
    }
    if feature_keys is not None:
        manifest["feature_keys"] = list(feature_keys)
    if sources is not None:
        manifest["sources"] = sorted({s for s in sources if s})
    if seed is not None:
        manifest["seed"] = seed
    if extra:
        manifest.update(extra)
    return manifest


def write_envelope(
    model_type: str,
    state: Mapping[str, object],
    path: str | Path,
    manifest: Mapping[str, object] | None = None,
) -> None:
    """Write a type-tagged joblib envelope. Load only from trusted files (it unpickles)."""
    _joblib().dump(
        {
            "format": _ENVELOPE_VERSION,
            "model_type": model_type,
            "state": dict(state),
            "manifest": dict(manifest) if manifest else {},
        },
        path,
    )


def read_envelope(path: str | Path) -> dict[str, Any]:
    """Read a joblib envelope, normalizing a legacy gradient-boosted dict into the new shape."""
    try:
        data = _joblib().load(path)
    except (OSError, ImportError):
        raise  # missing file, or the ml extra is absent: callers handle these
    except Exception as exc:  # a corrupt or non-joblib file: give a clean domain error
        raise ConfigError(f"could not read model file {path}: {exc}") from exc
    if isinstance(data, Mapping) and "model_type" in data:
        return {
            "model_type": str(data["model_type"]),
            "state": dict(data.get("state", {})),
            "manifest": dict(data.get("manifest", {})),
        }
    if isinstance(data, Mapping) and "estimator" in data:  # legacy gradient-boosted save
        return {"model_type": "gradient-boosted", "state": dict(data), "manifest": {}}
    raise ConfigError(f"unrecognized model file: {path}")


def save_model(
    model: StressModel, path: str | Path, manifest: Mapping[str, object] | None = None
) -> None:
    """Persist a stress model (heuristic, gradient-boosted, or ensemble) to ``path``."""
    if not isinstance(model, Persistable):
        raise ConfigError(f"{type(model).__name__} cannot be saved (not persistable)")
    write_envelope(model.MODEL_TYPE, model.state(), path, manifest)


def load_model(path: str | Path) -> StressModel:
    """Load a stress model saved by :func:`save_model` (or a legacy gradient-boosted file)."""
    envelope = read_envelope(path)
    return model_from_state(envelope["model_type"], envelope["state"])


def load_saved(path: str | Path) -> StressModel | SplitConformalClassifier:
    """Load any saved file: a calibrated file yields a conformal wrapper, else a stress model."""
    envelope = read_envelope(path)
    if envelope["model_type"] == "conformal":
        from phytovision.models.conformal import SplitConformalClassifier

        return SplitConformalClassifier.from_state(envelope["state"])
    return model_from_state(envelope["model_type"], envelope["state"])


def model_from_state(model_type: str, state: Mapping[str, Any]) -> StressModel:
    """Reconstruct a stress model from its type tag and state. Imports are lazy to avoid cycles."""
    if model_type == "heuristic":
        from phytovision.models.stress.heuristic import HeuristicStressModel

        return HeuristicStressModel.from_state(state)
    if model_type == "gradient-boosted":
        from phytovision.models.stress.gradient_boosted import GradientBoostedStressModel

        return GradientBoostedStressModel.from_state(state)
    if model_type == "ensemble":
        from phytovision.models.stress.ensemble import EnsembleStressModel

        return EnsembleStressModel.from_state(state)
    raise ConfigError(f"unknown model_type {model_type!r}")


def _versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for dist in ("numpy", "scikit-learn", "phytovision"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:  # pragma: no cover - environment dependent
            continue
    return out


def _joblib() -> Any:
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "model persistence needs the 'ml' extra: pip install -e \".[ml]\""
        ) from exc
    return joblib
