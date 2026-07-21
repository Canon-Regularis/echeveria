"""A small, seeded generative model of a succulent dry-down.

No succulent water-stress time series ships with this project, so the advanced forecasters, the
survival model, and the benchmark have nothing to fit or score. This module fills that gap with
honest synthetic data. It is a compact generative tool for benchmark sequences, not a plant
physiology model: the "physiology" it borrows is one latent stress state that rises under a
dry-down forcing.

Each synthetic plant has a latent stress state that starts near zero and climbs toward one as the
plant dries. The observed stress score is that state plus observation noise. A wilt event fires the
first step the latent state crosses the stressed cut, so the event uses the same threshold the
forecaster and the verdict use. A sequence that never crosses within the window is right censored.
The per-step feature vectors use the real feature namespaces (``colour.*``, ``geometry.*``, ...), so
the same models that run on real images run unchanged on these synthetic vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from phytovision._num import clip01
from phytovision.models.base import STRESSED_THRESHOLD
from phytovision.temporal.history import Observation

# Each feature moves monotonically from a healthy endpoint (latent 0) to a stressed endpoint
# (latent 1), so a model reading these vectors sees greenness and turgor fall while yellowing,
# browning, reddening, texture, and outline concavity rise. The endpoints sit inside the heuristic
# model's own term ranges, so its score tracks the latent state. Each tuple is
# (healthy_value, stressed_value, clamp_low, clamp_high).
_FEATURE_CURVES: dict[str, tuple[float, float, float, float]] = {
    "colour.gcc_mean": (0.40, 0.28, 0.0, 1.0),
    "colour.exg_mean": (0.22, 0.00, -0.5, 1.0),
    "colour.greenness_ratio": (0.90, 0.15, 0.0, 1.0),
    "colour.saturation_mean": (0.55, 0.18, 0.0, 1.0),
    "geometry.solidity": (0.93, 0.45, 0.0, 1.0),
    "colour.yellow_fraction": (0.02, 0.45, 0.0, 1.0),
    "colour.brown_fraction": (0.01, 0.40, 0.0, 1.0),
    "colour.red_fraction": (0.02, 0.30, 0.0, 1.0),
    "texture.entropy": (2.2, 4.8, 0.0, 8.0),
    "texture.glcm_contrast": (0.30, 4.00, 0.0, 16.0),
    "morphology.concavity": (0.03, 0.42, 0.0, 1.0),
}


def feature_keys() -> list[str]:
    """The feature keys every synthetic observation carries, in a stable sorted order."""
    return sorted(_FEATURE_CURVES)


@dataclass(frozen=True, slots=True)
class DryDownParams:
    """The knobs of the dry-down. Every source of randomness is explicit and documented.

    ``base_decline_rate`` sets how fast the latent state saturates toward one, and
    ``decline_rate_spread`` gives each plant its own rate so a cohort is heterogeneous.
    ``process_noise`` perturbs the latent walk; ``observation_noise`` corrupts the readout;
    ``feature_noise`` jitters each synthetic feature. ``watering_steps`` rehydrate the plant by
    ``watering_amount`` at those step indices.
    """

    n_steps: int = 20
    base_decline_rate: float = 0.12
    decline_rate_spread: float = 0.04
    process_noise: float = 0.01
    observation_noise: float = 0.03
    feature_noise: float = 0.02
    initial_stress: float = 0.05
    watering_steps: tuple[int, ...] = ()
    watering_amount: float = 0.25
    base_date: str = "2024-01-01"
    step_days: int = 1


@dataclass(frozen=True, slots=True)
class SyntheticSeries:
    """One synthetic plant: its observed sequence, its latent truth, and its wilt event."""

    plant_id: str
    decline_rate: float
    observations: tuple[Observation, ...]
    latent: tuple[float, ...] = field(default_factory=tuple)
    event_step: int | None = None
    censored: bool = True

    @property
    def duration(self) -> int:
        """Steps until the wilt event, or the last observed step for a censored sequence."""
        return self.event_step if self.event_step is not None else len(self.latent) - 1

    @property
    def event_time(self) -> str:
        """The timestamp at the wilt event, or the last timestamp for a censored sequence."""
        return self.observations[self.duration].timestamp


def _draw_rate(params: DryDownParams, rng: np.random.Generator) -> float:
    """A per-plant decline rate: the base rate times a positive lognormal factor around one."""
    factor = float(np.exp(rng.normal(0.0, params.decline_rate_spread)))
    return params.base_decline_rate * factor


def _timestamp(params: DryDownParams, step: int) -> str:
    """The ISO date for a step, so timestamps sort as text and space evenly by ``step_days``."""
    start = date.fromisoformat(params.base_date)
    return (start + timedelta(days=step * params.step_days)).isoformat()


def _synthesize_features(
    latent: float, params: DryDownParams, rng: np.random.Generator
) -> dict[str, float]:
    """Turn a latent stress value into a real-namespace feature vector with small noise."""
    features: dict[str, float] = {}
    for key, (healthy, stressed, low, high) in _FEATURE_CURVES.items():
        base = healthy + latent * (stressed - healthy)
        noisy = base + float(rng.normal(0.0, params.feature_noise))
        features[key] = min(high, max(low, noisy))
    return features


def simulate_plant(
    plant_id: str,
    params: DryDownParams,
    rng: np.random.Generator,
    decline_rate: float | None = None,
) -> SyntheticSeries:
    """Simulate one plant's dry-down. Pass ``decline_rate`` to pin it; otherwise it is drawn."""
    rate = decline_rate if decline_rate is not None else _draw_rate(params, rng)
    watering = set(params.watering_steps)

    # Watering at step 0 acts on the initial state; the update loop below starts at step 1, so a
    # step-0 watering event would otherwise be silently ignored.
    initial = params.initial_stress - (params.watering_amount if 0 in watering else 0.0)
    latent: list[float] = [clip01(initial)]
    for step in range(1, params.n_steps):
        previous = latent[-1]
        drift = rate * (1.0 - previous)
        value = previous + drift + float(rng.normal(0.0, params.process_noise))
        if step in watering:
            value -= params.watering_amount
        latent.append(clip01(value))

    observations = tuple(
        Observation(
            plant_id=plant_id,
            timestamp=_timestamp(params, step),
            stress_score=clip01(value + float(rng.normal(0.0, params.observation_noise))),
            features=_synthesize_features(value, params, rng),
        )
        for step, value in enumerate(latent)
    )

    event_step = next(
        (step for step, value in enumerate(latent) if value >= STRESSED_THRESHOLD), None
    )
    return SyntheticSeries(
        plant_id=plant_id,
        decline_rate=rate,
        observations=observations,
        latent=tuple(latent),
        event_step=event_step,
        censored=event_step is None,
    )
