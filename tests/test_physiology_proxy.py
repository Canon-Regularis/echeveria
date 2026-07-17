"""Physiology proxies: crude RGB-proxy water potential, stomatal conductance, and transpiration."""

from __future__ import annotations

import math

import numpy as np

from phytovision.models.drought.rule_based import (
    physiology_proxies,
    stomatal_conductance_proxy,
    transpiration_proxy,
    water_potential_proxy,
)
from phytovision.simulation import DryDownParams, simulate_plant

_WET = {
    "colour.gcc_mean": 0.42,
    "geometry.solidity": 0.95,
    "colour.yellow_fraction": 0.0,
    "colour.brown_fraction": 0.0,
    "colour.red_fraction": 0.0,
    "morphology.concavity": 0.0,
    "morphology.radial_variation": 0.0,
}
_DRY = {
    "colour.gcc_mean": 0.28,
    "geometry.solidity": 0.45,
    "colour.yellow_fraction": 0.4,
    "colour.brown_fraction": 0.3,
    "colour.red_fraction": 0.3,
    "morphology.concavity": 0.4,
    "morphology.radial_variation": 0.4,
}

_PROXIES = (water_potential_proxy, stomatal_conductance_proxy, transpiration_proxy)


def test_proxies_finite_and_bounded() -> None:
    mid = {key: (_WET[key] + _DRY[key]) / 2 for key in _WET}
    for values in ({}, _WET, mid, _DRY):
        for proxy in _PROXIES:
            score = proxy(values)
            assert math.isfinite(score) and 0.0 <= score <= 1.0


def test_empty_values_read_well_watered() -> None:
    # An absent feature never fabricates stress: no deficit, fully open, full transpiration.
    assert water_potential_proxy({}) == 0.0
    assert stomatal_conductance_proxy({}) == 1.0
    assert transpiration_proxy({}) == 1.0


def test_directions_under_a_hand_built_drydown() -> None:
    assert water_potential_proxy(_DRY) > water_potential_proxy(_WET)  # deficit rises
    assert stomatal_conductance_proxy(_DRY) < stomatal_conductance_proxy(_WET)  # conductance falls
    assert transpiration_proxy(_DRY) < transpiration_proxy(_WET)  # transpiration falls


def test_directions_under_a_simulated_drydown() -> None:
    params = DryDownParams(n_steps=20, process_noise=0.0, observation_noise=0.0, feature_noise=0.0)
    plant = simulate_plant("p", params, np.random.default_rng(0), decline_rate=0.2)
    first = plant.observations[0].features
    last = plant.observations[-1].features
    assert water_potential_proxy(last) > water_potential_proxy(first)
    assert stomatal_conductance_proxy(last) < stomatal_conductance_proxy(first)
    assert transpiration_proxy(last) < transpiration_proxy(first)


def test_graded_monotone_under_a_coherent_drydown() -> None:
    # A sequence of increasingly dry states moves each proxy in its documented direction throughout.
    sequence = [
        {key: _WET[key] + fraction * (_DRY[key] - _WET[key]) for key in _WET}
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    deficits = [water_potential_proxy(v) for v in sequence]
    conductances = [stomatal_conductance_proxy(v) for v in sequence]
    transpirations = [transpiration_proxy(v) for v in sequence]
    assert all(a <= b for a, b in zip(deficits, deficits[1:], strict=False))
    assert all(a >= b for a, b in zip(conductances, conductances[1:], strict=False))
    assert all(a >= b for a, b in zip(transpirations, transpirations[1:], strict=False))


def test_transpiration_never_exceeds_conductance() -> None:
    # The green-area factor is at most one, so transpiration is bounded by conductance every step.
    params = DryDownParams(n_steps=20)
    plant = simulate_plant("p", params, np.random.default_rng(1))
    for observation in plant.observations:
        features = observation.features
        assert transpiration_proxy(features) <= stomatal_conductance_proxy(features) + 1e-9


def test_physiology_proxies_shape() -> None:
    block = physiology_proxies(_DRY)
    assert set(block) == {
        "water_potential_proxy",
        "stomatal_conductance_proxy",
        "transpiration_proxy",
    }
    for value in block.values():
        assert 0.0 <= value <= 1.0
        assert round(value, 3) == value  # rounded to three decimals
