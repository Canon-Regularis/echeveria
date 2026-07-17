"""The global seeding utility and its CLI wiring."""

from __future__ import annotations

import random

import numpy as np

from phytovision.seeding import set_global_seed


def test_set_global_seed_is_deterministic() -> None:
    set_global_seed(0)
    first = (random.random(), float(np.random.random()))
    set_global_seed(0)
    second = (random.random(), float(np.random.random()))
    assert first == second


def test_different_seeds_produce_different_streams() -> None:
    set_global_seed(1)
    one = (random.random(), float(np.random.random()))
    set_global_seed(2)
    two = (random.random(), float(np.random.random()))
    assert one != two


def test_out_of_range_seeds_do_not_crash_and_stay_deterministic() -> None:
    # numpy's legacy generator only accepts [0, 2**32); a negative or very large --seed must be
    # reduced into range, not crash, and still reproduce.
    for seed in (-1, 2**32 + 5, -987654321):
        set_global_seed(seed)
        first = (random.random(), float(np.random.random()))
        set_global_seed(seed)
        assert (random.random(), float(np.random.random())) == first


def test_cli_seed_flag_seeds_the_global_rngs(tmp_path, monkeypatch) -> None:
    # A command with --seed calls set_global_seed once, in the CLI entry point.
    import phytovision.seeding as seeding
    from phytovision.cli import main

    recorded: list[int] = []
    monkeypatch.setattr(seeding, "set_global_seed", recorded.append)
    rc = main(
        [
            "simulate",
            "--out",
            str(tmp_path / "c.csv"),
            "--plants",
            "2",
            "--steps",
            "5",
            "--seed",
            "9",
        ]
    )
    assert rc == 0
    assert recorded == [9]


def test_cli_without_a_seed_does_not_seed_globally(image_path, monkeypatch) -> None:
    import phytovision.seeding as seeding
    from phytovision.cli import main

    recorded: list[int] = []
    monkeypatch.setattr(seeding, "set_global_seed", recorded.append)
    assert main(["analyze", str(image_path)]) == 0
    assert recorded == []  # analyze has no --seed, so the global RNGs are left untouched
