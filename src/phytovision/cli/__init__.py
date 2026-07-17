"""Command-line surface: ``phytovision [-v] <command> ...``.

Each command is a self-contained module exposing ``add_parser(subparsers)`` and ``run(args)``. A new
command is one new module added to ``_COMMANDS``: the parser wiring and the dispatch stay untouched.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from phytovision.cli import (
    analyze,
    batch,
    benchmark,
    dashboard,
    evaluate,
    phenotype,
    serve,
    simulate,
    train,
    validate,
)
from phytovision.cli._shared import configure_logging, parse_horizons

# Re-exported under its historical name so existing imports keep working.
_parse_horizons = parse_horizons

_COMMANDS = (
    analyze,
    batch,
    train,
    evaluate,
    serve,
    dashboard,
    phenotype,
    validate,
    simulate,
    benchmark,
)

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="phytovision",
        description="Explainable plant water-stress analysis.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable debug logging on stderr"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for command in _COMMANDS:
        command.add_parser(sub)

    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    # A command that accepts --seed also seeds the global RNGs here, in one place, so any stage that
    # falls back to the global random state is reproducible alongside the per-stage seeds.
    seed = getattr(args, "seed", None)
    if seed is not None:
        from phytovision.seeding import set_global_seed

        set_global_seed(seed)
    exit_code: int = args.func(args)
    return exit_code
