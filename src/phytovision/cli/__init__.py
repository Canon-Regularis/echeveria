"""Command-line surface: ``phytovision [-v] <command> ...``.

Each command is a self-contained module exposing ``add_parser(subparsers)`` and ``run(args)``. A new
command is one new module added to ``_COMMANDS``: the parser wiring and the dispatch stay untouched.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from phytovision.cli import analyze, batch, dashboard, evaluate, phenotype, serve, train
from phytovision.cli._shared import configure_logging, parse_horizons

# Re-exported under its historical name so existing imports keep working.
_parse_horizons = parse_horizons

_COMMANDS = (analyze, batch, train, evaluate, serve, dashboard, phenotype)

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
    exit_code: int = args.func(args)
    return exit_code
