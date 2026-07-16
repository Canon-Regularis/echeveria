"""The ``serve`` command: run the HTTP API (needs the 'api' extra)."""

from __future__ import annotations

import argparse
from pathlib import Path

from phytovision.cli._shared import fail
from phytovision.config import read_config
from phytovision.exceptions import PhytoVisionError
from phytovision.models.persistence import load_saved


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("serve", help="run the HTTP API (needs the 'api' extra)")
    parser.add_argument("--host", default="127.0.0.1", help="bind host")
    parser.add_argument("--port", type=int, default=8000, help="bind port")
    parser.add_argument(
        "--config", metavar="FILE", help="pipeline config (.toml/.json) for the served app"
    )
    parser.add_argument(
        "--model-path", metavar="FILE", help="trained or calibrated .joblib model to serve"
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        return fail('serving needs the "api" extra: pip install -e ".[api]"')

    # Validate the chosen pipeline here so a bad path is a clean error, not a traceback when the
    # app is imported by string below (create_app runs at import and reads these files).
    try:
        if args.config:
            read_config(args.config)
        if args.model_path:
            load_saved(args.model_path)
    except (OSError, ImportError, PhytoVisionError) as exc:
        return fail(str(exc))

    import os

    if args.config:
        os.environ["PHYTOVISION_CONFIG"] = str(Path(args.config))
    if args.model_path:
        os.environ["PHYTOVISION_MODEL_PATH"] = str(Path(args.model_path))
    uvicorn.run("phytovision.api:app", host=args.host, port=args.port)  # pragma: no cover
    return 0  # pragma: no cover
