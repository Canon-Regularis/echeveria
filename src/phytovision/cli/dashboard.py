"""The ``dashboard`` command: launch the Streamlit dashboard (needs the 'dashboard' extra)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from phytovision.cli._shared import fail
from phytovision.exceptions import PhytoVisionError
from phytovision.serving import serving_env, validate_serving_selection


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "dashboard", help="run the Streamlit dashboard (needs the 'dashboard' extra)"
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind host")
    parser.add_argument("--port", type=int, default=8501, help="bind port")
    parser.add_argument(
        "--config", metavar="FILE", help="pipeline config (.toml/.json) for the dashboard"
    )
    parser.add_argument(
        "--model-path", metavar="FILE", help="trained or calibrated .joblib model to analyze with"
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    try:
        import streamlit  # noqa: F401  (import only to confirm the 'dashboard' extra is present)
    except ImportError:
        return fail('the dashboard needs the "dashboard" extra: pip install -e ".[dashboard]"')

    try:
        validate_serving_selection(args.config, args.model_path)
    except (OSError, ImportError, PhytoVisionError) as exc:
        return fail(str(exc))

    import os
    import subprocess

    env = {**os.environ, **serving_env(args.config, args.model_path)}
    script = str(Path(__file__).parent.parent / "dashboard.py")
    command = [  # pragma: no cover: launches the external Streamlit server
        sys.executable,
        "-m",
        "streamlit",
        "run",
        script,
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        # A neutral-dark terminal theme: near-black panels, light-gray monospace text, one accent.
        "--theme.base",
        "dark",
        "--theme.backgroundColor",
        "#0e1116",
        "--theme.secondaryBackgroundColor",
        "#161b22",
        "--theme.textColor",
        "#c9d1d9",
        "--theme.primaryColor",
        "#4c9aff",
        "--theme.font",
        "monospace",
    ]
    return subprocess.call(command, env=env)  # pragma: no cover
