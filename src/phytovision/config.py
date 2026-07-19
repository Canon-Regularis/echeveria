"""Read a pipeline configuration file into a plain dict.

One place for the CLI and the serving surfaces to parse a config, so the accepted formats and their
error messages stay identical. This module imports nothing from the package beyond the exception
type, so any layer can depend on it without a cycle.
"""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path

from phytovision.exceptions import ConfigError


def read_config(path: str | os.PathLike[str]) -> dict[str, object]:
    """Parse a .toml or .json config file into a dict, raising ConfigError on any problem."""
    file = Path(path)
    suffix = file.suffix.lower()
    if suffix not in {".toml", ".json"}:  # reject the extension before reading the bytes
        raise ConfigError(f"config must be .toml or .json: {file}")
    try:
        # utf-8-sig strips a byte-order mark if present (editors like Notepad add one), so a valid
        # BOM-prefixed config is not rejected as unparseable; plain UTF-8 is read unchanged.
        text = file.read_text(encoding="utf-8-sig")  # a missing file raises FileNotFoundError
    except UnicodeDecodeError as exc:
        raise ConfigError(f"config {file} is not valid UTF-8 text: {exc}") from exc
    try:
        data = tomllib.loads(text) if suffix == ".toml" else json.loads(text)
    except (tomllib.TOMLDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"could not parse config {file}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config must be a table/object at the top level: {file}")
    return data
