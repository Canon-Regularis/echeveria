# Contributing

Thanks for helping improve **echeveria / PhytoVision**.

## Development setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,ml]"
pre-commit install        # run lint/format/type checks on every commit
```

## The checks CI enforces

Run them locally before pushing — CI runs the exact same commands (see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml)) across Python 3.11–3.13:

```bash
ruff check .            # lint
ruff format --check .   # formatting
mypy                    # strict type check (src)
pytest                  # tests + coverage (fails under 85%)
```

`pre-commit run --all-files` runs ruff, formatting, hygiene hooks, and mypy in one go.

## Adding a pipeline component (the modular path)

Every stage is behind an interface (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)). To add one:

1. Implement the relevant base class / protocol (e.g. `PlantSegmenter`, `FeatureExtractor`,
   `StressModel`).
2. Register it in [`src/phytovision/registries.py`](src/phytovision/registries.py) under a stable name.
3. It is now selectable via `Pipeline.from_names(...)`, `Pipeline.from_config({...})`, and the CLI —
   **no edit to the orchestrator**.
4. Add tests, including an LSP substitution test if it introduces a new implementation of an existing
   seam (see `tests/test_region_providers.py`).

## Conventions

- Public functions/classes are typed and have docstrings; the package is `py.typed`.
- Library code uses `logging` (module-level `logger = logging.getLogger(__name__)`), never `print`.
- Raise from the `phytovision.exceptions` hierarchy for library errors.
- Keep changes covered by tests; note user-facing changes in [CHANGELOG.md](CHANGELOG.md).
