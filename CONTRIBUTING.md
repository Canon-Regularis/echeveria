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

Run them locally before pushing. CI runs the exact same commands (see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml)) across Python 3.11 to 3.13:

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
3. It is now selectable via `Pipeline.from_names(...)`, `Pipeline.from_config({...})`, and the CLI,
   with **no edit to the orchestrator**.
4. Add tests. The contract suite in [`tests/contracts/`](tests/contracts/) is registry-driven, so your
   component is checked against the shared substitutability invariants the moment it is registered. Add
   component-specific tests for behaviour beyond the shared contract.

## Test rigor

- The contract suite ([`tests/contracts/`](tests/contracts/)) runs one shared invariant set over every
  registered segmenter, stress model, feature extractor, and region provider, parametrized off the
  registries. Registration alone enrolls a component.
- Property-based tests (Hypothesis) fuzz the same invariants over generated images and feature vectors.
- Metamorphic tests ([`tests/test_metamorphic.py`](tests/test_metamorphic.py)) assert that flips,
  rotation, and uniform brightness do not change the verdict.
- Benchmarks live in [`tests/benchmarks/`](tests/benchmarks/) and are disabled by default. Measure the
  classical pipeline with `pytest tests/benchmarks --benchmark-enable --no-cov`.

## Mutation testing

Mutation testing measures whether the assertions actually catch changes in behaviour. It uses
[`mutmut`](https://github.com/boxed/mutmut), which needs Linux or WSL (it does not run on native
Windows) and runs in CI on Linux. Disable coverage for the per-mutant runs first, because the 85% gate
would otherwise mark mutants killed for the wrong reason:

```bash
PYTEST_ADDOPTS="--no-cov" mutmut run   # mutate the subset in [tool.mutmut] and run the suite
mutmut results                         # list surviving mutants
mutmut browse                          # inspect them interactively
```

A surviving mutant points at a weak assertion. Strengthen the test that should have caught it.

## Conventions

- Public functions/classes are typed and have docstrings; the package is `py.typed`.
- Library code uses `logging` (module-level `logger = logging.getLogger(__name__)`), never `print`.
- Raise from the `phytovision.exceptions` hierarchy for library errors.
- Keep changes covered by tests; note user-facing changes in [CHANGELOG.md](CHANGELOG.md).
