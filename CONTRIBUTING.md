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

## Reproducibility and data versioning

- **Seeding.** The commands that draw randomness accept `--seed` (`train`, `evaluate`, `simulate`).
  A seed threads into the per-stage generators (the simulator's per-plant streams, the
  cross-validation splits, the gradient-boosted `random_state`) and, in one place in the CLI entry
  point, into `set_global_seed` from [`src/phytovision/seeding.py`](src/phytovision/seeding.py), which
  seeds Python's `random` and numpy's legacy global generator. Re-running with the same seed and inputs
  reproduces the run. The `benchmark` command is already deterministic (its forecasters carry fixed
  internal seeds), so it needs no `--seed`.
- **Validated config.** `Pipeline.from_config` routes the parsed dict through
  [`config_schema.PipelineConfig`](src/phytovision/config_schema.py), which rejects an unknown
  top-level key (so a mistyped slot name fails loudly instead of silently) and fills every slot's
  default. `PipelineConfig.from_mapping(cfg).as_dict()` gives a canonical, resolved config you can log
  or diff.
- **Experiment tracking (optional).** With the `tracking` extra installed, `phytovision benchmark
  --mlflow` logs the forecaster comparison (params and per-model metrics) to MLflow, so runs are
  comparable across changes.
- **Data and artifact versioning with DVC (optional).** DVC is not a dependency; it is a recommended
  way to version large datasets and model artifacts without committing them to git. Set it up once:

  ```bash
  pip install dvc            # optional, not in any extra
  dvc init                   # creates .dvc/ and a .dvc/config
  dvc add data/ models/      # track large paths; commit the small .dvc pointer files, not the data
  dvc remote add -d storage s3://your-bucket/echeveria   # or gdrive, ssh, a local path, ...
  dvc push                   # upload the tracked data to the remote
  ```

  A minimal `.dvc/config` then reads:

  ```ini
  [core]
      remote = storage
  ['remote "storage"']
      url = s3://your-bucket/echeveria
  ```

  Teammates run `dvc pull` to fetch the exact data a commit points at, so an experiment is
  reproducible from the git SHA plus the DVC pointers.

## Conventions

- Public functions/classes are typed and have docstrings; the package is `py.typed`.
- Library code uses `logging` (module-level `logger = logging.getLogger(__name__)`), never `print`.
- Raise from the `phytovision.exceptions` hierarchy for library errors.
- Keep changes covered by tests; note user-facing changes in [CHANGELOG.md](CHANGELOG.md).
