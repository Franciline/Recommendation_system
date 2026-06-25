# Repository Guidelines

## Project Structure & Module Organization

Reusable Python code lives in `src/boardgames_recsys/`, split into `data/`, `models/`, `evaluation/`, `text/`, and `app/`. Top-level modules such as `boardgames_recsys.CF_knn` are compatibility wrappers only. The temporary `reco_systems/` package is only a compatibility shim for old notebooks. Runtime app data lives in `data/app/`. Notebooks are grouped by workflow under `notebooks/01_cleaning/` through `notebooks/05_app_generation/`. Reports and videos live in `reports/`; maintenance notes live in `docs/`; tests live in `tests/`.

## Build, Test, and Development Commands

Use `uv` with Python 3.11:

```bash
uv sync --python 3.11
uv run --python 3.11 pytest
uv run --python 3.11 ruff check .
uv run --python 3.11 boardgames-recsys-app
uv run --python 3.11 jupyter lab
```

`uv.lock` is generated for local checks but intentionally ignored. The app reads `data/app/` unless `BOARDGAMES_RECSYS_APP_DATA` points elsewhere.

## Coding Style & Naming Conventions

Use Python 3.11, 4-space indentation, snake_case functions, and descriptive module names. Prefer canonical imports such as `boardgames_recsys.data.matrix` over compatibility wrappers, and avoid `from ... import *` in new code. Keep reusable logic in `src/boardgames_recsys/`; notebooks should orchestrate experiments, not define shared algorithms. Existing public helpers use `Parameters` and `Returns` docstring sections; keep that style for now.

## Testing Guidelines

Add focused `pytest` tests under `tests/`, named `test_<module>.py` with functions named `test_<behavior>()`. Use small deterministic DataFrames and sparse matrices; set random seeds for sampling-based metrics. For notebook changes, run the affected notebook top-to-bottom and note required data inputs in the PR.

## Commit & Pull Request Guidelines

Commit small, reversible steps. Existing history uses short messages such as `Video added`, `project report added`, and `cleaning`; keep messages concise but more specific where possible. PRs should explain moved paths, commands run, data artifacts added or removed, and app screenshots or recordings for UI changes.

## Data & Artifact Policy

Do not commit raw scraped DBs, local cleaned DBs, large generated tables, model checkpoints, caches, or local environments. See `docs/data-and-artifacts.md` for ignored paths and allowed small demo fixtures.
