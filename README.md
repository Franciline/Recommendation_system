# Explainable Board-Game Recommendation Systems

Research project on explainable recommendation systems for board games. Source data was scraped from TricTrac in early 2023. The final report and demo video live in `reports/`.

## Repository Layout

- `src/boardgames_recsys/`: reusable Python package.
- `src/boardgames_recsys/data/`: filtering and user-game matrix helpers.
- `src/boardgames_recsys/models/`: collaborative filtering helpers.
- `src/boardgames_recsys/evaluation/`: rating, weighted, and bigram evaluation helpers.
- `src/boardgames_recsys/text/`: text filtering, lemmatization, embedding, and LLM helpers.
- `src/boardgames_recsys/app/`: Dash application.
- `reco_systems/`: temporary compatibility shim for older notebooks that still import `reco_systems.*`.
- `notebooks/01_cleaning/`: data cleaning notebooks.
- `notebooks/02_exploration/`: exploratory analysis and visualization notebooks.
- `notebooks/03_models/`: modeling, clustering, embedding, and LLM notebooks.
- `notebooks/04_evaluation/`: metric and recommendation evaluation notebooks.
- `notebooks/05_app_generation/`: notebooks used to generate Dash app artifacts.
- `data/app/`: small runtime artifacts required by the Dash demo.
- `docs/`: project maintenance and data/artifact policy docs.
- `tests/`: focused pytest checks for reusable code and app import smoke tests.

## Setup

Use `uv` with Python 3.11:

```bash
uv sync --python 3.11
```

`uv.lock` is intentionally ignored. Regenerate it locally when dependency resolution needs checking:

```bash
uv lock --python 3.11
```

## Run

Launch the interactive ELA app:

```bash
uv run --python 3.11 boardgames-recsys-app
```

The app reads data from `data/app/` by default. Override this with:

```bash
BOARDGAMES_RECSYS_APP_DATA=/path/to/app-data uv run --python 3.11 boardgames-recsys-app
```

Start notebooks:

```bash
uv run --python 3.11 jupyter lab
```

## Verify

```bash
uv run --python 3.11 pytest
uv run --python 3.11 ruff check .
```

Some legacy lint categories are temporarily ignored per file in `pyproject.toml` until source cleanup continues.
