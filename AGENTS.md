# Repository Guidelines

## Project Structure & Module Organization

This repository contains research code for explainable board-game recommendation systems. Reusable Python utilities live in `reco_systems/`, including collaborative filtering, matrix construction, text processing, LLM helpers, and evaluation functions. Exploratory and evaluation workflows are kept as Jupyter notebooks in `notebooks/`, with earlier cleaning and visualization notebooks in `data_cleaning_analysis/` and `data_visualization/`. The Dash/DeckGL demo application is in `app_interactive/`; its app-local JSON and NumPy artifacts are required at runtime. Shared result artifacts are in `generated_data/`. Large deliverables such as `Project_Report.pdf` and `Preview_video.mp4` document the project output.

## Build, Test, and Development Commands

No package manager or test runner is configured in the repository. Use a local virtual environment and install dependencies inferred from imports:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy scikit-learn matplotlib dash dash-bootstrap-components plotly pydeck dash-deck dash-extensions jupyter
```

Run the interactive app from its data directory so relative file paths resolve:

```bash
cd app_interactive
python app.py
```

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation and descriptive snake_case names for functions, variables, and modules. Keep reusable logic in `reco_systems/`; notebooks should call those helpers instead of duplicating algorithms. Prefer NumPy, pandas, SciPy sparse matrices, and scikit-learn APIs over manual loops where practical. Existing docstrings use `Parameters` and `Returns` sections; follow that style for public helpers. Avoid committing generated caches such as `__pycache__/` or `.DS_Store`.

## Testing Guidelines

There is currently no formal `tests/` directory. For new reusable code, add focused tests under `tests/` using `pytest`, with filenames like `test_evaluation.py` and test functions named `test_<behavior>()`. For notebook-only changes, run the affected notebook top-to-bottom and record any required data inputs. For recommendation metrics, include small deterministic fixtures and set random seeds before sampling hidden ratings.

## Commit & Pull Request Guidelines

Recent history uses short, imperative-style messages such as `Video added`, `project report added`, and `cleaning`. Keep commits concise and scoped to one change. Pull requests should summarize the affected workflow, list commands or notebooks run, note data artifacts added or regenerated, and include screenshots or a short recording for `app_interactive` UI changes. Link related issues when available.

## Security & Configuration Tips

Do not commit raw scraped databases or private/local datasets; `.gitignore` already excludes `trictrac_database/`, `database_cleaned/`, and most CSV files. Document any required external data location and keep app assets small enough for normal repository use.
