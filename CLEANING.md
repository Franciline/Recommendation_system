# Repository cleaning

The project has already been finished, but the currently structured repository is extremely messy.
This file contains TODOs to clean the repository. Once the task in `## TODO` section is done, it should be marked as done, i.e. [v].

## Problems addressed

- Reusable Python code is in the top-level `reco_systems/` package while an empty `src/` directory already exists.
- Runtime app code, app-generated data, and data-generation notebooks are mixed together in `app_interactive/`.
- Research notebooks are split across `notebooks/`, `data_cleaning_analysis/`, `data_visualization/`, and `app_interactive/` without a clear naming scheme.
- `pyproject.toml` exists but has no dependencies, scripts, package mapping, linting, or test configuration.
- Large generated artifacts are tracked in normal Git, especially `generated_data/comments_stemmed.csv`, `Preview_video.mp4`, and output-heavy notebooks.
- The Dash app uses current-working-directory relative paths such as `np.load("tsne_pushed.npy")`, so it only runs from `app_interactive/`.
- Several modules use wildcard imports and inconsistent names, for example `CF_knn.py` and `evaluation_bigrams_func.py`.

## Target structure

Recommended final layout:

```text
.
├── README.md
├── pyproject.toml
├── docs/
│   ├── report.md
│   └── app.md
├── config/
│   └── ollama/
│       └── Modelfile
├── src/
│   └── boardgames_recsys/
│       ├── __init__.py
│       ├── app/
│       │   ├── app.py
│       │   └── assets/
│       ├── data/
│       ├── evaluation/
│       ├── models/
│       ├── text/
│       └── visualization/
├── notebooks/
│   ├── 01_cleaning/
│   ├── 02_exploration/
│   ├── 03_models/
│   ├── 04_evaluation/
│   └── 05_app_generation/
├── data/
│   ├── raw/              # ignored, local only
│   ├── interim/          # ignored unless small/reproducible
│   ├── processed/        # ignored unless required demo fixture
│   └── app/              # small files required by Dash demo
├── reports/
│   ├── Project_Report.pdf
│   └── Preview_video.mp4
└── tests/
```

## TODO

### Packaging and dependency management

- [v] Complete `pyproject.toml` for `uv`.
  - Set a supported Python version. The current `requires-python = ">=3.13"` is probably too strict because the notebooks reference Python 3.11 environments and some scientific packages may lag on 3.13.
  - Add runtime dependencies found in the code: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `dash`, `dash-bootstrap-components`, `plotly`, `pydeck`, `dash-deck`, `dash-extensions`, `nltk`, `unidecode`, `sacrebleu`, `rouge`, `ollama`, `pillow`, and `scikit-surprise` if NNMF/SVD notebooks must remain runnable.
  - Add optional dependency groups such as `dev` (`pytest`, `ruff`, `nbstripout`, `jupyterlab`) and `app` if the Dash demo should be installable separately.
  - Generate and DO NOT commit `uv.lock` with `uv lock`.
- [v] Add useful project scripts.
  - Example: `boardgames-recsys-app = "boardgames_recsys.app.app:main"` for the Dash app.
  - Optionally add a CLI entry point for reproducible data generation/evaluation commands.

### Source package migration

- [v] Move `reco_systems/` into `src/boardgames_recsys/`.
  - Python imports were converted from `reco_systems.*` to `boardgames_recsys.*`.
  - The temporary `reco_systems/` compatibility shim was removed after notebook imports were migrated.
  - [v] Convert notebook imports from `reco_systems.*` to `boardgames_recsys.*`.
- [v] Rename unclear modules while moving them.
  - `CF_knn.py` -> `models/collaborative_filtering.py`
  - `user_game_matrix.py` -> `data/matrix.py`
  - `filter.py` -> `data/filtering.py`
  - `text_filtering.py`, `lemmatization.py`, `embeds_utils.py`, `llm.py` -> `text/`
  - `evaluation.py`, `evaluation_weight.py`, `evaluation_bigrams_func.py` -> `evaluation/`
- [v] Remove wildcard imports from reusable code.
  - Replace `from ... import *` with explicit imports to make dependencies and public APIs clear.
  - Legacy compatibility wrappers were removed after imports were migrated.
- [v] Split oversized modules.
  - Bigram summary, filtering, and neighbor-selection helpers were moved out of `evaluation/bigrams.py`.
  - `evaluation.py` and `evaluation_weight.py` duplicate concepts; shared hide/predict/metric code was merged into `evaluation/ratings.py`.

### `src` cleanup

- [v] Rename modules and functions whose names do not match the logic they implement.
  - `calc_similarity_matrix` was renamed to `calc_distance_matrix`; the old name remains as a compatibility alias.
- [v] Rename modules/functions if necessary to make their naming more explicit.

### Dash app cleanup

- [v] Move `app_interactive/app.py` to `src/boardgames_recsys/app/app.py`.
- [v] Move `app_interactive/assets/style.css` with the app package.
- [v] Move app runtime artifacts (`clusters.npy`, `tsne_pushed.npy`, `nnmf_prediction.npy`, `games_info.json`, `users_info.json`, `summaries.json`, `special_user_comments.json`) to `data/app/` or `src/boardgames_recsys/app/data/`.
- [v] Replace working-directory relative paths with paths based on `pathlib.Path(__file__)` or configurable environment variables.
- [v] Move `app_interactive/data_gen.ipynb` and `comment_gen.ipynb` to `notebooks/05_app_generation/`.

### Notebook organization

- [v] Consolidate all notebooks under `notebooks/` by workflow stage.
  - `data_cleaning_analysis/jeux_cleaning.ipynb` -> `notebooks/01_cleaning/`
  - `data_visualization/*.ipynb` -> `notebooks/02_exploration/`
  - model training and clustering notebooks -> `notebooks/03_models/`
  - evaluation notebooks -> `notebooks/04_evaluation/`
  - app data generation notebooks -> `notebooks/05_app_generation/`
- [v] Rename notebooks with numeric prefixes and descriptive names, for example `04_nnmf_matrix_eval.ipynb`.
- [v] Strip notebook outputs before committing future changes.
  - Configure `nbstripout` or a pre-commit hook through `uv`.

### Data and artifact policy

- [v] Move `Project_Report.pdf` and `Preview_video.mp4` to `reports/`.
- [v] Decide which generated files must stay in Git.
  - Keep small demo fixtures needed to run the app.
  - Move large reproducible outputs such as `generated_data/comments_stemmed.csv` to ignored `data/processed/` unless needed for review.
- [v] Update `.gitignore`.
  - Ignore `data/raw/`, `data/interim/`, `data/processed/`, notebook checkpoints, model caches, and local `.venv/`.
  - Stop ignoring every `*.csv` globally if small committed fixtures are expected; prefer directory-based ignore rules.
- [v] DO NOT use Git LFS or external release assets for video, PDF, large CSV, and future model/data artifacts. If the file it too large, it should not be commited.
- [v] Create a markdown file that specifies what we avoid commiting to git and what was the function of the ignored file.

### Documentation

- [v] Rewrite `README.md` around the cleaned structure.
  - Include project purpose, installation with `uv sync`, how to run the Dash app, how to reproduce key analyses, and where data must be placed.
- [v] Move long-form material to `docs/` if the README becomes too large.
  - Suggested files: `docs/data.md`, `docs/app.md`, `docs/reproducibility.md`, `docs/modeling.md`.
- [v] Update `AGENTS.md` after restructuring so contributor guidance matches the new layout and commands.

### Tests, linting, and verification

- [v] Add `tests/` for reusable functions.
  - Start with deterministic tests for `filter_df`, matrix construction, KNN neighbor selection, hiding ratings, and metric calculations.
- [v] Configure `ruff` in `pyproject.toml`.
  - Enforce import sorting, unused imports, and obvious bug checks before doing broader style cleanup.
- [v] Add a minimal smoke test for the app import or layout creation.
- [v] Define verification commands.
  - `uv run pytest`
  - `uv run ruff check .`
  - `uv run boardgames-recsys-app`

### Completed migration order

1. Finish `pyproject.toml`, create `uv.lock`, and confirm the current app/notebooks can import dependencies.
2. Move `reco_systems/` into `src/boardgames_recsys/`, migrate imports, then delete legacy shims.
3. Update imports in Python files, then notebooks.
4. Move app code and data, then fix file loading paths.
5. Reorganize notebooks and reports.
6. Tighten `.gitignore`, strip notebook outputs, and decide what large artifacts remain tracked.
7. Add tests and linting once imports and paths are stable.
