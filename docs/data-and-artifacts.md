# Data and Artifact Policy

This repo should contain source code, small reproducible fixtures, notebooks, and final documentation. Avoid committing raw scraped data, local database exports, large generated tables, model checkpoints, caches, and environment-specific files.

## Ignored Paths

- `trictrac_database/`: original scraped TricTrac database; large, local, and not required in Git.
- `database_cleaned/`: local cleaned database export; reproducible from raw data and cleaning notebooks.
- `data/raw/`: raw local inputs. Keep outside Git because provenance and size can vary.
- `data/interim/`: temporary transformation outputs between cleaning/modeling steps.
- `data/processed/`: large processed outputs that can be regenerated.
- `generated_data/comments_stemmed.csv`: generated NLP table; too large for normal Git history.
- `.venv/`: local Python environment created by `uv`.
- `.pytest_cache/`, `.ruff_cache/`: local verification caches.
- `uv.lock`: generated for local reproducibility checks, but intentionally not committed for this research cleanup.
- `.ipynb_checkpoints/`, `__pycache__/`, `.DS_Store`: tool and OS caches.

## Allowed Small Artifacts

Small app fixtures may stay in Git when they are required to run the demo and are not easy to regenerate during review. Examples include compact JSON or NumPy arrays used by the Dash app. If an artifact grows beyond practical review size, document how to recreate it instead of committing it.
