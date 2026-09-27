"""
paths.py
--------
Resolves the project root regardless of the current working directory, so
`etl.py`, `forecasting.py`, and notebooks under `notebooks/` all find the
same `data/processed/armani_trading.db` whether you run:
    python src/etl.py                 (cwd = repo root)
    jupyter notebook notebooks/x.ipynb (cwd = notebooks/)
    pytest from anywhere in the repo
"""

from pathlib import Path

_MARKERS = ("requirements.txt", ".git")


def get_project_root() -> Path:
    start = Path(__file__).resolve().parent  # .../<repo>/src
    for parent in [start] + list(start.parents):
        if any((parent / marker).exists() for marker in _MARKERS):
            return parent
    return start.parent  # fallback: assume src/'s parent is the root


PROJECT_ROOT = get_project_root()


def resolve(relative_path: str) -> str:
    """Resolve a path like 'data/processed/armani_trading.db' against the
    project root, returning an absolute path string safe to use from
    anywhere (scripts, notebooks, tests)."""
    return str(PROJECT_ROOT / relative_path)
