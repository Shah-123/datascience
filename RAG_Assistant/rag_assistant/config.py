"""Paths and environment handling."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_DIR / "data" / "corpus"
GOLDEN_SET = PROJECT_DIR / "data" / "eval" / "golden_set.yaml"
CALIBRATION_SET = PROJECT_DIR / "data" / "eval" / "support_calibration.yaml"
REPORTS_DIR = PROJECT_DIR / "reports"
BEST_CONFIG = REPORTS_DIR / "best_config.json"
GATES = REPORTS_DIR / "gates.json"


def load_env_file(path: str | Path | None = None) -> bool:
    """Load ``KEY=VALUE`` lines from a .env file into ``os.environ`` (never overriding real env vars).

    Kept dependency-free on purpose. The default location is ``RAG_Assistant/.env``, which is
    git-ignored - API keys must never be committed.
    """
    path = Path(path or os.environ.get("RAG_ENV_FILE") or PROJECT_DIR / ".env")
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip("'\"")
        if value:  # a blank `KEY=` line in .env.example must not shadow a real variable
            os.environ.setdefault(key.strip().removeprefix("export "), value)
    return True
