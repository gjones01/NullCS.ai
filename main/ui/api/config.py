from __future__ import annotations

import os
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(os.getenv("CLARITY_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
DEFAULT_VENV_PYTHON = PROJECT_ROOT / "NewAnubisTri" / ".venv" / "Scripts" / "python.exe"

def _is_healthy_python(candidate: str) -> bool:
    try:
        probe = subprocess.run(
            [candidate, "-c", "import pandas, polars, xgboost"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        return probe.returncode == 0
    except Exception:
        return False


def _pick_python_exe() -> str:
    env_python = os.getenv("CLARITY_PYTHON_EXE", "").strip()
    if env_python:
        return env_python
    venv_python = str(DEFAULT_VENV_PYTHON)
    if DEFAULT_VENV_PYTHON.exists() and _is_healthy_python(venv_python):
        return venv_python
    return "python"


PYTHON_EXE = Path(_pick_python_exe())
RAW_UPLOADS_DIR = Path(os.getenv("CLARITY_RAW_UPLOADS_DIR", str(PROJECT_ROOT / "main" / "data" / "raw_uploads")))
PROCESSED_DIR = Path(os.getenv("CLARITY_PROCESSED_DIR", str(PROJECT_ROOT / "main" / "data" / "processed")))
SCRIPTS_DIR = Path(os.getenv("CLARITY_SCRIPTS_DIR", str(PROJECT_ROOT / "main" / "scripts")))
MODEL_ARTIFACT = os.getenv("CLARITY_MODEL_ARTIFACT", "").strip()

STATE_DIR = PROJECT_ROOT / "main" / "ui" / "api" / "state"
JOBS_PATH = STATE_DIR / "jobs.json"

REPORTS_DIR = PROCESSED_DIR / "reports"
INFER_CSV = REPORTS_DIR / "ranked_player_demo_suspicion_infer.csv"
