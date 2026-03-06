from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import zipfile
import subprocess
from functools import lru_cache

from awpy.visibility import VisibilityChecker  # pip-installed awpy

# ---- CONFIG ----
AWPY_EXE = Path(r"C:\NullCS\NewAnubisTri\.venv\Scripts\awpy.exe")
AWPY_HOME = Path(os.environ.get("AWPY_HOME", str(Path.home() / ".awpy")))
TRIS_DIR = AWPY_HOME / "tris"


def map_name_from_zip(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path, "r") as z:
        header = json.loads(z.read("header.json").decode("utf-8"))
    return header["map_name"]


def _run_awpy_get(which: str) -> None:
    # which: "tris" / "maps" / "navs"
    subprocess.run([str(AWPY_EXE), "get", which], check=True)


def tri_path_for_map(map_name: str) -> Path:
    """Finds the .tri file for a map, downloading tris bundle if missing."""
    TRIS_DIR.mkdir(parents=True, exist_ok=True)

    matches = sorted(TRIS_DIR.glob(f"*{map_name}*.tri"))
    if matches:
        return matches[0]

    # Download if missing
    _run_awpy_get("tris")

    matches = sorted(TRIS_DIR.glob(f"*{map_name}*.tri"))
    if not matches:
        raise FileNotFoundError(f"Tri asset missing for '{map_name}' in {TRIS_DIR}")
    return matches[0]


@lru_cache(maxsize=64)
def get_visibility_checker(map_name: str) -> VisibilityChecker:
    """Cached per-map visibility checker (BVH build is the expensive part)."""
    tri_path = tri_path_for_map(map_name)
    return VisibilityChecker(path=tri_path)


@dataclass(frozen=True)
class Point3:
    x: float
    y: float
    z: float


def is_visible(map_name: str, p1: Point3, p2: Point3) -> bool:
    vc = get_visibility_checker(map_name)
    return bool(vc.is_visible((p1.x, p1.y, p1.z), (p2.x, p2.y, p2.z)))
