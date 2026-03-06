from pathlib import Path
import os
import subprocess

# --- CONFIG ---
AWPY_EXE = Path(r"C:\NullCS\NewAnubisTri\.venv\Scripts\awpy.exe")
MAP_NAME = "de_dust2"

AWPY_HOME = Path(os.environ.get("AWPY_HOME", str(Path.home() / ".awpy")))
TRIS_DIR = AWPY_HOME / "tris"


def run_awpy_get(which: str):
    print(f"[AWPY] running: awpy get {which}")
    subprocess.run([str(AWPY_EXE), "get", which], check=True)


def find_map_asset(folder: Path, map_name: str) -> Path | None:
    if not folder.exists():
        return None
    # Any file that contains the map name
    matches = sorted(folder.glob(f"*{map_name}*"))
    return matches[0] if matches else None


def ensure_map_tris(map_name: str) -> Path:
    tri = find_map_asset(TRIS_DIR, map_name)
    if tri:
        return tri

    run_awpy_get("tris")

    tri = find_map_asset(TRIS_DIR, map_name)
    if not tri:
        raise FileNotFoundError(f"Tri asset still missing for '{map_name}' in {TRIS_DIR}")
    return tri


def main():
    print("[INFO] AWPY_EXE:", AWPY_EXE)
    print("[INFO] AWPY_HOME:", AWPY_HOME)
    print("[INFO] TRIS_DIR:", TRIS_DIR)

    tri_path = ensure_map_tris(MAP_NAME)
    print("[OK] tri_path:", tri_path)


if __name__ == "__main__":
    main()
