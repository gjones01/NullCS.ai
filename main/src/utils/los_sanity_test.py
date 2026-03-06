from __future__ import annotations

from pathlib import Path
import os, struct, zipfile, json
import numpy as np

AWPY_EXE = Path(r"C:\NullCS\NewAnubisTri\.venv\Scripts\awpy.exe")
AWPY_HOME = Path(os.environ.get("AWPY_HOME", str(Path.home() / ".awpy")))
TRIS_DIR = AWPY_HOME / "tris"

ZIP_PATH = Path(r"C:\NullCS\parsed_zips\Normal001.zip")  # change demo here


def get_map_name_from_zip(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path, "r") as z:
        header = json.loads(z.read("header.json").decode("utf-8"))
    return header["map_name"]


def find_tri_path(map_name: str) -> Path:
    tri = sorted(TRIS_DIR.glob(f"*{map_name}*.tri"))
    if not tri:
        raise FileNotFoundError(f"No .tri for map '{map_name}' in {TRIS_DIR}")
    return tri[0]


def load_tri_vertices(tri_path: Path) -> np.ndarray:
    """
    NOTE: .tri format varies by tool/version.
    We'll implement the correct parser after a quick file sniff.
    For now, just return raw bytes length to confirm file reads.
    """
    data = tri_path.read_bytes()
    print("[INFO] tri bytes:", len(data))
    return np.empty((0, 3, 3), dtype=np.float32)


def main():
    map_name = get_map_name_from_zip(ZIP_PATH)
    print("[INFO] map_name:", map_name)

    tri_path = find_tri_path(map_name)
    print("[OK] tri_path:", tri_path)

    tris = load_tri_vertices(tri_path)
    print("[INFO] tris loaded:", tris.shape)


if __name__ == "__main__":
    main()
