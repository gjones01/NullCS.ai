from __future__ import annotations

from pathlib import Path
import sys
import zipfile
import tempfile
import traceback
import json

# ---- CONFIG ----
NORMAL_DIR = Path(r"C:\NullCS\LegitDemos\NormalRenamed")
PRO_DIR    = Path(r"C:\NullCS\LegitDemos\ProsRenamed")
CHEATER_DIR = Path(r"C:\NullCS\CheaterDemos")  # optional later

OUT_ROOT = Path(r"C:\NullCS\parsed_zips")  # outputs: <DemoStem>.zip
MAX_DEMOS = None

IGNORE_DIR_NAMES = {"_tmp_extract", "tmp_extract", "__pycache__"}



def list_demos(root: Path) -> list[Path]:
    if not root.exists():
        return []
    demos = [p for p in root.rglob("*.dem") if p.is_file()]
    out = []
    for p in demos:
        if any(part in IGNORE_DIR_NAMES for part in p.parts):
            continue
        out.append(p)
    return sorted(out)

def write_df_to_parquet(df, path: Path) -> None:
    # df is typically a Polars DataFrame in AWPy
    df.write_parquet(str(path))

def parse_one(demo_path: Path) -> bool:
    from awpy import Demo  # import here so missing awpy errors are obvious

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    zip_path = OUT_ROOT / f"{demo_path.stem}.zip"

    if zip_path.exists() and zip_path.stat().st_size > 0:
        print(f"[SKIP] {demo_path.name} -> already parsed")
        return True

    print(f"[PARSE] {demo_path.name}")
    try:
        demo = Demo(str(demo_path))
        demo.parse()  # parse only (no compress that forces footsteps)

        # Write parquet files to a temp folder then zip them
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Required / common
            tables_written = []

            # Each of these are cached properties. If an event doesn't exist, some can raise KeyError.
            # We'll treat those as "optional" rather than hard-failing.
            def try_write(name: str, getter):
                nonlocal tables_written
                try:
                    df = getter()
                    # Some properties might return empty DF; still write for consistency
                    write_df_to_parquet(df, tmpdir / f"{name}.parquet")
                    tables_written.append(f"{name}.parquet")
                except KeyError:
                    print(f"  [SKIP] {name} (missing required event in demo)")
                except Exception as e:
                    print(f"  [WARN] {name} failed: {e}")

            # These are the core ones you actually need for your pipeline:
            try_write("kills",     lambda: demo.kills)
            try_write("damages",   lambda: demo.damages)
            try_write("shots",     lambda: demo.shots)
            try_write("grenades",  lambda: demo.grenades)
            try_write("smokes",    lambda: demo.smokes)
            try_write("infernos",  lambda: demo.infernos)
            try_write("bomb",      lambda: demo.bomb)
            try_write("ticks",     lambda: demo.ticks)
            try_write("rounds",    lambda: demo.rounds)

            # Footsteps is the one that blows up when player_sound is missing.
            # We WANT to skip it safely.
            try_write("footsteps", lambda: demo.footsteps)

            # Header: always include (works whether demo.header is dict or pydantic model)
            header_path = tmpdir / "header.json"
            header_obj = demo.header

            if hasattr(header_obj, "model_dump"):      # pydantic v2
                header_obj = header_obj.model_dump()
            elif hasattr(header_obj, "dict"):          # pydantic v1
                header_obj = header_obj.dict()

            header_path.write_text(json.dumps(header_obj, indent=2), encoding="utf-8")
            tables_written.append("header.json")

            # Zip it
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for fname in tables_written:
                    z.write(tmpdir / fname, arcname=fname)

        print(f"[OK]   {demo_path.name} -> {zip_path.name}")
        return True

    except Exception:
        print(f"[FAIL] {demo_path.name}")
        traceback.print_exc()
        return False

def main() -> int:
    print(f"Using python: {sys.executable}")

    sources = [
        ("normal", NORMAL_DIR),
        ("pro", PRO_DIR),
        ("cheater", CHEATER_DIR),
    ]

    all_demos: list[Path] = []
    for label, folder in sources:
        demos = list_demos(folder)
        print(f"Found {len(demos)} demos in {label}: {folder}")
        all_demos.extend(demos)

    if not all_demos:
        print("No demos found. Check folder paths in CONFIG.")
        return 1

    ok = 0
    fail = 0
    for i, demo in enumerate(all_demos, start=1):
        if MAX_DEMOS is not None and i > MAX_DEMOS:
            break
        if parse_one(demo):
            ok += 1
        else:
            fail += 1

    print(f"Done. OK={ok}, FAIL={fail}, out={OUT_ROOT}")
    return 0 if fail == 0 else 2

if __name__ == "__main__":
    raise SystemExit(main())
