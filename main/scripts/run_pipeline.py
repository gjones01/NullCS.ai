from __future__ import annotations

from pathlib import Path
import subprocess
import sys

PY = Path(r"C:\NullCS\NewAnubisTri\.venv\Scripts\python.exe")

# scripts to run (in order)
PARSE_SCRIPT = Path(r"C:\NullCS\main\scripts\parse_demos_awpy_api.py")
FEATURES_SCRIPT = Path(r"C:\NullCS\main\scripts\build_engagement_features.py")

def run_step(label: str, script: Path) -> None:
    if not script.exists():
        raise FileNotFoundError(f"{label} script not found: {script}")

    print(f"\n=== {label} ===")
    cmd = [str(PY), str(script)]
    print("[RUN]", " ".join(cmd))

    # stream output live
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {p.returncode}")

def main():
    # Step 1: parse demos -> parsed_zips (should skip already-parsed)
    run_step("PARSE DEMOS", PARSE_SCRIPT)

    # Step 2: build engagement features -> processed/demos/<demo>/engagement_features.parquet
    run_step("BUILD ENGAGEMENT FEATURES", FEATURES_SCRIPT)

    print("\n✅ Pipeline complete.")

if __name__ == "__main__":
    main()
