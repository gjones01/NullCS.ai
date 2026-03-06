from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run inference twice on same demo and assert deterministic raw_proba/model.")
    ap.add_argument("--dem_path", required=True, help="Path to a .dem file")
    ap.add_argument("--out_dir", default=r"C:\NullCS\main\data\processed")
    ap.add_argument("--model-artifact", default=None, help="Optional model artifact filename/path")
    return ap.parse_args()


def run_once(dem_path: Path, out_dir: Path, demo_id: str, model_artifact: str | None) -> Path:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_infer_pipeline.py"),
        "--dem_path",
        str(dem_path),
        "--demo_id",
        demo_id,
        "--out_dir",
        str(out_dir),
    ]
    if model_artifact:
        cmd.extend(["--model-artifact", model_artifact])
    print("[RUN]", " ".join(cmd))
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"Inference run failed for {demo_id} with code {res.returncode}")
    return out_dir / "reports" / demo_id / "debug_score_trace.json"


def load_trace(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing debug trace: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def players_df(trace: dict) -> pd.DataFrame:
    rows = []
    for p in trace.get("players", []):
        rows.append(
            {
                "steamid": str(p.get("steamid", "")),
                "raw_proba": float(p.get("raw_proba", 0.0)),
                "model_artifact_path": str(p.get("model_artifact_path", "")),
                "model_sha256": str(p.get("model_sha256", "")),
            }
        )
    return pd.DataFrame(rows).sort_values("steamid").reset_index(drop=True)


def main() -> int:
    args = parse_args()
    dem_path = Path(args.dem_path).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not dem_path.exists():
        raise FileNotFoundError(f"Demo file not found: {dem_path}")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    demo_a = f"DET_{ts}_A"
    demo_b = f"DET_{ts}_B"

    trace_a_path = run_once(dem_path, out_dir, demo_a, args.model_artifact)
    trace_b_path = run_once(dem_path, out_dir, demo_b, args.model_artifact)

    ta = load_trace(trace_a_path)
    tb = load_trace(trace_b_path)
    da = players_df(ta)
    db = players_df(tb)

    merged = da.merge(db, on="steamid", suffixes=("_a", "_b"))
    if merged.empty:
        raise RuntimeError("No overlapping players across runs to compare.")

    same_model_path = (merged["model_artifact_path_a"] == merged["model_artifact_path_b"]).all()
    same_model_hash = (merged["model_sha256_a"] == merged["model_sha256_b"]).all()
    same_raw = (merged["raw_proba_a"].round(12) == merged["raw_proba_b"].round(12)).all()

    print(f"[CHECK] same_model_path={bool(same_model_path)}")
    print(f"[CHECK] same_model_sha256={bool(same_model_hash)}")
    print(f"[CHECK] same_raw_proba={bool(same_raw)}")

    if not (same_model_path and same_model_hash and same_raw):
        diffs = merged[
            (merged["model_artifact_path_a"] != merged["model_artifact_path_b"])
            | (merged["model_sha256_a"] != merged["model_sha256_b"])
            | (merged["raw_proba_a"].round(12) != merged["raw_proba_b"].round(12))
        ]
        print(diffs.head(20).to_string(index=False))
        raise AssertionError("Determinism check failed.")

    print("[OK] Determinism check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
