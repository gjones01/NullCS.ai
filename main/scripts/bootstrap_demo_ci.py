from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../main
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.bootstrap_demo_ci import bootstrap_player_demo_ci


PROCESSED_ROOT = Path(r"C:\NullCS\main\data\processed")
REPORTS_ROOT = PROCESSED_ROOT / "reports"
DEMOS_ROOT = PROCESSED_ROOT / "demos"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Bootstrap CI for one player-demo risk prediction.")
    ap.add_argument("--demo", required=True)
    ap.add_argument("--steamid", required=True)
    ap.add_argument("--mode", choices=["oof", "insample", "infer"], default="infer")
    ap.add_argument("--n", type=int, default=100, help="Bootstrap iterations.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    demo_id = str(args.demo).strip()
    steamid = str(args.steamid).strip()

    ranked_path = REPORTS_ROOT / f"ranked_player_demo_suspicion_{args.mode}.csv"
    if not ranked_path.exists():
        raise FileNotFoundError(f"Missing ranked file: {ranked_path}")
    ranked = pd.read_csv(ranked_path)
    ranked["attacker_steamid"] = ranked["attacker_steamid"].astype(str).str.strip()
    row = ranked[
        (ranked["demo_id"].astype(str) == demo_id)
        & (ranked["attacker_steamid"] == steamid)
    ]
    if row.empty:
        raise ValueError(f"Row not found for demo={demo_id} steamid={steamid} in {ranked_path}")

    eng_path = DEMOS_ROOT / demo_id / "engagement_features.parquet"
    if not eng_path.exists():
        raise FileNotFoundError(f"Missing engagement features: {eng_path}")
    eng = pd.read_parquet(eng_path)
    if "demo_id" not in eng.columns:
        eng["demo_id"] = demo_id

    ci = bootstrap_player_demo_ci(
        demo_id=demo_id,
        steamid=steamid,
        ranked_row=row.iloc[0],
        engagement_df=eng,
        n_boot=int(args.n),
    )

    out = {
        "demo_id": demo_id,
        "steamid": steamid,
        "mode": args.mode,
        "n_boot": ci.n_boot,
        "risk_p05": ci.risk_p05,
        "risk_p50": ci.risk_p50,
        "risk_p95": ci.risk_p95,
        "ci_width": ci.ci_width,
    }
    out_dir = REPORTS_ROOT / demo_id / steamid
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ci.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

