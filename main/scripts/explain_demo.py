from __future__ import annotations
import sys
from pathlib import Path

# Ensure project root (main/) is on sys.path BEFORE importing src.*
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # ...\main
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from src.utils.explain_demo import default_config, explain_demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True, help="Demo ID like CDemo3, Normal012, Pro007")
    ap.add_argument("--steamid", default=None, help="Specific attacker_steamid to explain.")
    ap.add_argument("--name", default=None, help="Force a specific attacker_name (exact match)")
    ap.add_argument(
        "--mode",
        choices=["oof", "insample", "infer"],
        default="oof",
        help="Which ranked file to explain from.",
    )
    ap.add_argument("--ci", action="store_true", help="Compute bootstrap CI for this player-demo.")
    ap.add_argument("--n_boot", type=int, default=100, help="Bootstrap iterations when --ci is enabled.")

    args = ap.parse_args()

    cfg = default_config(mode=args.mode)
    explain_demo(cfg, args.demo, steamid=args.steamid, name=args.name, with_ci=args.ci, n_boot=args.n_boot)



if __name__ == "__main__":
    main()
