from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../main
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.adapters import cs2cd_adapter


OUT_ROOT = Path(r"C:\NullCS\main\data\processed\demos")
W_PRE = 128


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build engagement_features.parquet for CS2CD matches.")
    ap.add_argument(
        "--max-matches",
        type=int,
        default=None,
        help="Maximum matches per split. Example smoke test: --max-matches 5",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing CS2CD demo outputs.")
    return ap.parse_args()


def _first_shot_tick(shots: pd.DataFrame, attacker: str, round_num: int, kill_tick: int) -> float:
    g = shots[(shots["attacker_steamid"] == attacker) & (shots["round_num"] == round_num)]
    if g.empty:
        return np.nan
    g = g[(g["shot_tick"] <= kill_tick) & (g["shot_tick"] >= kill_tick - W_PRE)]
    if g.empty:
        return np.nan
    return float(g["shot_tick"].min())


def _shots_last_window(shots: pd.DataFrame, attacker: str, kill_tick: int, width: int) -> int:
    g = shots[(shots["attacker_steamid"] == attacker) & (shots["shot_tick"] <= kill_tick) & (shots["shot_tick"] >= kill_tick - width)]
    return int(len(g))


def build_match_rows(match: cs2cd_adapter.CS2CDMatch) -> pd.DataFrame:
    kills = match.kills.copy()
    if kills.empty:
        return pd.DataFrame()

    shots = match.shots.copy()
    if not shots.empty:
        # Reuse adapter round assignment for temporal consistency.
        starts = [int(x) for x in match.rounds["start_tick"].dropna().tolist()] if not match.rounds.empty else []
        shots["round_num"] = shots["shot_tick"].apply(lambda t: cs2cd_adapter._assign_round_num(int(t), starts))
    else:
        shots["round_num"] = pd.Series(dtype="int64")

    rows: list[dict[str, object]] = []
    for _, k in kills.iterrows():
        attacker = str(k["attacker_steamid"]).strip()
        victim = str(k["victim_steamid"]).strip()
        kill_tick = int(k["kill_tick"])
        round_num = int(k["round_num"])
        first_shot = _first_shot_tick(shots, attacker, round_num, kill_tick)

        label = 1 if attacker in match.cheater_ids else 0
        rows.append(
            {
                "demo_id": match.demo_id,
                "map_name": match.map_name,
                "round_num": round_num,
                "kill_tick": kill_tick,
                "t0_visible": np.nan,  # LoS is not available in CS2CD.
                "first_shot_tick": first_shot,
                "rt_ticks": float(kill_tick - first_shot) if pd.notna(first_shot) else np.nan,
                "attacker_steamid": attacker,
                "attacker_name": attacker,
                "victim_steamid": victim,
                "victim_name": victim,
                "weapon": str(k.get("weapon", "")),
                "headshot": bool(k.get("headshot", False)),
                "distance": float(k.get("distance", np.nan)),
                "visible_ticks_before_shot": np.nan,
                "visible_ticks_before_kill": np.nan,
                "shots_last64_before_kill": _shots_last_window(shots, attacker, kill_tick, 64),
                "shots_last128_before_kill": _shots_last_window(shots, attacker, kill_tick, 128),
                "is_micropeek_4": False,
                "is_micropeek_6": False,
                "is_micropeek_8": False,
                "is_thrusmoke": bool(k.get("is_thrusmoke", False)),
                "label": int(label),
                "label_source": "cs2cd_cheaters_json",
                "dataset_source": "cs2cd",
            }
        )

    return pd.DataFrame(rows)


def process_split(split: str, max_matches: int | None, overwrite: bool) -> tuple[int, int]:
    ids = cs2cd_adapter.list_match_ids(split)
    if max_matches is not None:
        ids = ids[:max_matches]

    built = 0
    skipped = 0
    for mid in ids:
        match = cs2cd_adapter.load_match(split, mid)
        demo_dir = OUT_ROOT / match.demo_id
        out_path = demo_dir / "engagement_features.parquet"
        meta_path = demo_dir / "meta.json"

        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        demo_dir.mkdir(parents=True, exist_ok=True)
        rows = build_match_rows(match)
        if rows.empty:
            skipped += 1
            continue
        rows.to_parquet(out_path, index=False)
        meta_path.write_text(
            (
                "{\n"
                f'  "demo_id": "{match.demo_id}",\n'
                f'  "split": "{split}",\n'
                f'  "match_id": "{mid}",\n'
                f'  "map_name": "{match.map_name}",\n'
                f'  "n_kills": {int(len(rows))},\n'
                f'  "n_cheaters_annotated": {int(len(match.cheater_ids))}\n'
                "}\n"
            ),
            encoding="utf-8",
        )
        built += 1

    return built, skipped


def main() -> int:
    args = parse_args()
    total_built = 0
    total_skipped = 0

    for split in cs2cd_adapter.SPLITS:
        built, skipped = process_split(split, args.max_matches, args.overwrite)
        total_built += built
        total_skipped += skipped
        print(f"[INFO] split={split} built={built} skipped={skipped}")

    print(f"[OK] CS2CD engagement build complete. built={total_built} skipped={total_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
