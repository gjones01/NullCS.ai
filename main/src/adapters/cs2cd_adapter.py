from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DATA_ROOT = Path(r"C:\NullCS\huggfacedata")
SPLITS = ("no_cheater_present", "with_cheater_present")


@dataclass
class CS2CDMatch:
    split: str
    match_id: str
    demo_id: str
    ticks_df: pd.DataFrame
    events_json: dict[str, Any]
    map_name: str
    cheater_ids: set[str]
    kills: pd.DataFrame
    shots: pd.DataFrame
    rounds: pd.DataFrame


def list_match_ids(split: str) -> list[str]:
    root = DATA_ROOT / split
    if not root.exists():
        return []
    ids = [p.stem for p in sorted(root.glob("*.parquet"))]
    return [i for i in ids if (root / f"{i}.json").exists()]


def _read_ticks(split: str, match_id: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_ROOT / split / f"{match_id}.parquet")


def _read_events(split: str, match_id: str) -> dict[str, Any]:
    p = DATA_ROOT / split / f"{match_id}.json"
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict JSON at {p}, got {type(obj).__name__}")
    return obj


def _map_name(events_json: dict[str, Any]) -> str:
    info = events_json.get("CSstats_info")
    if isinstance(info, list) and info and isinstance(info[0], dict):
        return str(info[0].get("map", "unknown_map"))
    return "unknown_map"


def _extract_cheaters(events_json: dict[str, Any]) -> set[str]:
    raw = events_json.get("cheaters")
    if not isinstance(raw, list):
        return set()
    out: set[str] = set()
    for row in raw:
        if isinstance(row, dict):
            sid = str(row.get("steamid", "")).strip()
            if sid:
                out.add(sid)
        elif isinstance(row, str):
            sid = row.strip()
            if sid:
                out.add(sid)
    return out


def _round_table(events_json: dict[str, Any]) -> pd.DataFrame:
    starts = events_json.get("round_freeze_end")
    ends = events_json.get("round_officially_ended")

    start_ticks = []
    if isinstance(starts, list):
        for row in starts:
            if isinstance(row, dict) and "tick" in row:
                start_ticks.append(int(row["tick"]))
    end_ticks = []
    if isinstance(ends, list):
        for row in ends:
            if isinstance(row, dict) and "tick" in row:
                end_ticks.append(int(row["tick"]))

    start_ticks = sorted(start_ticks)
    end_ticks = sorted(end_ticks)
    n = max(len(start_ticks), len(end_ticks))
    rows = []
    for i in range(n):
        rows.append(
            {
                "round_num": i + 1,
                "start_tick": start_ticks[i] if i < len(start_ticks) else None,
                "end_tick": end_ticks[i] if i < len(end_ticks) else None,
            }
        )
    return pd.DataFrame(rows)


def _assign_round_num(tick: int, round_starts: list[int]) -> int:
    if not round_starts:
        return 1
    idx = 0
    lo, hi = 0, len(round_starts)
    while lo < hi:
        mid = (lo + hi) // 2
        if round_starts[mid] <= tick:
            lo = mid + 1
        else:
            hi = mid
    idx = max(1, lo)
    return idx


def _kills_table(events_json: dict[str, Any], map_name: str) -> pd.DataFrame:
    rows = events_json.get("player_death")
    if not isinstance(rows, list):
        return pd.DataFrame(
            columns=[
                "kill_tick",
                "round_num",
                "attacker_steamid",
                "victim_steamid",
                "headshot",
                "weapon",
                "distance",
                "is_thrusmoke",
                "map_name",
            ]
        )

    rounds_df = _round_table(events_json)
    round_starts = [int(x) for x in rounds_df["start_tick"].dropna().tolist()] if not rounds_df.empty else []

    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        tick = row.get("tick")
        if tick is None:
            continue
        attacker = str(row.get("attacker_steamid", "")).strip()
        victim = str(row.get("user_steamid", "")).strip()
        if not attacker:
            continue
        out.append(
            {
                "kill_tick": int(tick),
                "round_num": _assign_round_num(int(tick), round_starts),
                "attacker_steamid": attacker,
                "victim_steamid": victim,
                "headshot": bool(row.get("headshot", False)),
                "weapon": str(row.get("weapon", "")),
                "distance": float(row.get("distance", float("nan"))),
                "is_thrusmoke": bool(row.get("thrusmoke", False)),
                "map_name": map_name,
            }
        )
    return pd.DataFrame(out)


def _shots_table(events_json: dict[str, Any]) -> pd.DataFrame:
    rows = events_json.get("weapon_fire")
    if not isinstance(rows, list):
        return pd.DataFrame(columns=["shot_tick", "attacker_steamid", "weapon"])
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        tick = row.get("tick")
        sid = str(row.get("user_steamid", "")).strip()
        weapon = str(row.get("weapon", "")).lower()
        if tick is None or not sid:
            continue
        # Exclude utility "shots" so first-shot logic stays firearm-focused.
        if "grenade" in weapon or weapon in {"weapon_flashbang", "weapon_smokegrenade", "weapon_hegrenade"}:
            continue
        out.append({"shot_tick": int(tick), "attacker_steamid": sid, "weapon": weapon})
    return pd.DataFrame(out)


def load_match(split: str, match_id: str) -> CS2CDMatch:
    ticks_df = _read_ticks(split, match_id)
    events_json = _read_events(split, match_id)
    map_name = _map_name(events_json)
    cheater_ids = _extract_cheaters(events_json)
    kills = _kills_table(events_json, map_name)
    shots = _shots_table(events_json)
    rounds = _round_table(events_json)
    demo_id = f"CS2CD_{split}_{match_id}"
    return CS2CDMatch(
        split=split,
        match_id=match_id,
        demo_id=demo_id,
        ticks_df=ticks_df,
        events_json=events_json,
        map_name=map_name,
        cheater_ids=cheater_ids,
        kills=kills,
        shots=shots,
        rounds=rounds,
    )
