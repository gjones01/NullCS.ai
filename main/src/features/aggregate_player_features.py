from __future__ import annotations

from pathlib import Path
import re
import math
import numpy as np
import pandas as pd

# -------- CONFIG --------
IN_ROOT = Path(r"C:\NullCS\main\data\processed\demos")
OUT_PATH = Path(r"C:\NullCS\main\data\processed\player_features.parquet")
CHEATER_CSV = Path(r"C:\NullCS\main\data\processed\CheaterSteamIDs.csv")

MIN_KILLS = 5
FAST_RT_TICKS = 8
LONG_RANGE_DIST = 1500.0
LAPLACE_ALPHA = 1.0
SHRINK_K = 10.0
MIN_RT_EVIDENCE = 8
OVERWRITE = True
# ------------------------


def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def _split_steamids(raw: str) -> list[str]:
    # Support multiple IDs in one cell, e.g. "id1;id2|id3 id4".
    parts = re.split(r"[;,|\s]+", str(raw).strip())
    return [p.strip() for p in parts if p and p.strip()]


def load_cheater_map(csv_path: Path) -> dict[str, set[str]]:
    if not csv_path.exists():
        print(f"[WARN] Cheater CSV not found: {csv_path}")
        return {}

    df = pd.read_csv(csv_path, dtype=str)
    cols = {c.lower().strip(): c for c in df.columns}
    demo_col = cols.get("cdemo id") or cols.get("demo id") or cols.get("demo_id")
    id_col = cols.get("name/id") or cols.get("steamid") or cols.get("steam_id") or cols.get("attacker_steamid")
    id_cols: list[str] = []
    if id_col is not None:
        id_cols.append(id_col)
    for lc, orig in cols.items():
        if lc.startswith("steamid") or lc.startswith("steam_id") or lc.startswith("cheater_steamid"):
            if orig not in id_cols:
                id_cols.append(orig)

    if demo_col is None or not id_cols:
        raise ValueError(f"Cheater CSV must have demo+steamid columns. Found: {list(df.columns)}")

    df = df[[demo_col, *id_cols]].copy()
    df = df.rename(columns={demo_col: "demo_id"})
    df["demo_id"] = df["demo_id"].astype(str).str.strip()
    df = df[df["demo_id"] != ""]

    exploded = []
    for _, row in df.iterrows():
        demo_id = str(row["demo_id"]).strip()
        for col in id_cols:
            for sid in _split_steamids(row[col]):
                exploded.append((demo_id, sid))

    if not exploded:
        print(f"[WARN] Cheater CSV has no usable demo/steamid rows: {csv_path}")
        return {}

    edf = pd.DataFrame(exploded, columns=["demo_id", "cheater_steamid"]).drop_duplicates()
    mp = edf.groupby("demo_id")["cheater_steamid"].apply(set).to_dict()
    print(f"[INFO] loaded cheater map demos: {len(mp)} rows: {len(edf)} steamid_cols: {len(id_cols)} from {csv_path}")
    return mp


def demo_base_label(demo_id: str) -> int | None:
    d = demo_id.lower()
    if d.startswith("pro") or d.startswith("normal"):
        return 0
    if d.startswith("cdemo"):
        return None
    return None


def weapon_family(weapon: object) -> str:
    w = str(weapon).lower().strip()
    rifles = {"ak47", "m4a1", "m4a1_silencer", "famas", "galilar", "aug", "sg556"}
    pistols = {"glock", "hkp2000", "usp_silencer", "p250", "elite", "fiveseven", "tec9", "cz75a", "deagle", "revolver"}
    smgs = {"mac10", "mp9", "mp7", "mp5sd", "ump45", "p90", "bizon"}
    awp_smg = {"awp", "ssg08", *smgs}
    if w in rifles:
        return "rifle"
    if w in pistols:
        return "pistol"
    if w in awp_smg:
        return "awp_smg"
    return "other"


def laplace(success: float, trials: float, alpha: float = LAPLACE_ALPHA) -> float:
    if pd.isna(success) or pd.isna(trials):
        return np.nan
    return float((success + alpha) / (trials + (2.0 * alpha)))


def longest_true_streak(bools: list[bool]) -> int:
    best = 0
    cur = 0
    for b in bools:
        if b:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def longest_consecutive_int_streak(vals: list[int]) -> int:
    if not vals:
        return 0
    vals = sorted(set(vals))
    best = 1
    cur = 1
    for i in range(1, len(vals)):
        if vals[i] == vals[i - 1] + 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def build_row(g: pd.DataFrame) -> dict[str, object]:
    g = g.sort_values("kill_tick")

    rt = pd.to_numeric(g["rt_ticks"], errors="coerce")
    rt_valid = rt.dropna()
    rt_n = int(rt_valid.shape[0])
    n_kills = int(len(g))

    prefire_mask = (rt <= -2).fillna(False)
    thrusmoke_mask = g["is_thrusmoke"].fillna(False).astype(bool)
    hs_mask = g["headshot"].fillna(False).astype(bool)
    fast_rt_mask = (rt <= FAST_RT_TICKS).fillna(False)
    rt_le_2_mask = (rt <= 2).fillna(False)
    rt_le_4_mask = (rt <= 4).fillna(False)

    dist = pd.to_numeric(g["distance"], errors="coerce")
    long_range_mask = (dist >= LONG_RANGE_DIST) & rt.notna()
    long_fast_mask = long_range_mask & (rt <= FAST_RT_TICKS)
    long_fast_4_mask = long_range_mask & (rt <= 4)
    prefire_long_range_mask = prefire_mask & (dist >= LONG_RANGE_DIST)

    rounds = pd.to_numeric(g["round_num"], errors="coerce")
    rounds_played = int(rounds.dropna().nunique())

    thr_rounds = rounds[thrusmoke_mask].dropna().astype(int).tolist()
    thrusmoke_rounds = len(set(thr_rounds))
    max_thr_round_streak = longest_consecutive_int_streak(thr_rounds)

    thr_per_round = (
        g.assign(_thr=thrusmoke_mask.values)
        .groupby("round_num", dropna=True)["_thr"]
        .sum()
    )
    thrusmoke_repeat_rounds = int((thr_per_round >= 2).sum())

    victims = g["victim_steamid"].astype(str).str.strip()
    victim_n = int(victims.nunique())
    prefire_victim_n = int(victims[prefire_mask].nunique())

    wf = g["weapon"].apply(weapon_family)

    def fam_counts(f: str) -> tuple[int, int, int, int]:
        m = wf == f
        return (
            int(m.sum()),
            int((m & fast_rt_mask).sum()),
            int((m & prefire_mask).sum()),
            int((m & thrusmoke_mask).sum()),
        )

    rifle_n, rifle_fast_n, rifle_prefire_n, rifle_thr_n = fam_counts("rifle")
    pistol_n, pistol_fast_n, pistol_prefire_n, pistol_thr_n = fam_counts("pistol")
    awp_smg_n, awp_smg_fast_n, awp_smg_prefire_n, awp_smg_thr_n = fam_counts("awp_smg")

    row = {
        "demo_id": g["demo_id"].iloc[0],
        "map_name": g["map_name"].iloc[0],
        "attacker_steamid": g["attacker_steamid"].iloc[0],
        "attacker_name": g["attacker_name"].iloc[0],
        "label": int(g["label"].iloc[0]),
        "n_kills": n_kills,
        "n_kills_with_rt": rt_n,
        "rt_n": rt_n,
        "hs_n": n_kills,
        "smoke_n": n_kills,
        "rounds_played": rounds_played,
        "n_victims": victim_n,
        "rt_mean": float(rt_valid.mean()) if rt_n else np.nan,
        "rt_median": float(rt_valid.median()) if rt_n else np.nan,
        "rt_p10": float(rt_valid.quantile(0.10, interpolation="nearest")) if rt_n else np.nan,
        "rt_p90": float(rt_valid.quantile(0.90, interpolation="nearest")) if rt_n else np.nan,
        "rt_std": float(rt_valid.std(ddof=1)) if rt_n > 1 else np.nan,
        "dist_mean": float(dist.mean()) if n_kills else np.nan,
        "dist_median": float(dist.median()) if n_kills else np.nan,
        "dist_p90": float(dist.quantile(0.90, interpolation="nearest")) if n_kills else np.nan,
        "weapon_n_unique": int(g["weapon"].astype(str).nunique()),
        "headshot_count": int(hs_mask.sum()),
        "fast_rt_count": int(fast_rt_mask.sum()),
        "rt_le_2_count": int(rt_le_2_mask.sum()),
        "rt_le_4_count": int(rt_le_4_mask.sum()),
        "prefire_count": int(prefire_mask.sum()),
        "prefire_long_range_count": int(prefire_long_range_mask.sum()),
        "prefire_victim_n": prefire_victim_n,
        "thrusmoke_kills": int(thrusmoke_mask.sum()),
        "thrusmoke_rounds": int(thrusmoke_rounds),
        "thrusmoke_repeat_rounds": int(thrusmoke_repeat_rounds),
        "max_thrusmoke_streak": int(longest_true_streak(thrusmoke_mask.tolist())),
        "max_thrusmoke_round_streak": int(max_thr_round_streak),
        "long_range_kills_with_rt": int(long_range_mask.sum()),
        "long_range_fast_rt_count": int(long_fast_mask.sum()),
        "long_range_fast_rt_4_count": int(long_fast_4_mask.sum()),
        "max_fast_rt_streak": int(longest_true_streak((rt <= 4).fillna(False).tolist())),
        "max_headshot_streak": int(longest_true_streak(hs_mask.tolist())),
        "max_prefire_streak": int(longest_true_streak(prefire_mask.tolist())),
        "rifle_kills": rifle_n,
        "pistol_kills": pistol_n,
        "awp_smg_kills": awp_smg_n,
        "rifle_fast_rt_count": rifle_fast_n,
        "pistol_fast_rt_count": pistol_fast_n,
        "awp_smg_fast_rt_count": awp_smg_fast_n,
        "rifle_prefire_count": rifle_prefire_n,
        "pistol_prefire_count": pistol_prefire_n,
        "awp_smg_prefire_count": awp_smg_prefire_n,
        "rifle_thrusmoke_count": rifle_thr_n,
        "pistol_thrusmoke_count": pistol_thr_n,
        "awp_smg_thrusmoke_count": awp_smg_thr_n,
    }
    return row


def add_demo_norms(df: pd.DataFrame, base_col: str, pct_col: str, z_col: str) -> pd.DataFrame:
    grp = df.groupby("demo_id")[base_col]
    mean = grp.transform("mean")
    std = grp.transform("std")
    rank = grp.rank(method="average", pct=True)
    df[pct_col] = rank
    df[z_col] = (df[base_col] - mean) / std.replace(0, np.nan)
    return df


def main():
    cheater_map = load_cheater_map(CHEATER_CSV)

    files = sorted(IN_ROOT.glob(r"*/engagement_features.parquet"))
    if not files:
        raise FileNotFoundError(f"No engagement_features.parquet found under {IN_ROOT}")

    print(f"[INFO] found {len(files)} engagement files")

    dfs: list[pd.DataFrame] = []
    for f in files:
        demo_id = f.parent.name
        df = pd.read_parquet(f)
        if df.empty:
            continue

        if "attacker_steamid" not in df.columns:
            print(f"[WARN] {demo_id}: missing attacker_steamid, skipping")
            continue

        base = demo_base_label(demo_id)
        is_cd = demo_id.lower().startswith("cdemo")

        df = df.copy()
        df["demo_id"] = demo_id
        sid = df["attacker_steamid"].astype(str).str.strip()

        has_inline_label = "label" in df.columns and df["label"].notna().any()
        if has_inline_label:
            df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
        elif is_cd:
            cheater_ids = cheater_map.get(demo_id)
            if not cheater_ids:
                print(f"[WARN] {demo_id}: no cheater SteamID found in {CHEATER_CSV}. Skipping this demo.")
                continue
            df["label"] = sid.isin(cheater_ids).astype(int)
        else:
            if base is None:
                print(f"[WARN] {demo_id}: unknown naming. Skipping.")
                continue
            df["label"] = int(base)

        df["attacker_steamid"] = sid
        if "attacker_name" not in df.columns:
            df["attacker_name"] = ""

        for col, default in [
            ("rt_ticks", np.nan),
            ("distance", np.nan),
            ("headshot", False),
            ("is_thrusmoke", False),
            ("round_num", np.nan),
            ("kill_tick", np.nan),
            ("victim_steamid", ""),
        ]:
            if col not in df.columns:
                df[col] = default

        dfs.append(df)

    if not dfs:
        raise RuntimeError("No demos loaded after labeling. Check paths and CheaterSteamIDs.csv.")

    all_kills = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] total kill rows: {len(all_kills)}")

    key_cols = ["demo_id", "map_name", "attacker_steamid", "attacker_name", "label"]
    rows = [build_row(g) for _, g in all_kills.groupby(key_cols, sort=False)]
    agg = pd.DataFrame(rows)

    agg = agg[agg["n_kills_with_rt"] >= MIN_KILLS].copy()

    # Demo context columns.
    n_players = all_kills.groupby("demo_id")["attacker_steamid"].nunique().rename("n_players")
    agg = agg.merge(n_players, on="demo_id", how="left")
    agg["kills_per_player"] = agg["n_kills"] / agg["n_players"].replace(0, np.nan)

    # Smoothed rates.
    agg["fast_rt_rate"] = [laplace(s, n) for s, n in zip(agg["fast_rt_count"], agg["rt_n"])]
    agg["headshot_rate"] = [laplace(s, n) for s, n in zip(agg["headshot_count"], agg["hs_n"])]
    agg["rt_le_2_rate"] = [laplace(s, n) for s, n in zip(agg["rt_le_2_count"], agg["rt_n"])]
    agg["rt_le_4_rate"] = [laplace(s, n) for s, n in zip(agg["rt_le_4_count"], agg["rt_n"])]
    agg["prefire_rate"] = [laplace(s, n) for s, n in zip(agg["prefire_count"], agg["n_kills"])]
    agg["prefire_long_range_rate"] = [laplace(s, n) for s, n in zip(agg["prefire_long_range_count"], agg["n_kills"])]
    agg["prefire_repeat_victims"] = [laplace(s, n) for s, n in zip(agg["prefire_victim_n"], agg["n_victims"])]
    agg["thrusmoke_kill_rate"] = [laplace(s, n) for s, n in zip(agg["thrusmoke_kills"], agg["n_kills"])]
    agg["thrusmoke_round_rate"] = [laplace(s, n) for s, n in zip(agg["thrusmoke_rounds"], agg["rounds_played"])]
    agg["long_range_fast_rt_rate"] = [laplace(s, n) for s, n in zip(agg["long_range_fast_rt_count"], agg["long_range_kills_with_rt"])]
    agg["long_range_fast_rt_rate_4"] = [laplace(s, n) for s, n in zip(agg["long_range_fast_rt_4_count"], agg["long_range_kills_with_rt"])]

    agg["rifle_kill_share"] = [laplace(s, n) for s, n in zip(agg["rifle_kills"], agg["n_kills"])]
    agg["pistol_kill_share"] = [laplace(s, n) for s, n in zip(agg["pistol_kills"], agg["n_kills"])]
    agg["awp_smg_kill_share"] = [laplace(s, n) for s, n in zip(agg["awp_smg_kills"], agg["n_kills"])]

    agg["rifle_fast_rt_rate"] = [laplace(s, n) for s, n in zip(agg["rifle_fast_rt_count"], agg["rifle_kills"])]
    agg["pistol_fast_rt_rate"] = [laplace(s, n) for s, n in zip(agg["pistol_fast_rt_count"], agg["pistol_kills"])]
    agg["awp_smg_fast_rt_rate"] = [laplace(s, n) for s, n in zip(agg["awp_smg_fast_rt_count"], agg["awp_smg_kills"])]

    agg["prefire_rate_rifle"] = [laplace(s, n) for s, n in zip(agg["rifle_prefire_count"], agg["rifle_kills"])]
    agg["prefire_rate_pistol"] = [laplace(s, n) for s, n in zip(agg["pistol_prefire_count"], agg["pistol_kills"])]
    agg["prefire_rate_awp_smg"] = [laplace(s, n) for s, n in zip(agg["awp_smg_prefire_count"], agg["awp_smg_kills"])]

    agg["thrusmoke_rate_rifle"] = [laplace(s, n) for s, n in zip(agg["rifle_thrusmoke_count"], agg["rifle_kills"])]
    agg["thrusmoke_rate_pistol"] = [laplace(s, n) for s, n in zip(agg["pistol_thrusmoke_count"], agg["pistol_kills"])]
    agg["thrusmoke_rate_awp_smg"] = [laplace(s, n) for s, n in zip(agg["awp_smg_thrusmoke_count"], agg["awp_smg_kills"])]

    # Weighted rates.
    agg["prefire_rate_w"] = agg["prefire_rate"] * np.log1p(agg["rt_n"])
    agg["thrusmoke_rate_w"] = agg["thrusmoke_kill_rate"] * np.log1p(agg["n_kills"])
    agg["fast_rt_rate_w"] = agg["fast_rt_rate"] * np.log1p(agg["rt_n"])

    # Derived tails.
    agg["rt_iqr_80"] = agg["rt_p90"] - agg["rt_p10"]
    agg["dist_tail"] = agg["dist_p90"] - agg["dist_median"]

    # Shrinkage.
    global_rt_median = float(agg["rt_median"].median(skipna=True))
    global_rt_p10 = float(agg["rt_p10"].median(skipna=True))
    global_rt_p90 = float(agg["rt_p90"].median(skipna=True))
    global_dist_median = float(agg["dist_median"].median(skipna=True))

    agg["rt_median_shrunk"] = (agg["rt_median"] * agg["rt_n"] + global_rt_median * SHRINK_K) / (agg["rt_n"] + SHRINK_K)
    agg["rt_p10_shrunk"] = (agg["rt_p10"] * agg["rt_n"] + global_rt_p10 * SHRINK_K) / (agg["rt_n"] + SHRINK_K)
    agg["rt_p90_shrunk"] = (agg["rt_p90"] * agg["rt_n"] + global_rt_p90 * SHRINK_K) / (agg["rt_n"] + SHRINK_K)
    agg["dist_median_shrunk"] = (agg["dist_median"] * agg["n_kills"] + global_dist_median * SHRINK_K) / (agg["n_kills"] + SHRINK_K)

    # Min evidence gating for RT-derived features.
    rt_derived = [
        "rt_mean",
        "rt_median",
        "rt_p10",
        "rt_p90",
        "rt_std",
        "rt_median_shrunk",
        "rt_p10_shrunk",
        "rt_p90_shrunk",
        "fast_rt_rate",
        "fast_rt_rate_w",
        "rt_le_2_rate",
        "rt_le_4_rate",
        "prefire_rate",
        "prefire_rate_w",
        "prefire_long_range_rate",
        "prefire_repeat_victims",
        "long_range_fast_rt_rate",
        "long_range_fast_rt_rate_4",
        "max_fast_rt_streak",
        "max_prefire_streak",
        "rifle_fast_rt_rate",
        "pistol_fast_rt_rate",
        "awp_smg_fast_rt_rate",
        "prefire_rate_rifle",
        "prefire_rate_pistol",
        "prefire_rate_awp_smg",
    ]
    low_evidence = agg["rt_n"] < MIN_RT_EVIDENCE
    for c in rt_derived:
        if c in agg.columns:
            agg.loc[low_evidence, c] = np.nan

    # Within-demo normalization.
    norm_map = {
        "rt_median": ("rt_median_pct", "rt_median_z"),
        "prefire_rate": ("prefire_pct", "prefire_z"),
        "thrusmoke_kill_rate": ("thrusmoke_pct", "thrusmoke_z"),
        "headshot_rate": ("hs_pct", "hs_z"),
        "long_range_fast_rt_rate_4": ("long_fast_rt_pct", "long_fast_rt_z"),
        "max_thrusmoke_round_streak": ("max_thr_round_streak_pct", "max_thr_round_streak_z"),
        "dist_tail": ("dist_tail_pct", "dist_tail_z"),
    }
    for base_col, (pct_col, z_col) in norm_map.items():
        if base_col in agg.columns:
            agg = add_demo_norms(agg, base_col, pct_col, z_col)

    helper_cols = [
        "headshot_count",
        "fast_rt_count",
        "rt_le_2_count",
        "rt_le_4_count",
        "prefire_count",
        "prefire_long_range_count",
        "prefire_victim_n",
        "long_range_fast_rt_count",
        "long_range_fast_rt_4_count",
        "rifle_fast_rt_count",
        "pistol_fast_rt_count",
        "awp_smg_fast_rt_count",
        "rifle_prefire_count",
        "pistol_prefire_count",
        "awp_smg_prefire_count",
        "rifle_thrusmoke_count",
        "pistol_thrusmoke_count",
        "awp_smg_thrusmoke_count",
    ]
    agg = agg.drop(columns=[c for c in helper_cols if c in agg.columns])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists() and not OVERWRITE:
        raise FileExistsError(f"{OUT_PATH} exists and OVERWRITE=False")

    agg.to_parquet(OUT_PATH, index=False)
    print(f"[OK] wrote: {OUT_PATH}")
    print(f"[INFO] rows (player-demo): {len(agg)}")
    safe_print(f"[INFO] label counts:\n{agg.groupby('label').size()}")


if __name__ == "__main__":
    main()
