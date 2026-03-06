from __future__ import annotations

from pathlib import Path
import sys
import zipfile, io, json
import polars as pl

# Make src importable
SRC_DIR = Path(r"C:\NullCS\main\src")
sys.path.insert(0, str(SRC_DIR))

from utils.visibility_awpy import map_name_from_zip, is_visible, Point3

# ---------------- CONFIG ----------------
ZIPS_DIR = Path(r"C:\NullCS\parsed_zips")
OUT_ROOT = Path(r"C:\NullCS\main\data\processed\demos")

W_PRE = 128          # ticks before kill to search
REQ_CONSEC = 2       # require N consecutive visible ticks
EYE_Z = 64           # attacker eye height offset
CHEST_Z = 56         # victim target offset
LONG_RANGE_DIST = 1500.0

MAX_DEMOS = None  # set to 5 for a quick test; None = all
OVERWRITE = True     # overwrite per-demo output parquet
# ----------------------------------------


def first_visible_tick_los(
    ticks: pl.DataFrame,
    map_name: str,
    attacker_id: int,
    victim_id: int,
    kill_tick: int,
) -> int | None:
    start = max(0, kill_tick - W_PRE)

    # Filter to the time window and just the two players
    window = ticks.filter(
        (pl.col("tick") >= start) & (pl.col("tick") <= kill_tick)
        & (pl.col("steamid").is_in([attacker_id, victim_id]))
    ).select(["tick", "steamid", "X", "Y", "Z"]).sort(["tick", "steamid"])

    if window.is_empty():
        return None

    # Join attacker/victim positions per tick
    a = (
        window.filter(pl.col("steamid") == attacker_id)
        .rename({"X": "aX", "Y": "aY", "Z": "aZ"})
        .select(["tick", "aX", "aY", "aZ"])
    )
    v = (
        window.filter(pl.col("steamid") == victim_id)
        .rename({"X": "vX", "Y": "vY", "Z": "vZ"})
        .select(["tick", "vX", "vY", "vZ"])
    )

    tv = a.join(v, on="tick", how="inner").sort("tick")
    if tv.is_empty():
        return None

    consec = 0
    first_tick = None

    for row in tv.iter_rows(named=True):
        t = int(row["tick"])

        p1 = Point3(float(row["aX"]), float(row["aY"]), float(row["aZ"]) + EYE_Z)
        p2 = Point3(float(row["vX"]), float(row["vY"]), float(row["vZ"]) + CHEST_Z)

        vis = is_visible(map_name, p1, p2)
        if vis:
            consec += 1
            if first_tick is None:
                first_tick = t
            if consec >= REQ_CONSEC:
                return first_tick
        else:
            consec = 0
            first_tick = None

    return None


def first_shot_tick(
    shots: pl.DataFrame,
    attacker_id: int,
    start_tick: int,
    kill_tick: int
) -> int | None:
    s = shots.filter(
        (pl.col("player_steamid") == attacker_id)
        & (pl.col("tick") >= start_tick)
        & (pl.col("tick") <= kill_tick)
    ).select(["tick"]).sort("tick")

    if s.is_empty():
        return None
    return int(s.row(0)[0])


def load_parquet_from_zip(z: zipfile.ZipFile, name: str) -> pl.DataFrame:
    return pl.read_parquet(io.BytesIO(z.read(name)))


def build_for_zip(zip_path: Path) -> pl.DataFrame:
    demo_id = zip_path.stem
    map_name = map_name_from_zip(zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        ticks = load_parquet_from_zip(z, "ticks.parquet")
        kills = load_parquet_from_zip(z, "kills.parquet")
        shots = load_parquet_from_zip(z, "shots.parquet")

    rows = []
    for k in kills.iter_rows(named=True):
        a_raw = k.get("attacker_steamid")
        v_raw = k.get("victim_steamid")
        t_raw = k.get("tick")

        # skip weird/incomplete rows (world kills, bomb, fall, etc.)
        if a_raw is None or v_raw is None or t_raw is None:
            continue

        attacker = int(a_raw)
        victim = int(v_raw)
        kt = int(t_raw)
        round_num = int(k.get("round_num", -1))

        t0 = first_visible_tick_los(ticks, map_name, attacker, victim, kt)

        pre_start = max(0, kt - W_PRE)
        fs = first_shot_tick(shots, attacker, pre_start, kt)
        rt = None
        if t0 is not None and fs is not None:
            rt = int(fs - t0)

        # shots counts before kill (discipline vs spray/correction)
        shots_64 = shots.filter(
            (pl.col("player_steamid") == attacker)
            & (pl.col("tick") >= kt - 64)
            & (pl.col("tick") <= kt)
        ).height

        shots_128 = shots.filter(
            (pl.col("player_steamid") == attacker)
            & (pl.col("tick") >= kt - 128)
            & (pl.col("tick") <= kt)
        ).height

        vis_before_shot = (fs - t0) if (t0 is not None and fs is not None) else None
        vis_before_kill = (kt - t0) if (t0 is not None) else None

        is_mp4 = (vis_before_shot is not None and vis_before_shot <= 4)
        is_mp6 = (vis_before_shot is not None and vis_before_shot <= 6)
        is_mp8 = (vis_before_shot is not None and vis_before_shot <= 8)
        is_prefire = (rt is not None and rt <= -2)
        is_thrusmoke = bool(k.get("thrusmoke", False))
        is_long_range_fast_rt_4 = (
            rt is not None
            and k.get("distance") is not None
            and float(k.get("distance")) >= LONG_RANGE_DIST
            and int(rt) <= 4
        )



        rows.append(
            {
                "demo_id": demo_id,
                "map_name": map_name,
                "round_num": round_num,
                "kill_tick": kt,
                "t0_visible": t0,
                "first_shot_tick": fs,
                "rt_ticks": rt,
                "attacker_steamid": attacker,
                "attacker_name": k.get("attacker_name"),
                "victim_steamid": victim,
                "victim_name": k.get("victim_name"),
                "weapon": k.get("weapon"),
                "headshot": bool(k.get("headshot", False)),
                "distance": float(k.get("distance", 0.0)) if k.get("distance") is not None else None,
                "visible_ticks_before_shot": vis_before_shot,
                "visible_ticks_before_kill": vis_before_kill,
                "shots_last64_before_kill": shots_64,
                "shots_last128_before_kill": shots_128,
                "is_micropeek_4": is_mp4,
                "is_micropeek_6": is_mp6,
                "is_micropeek_8": is_mp8,
                "is_prefire": is_prefire,
                "is_thrusmoke": is_thrusmoke,
                "is_long_range_fast_rt_4": is_long_range_fast_rt_4,
            }
        )

    return pl.DataFrame(rows)


def main():
    zips = sorted(ZIPS_DIR.glob("*.zip"))
    if MAX_DEMOS is not None:
        zips = zips[: int(MAX_DEMOS)]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    ok = 0
    fail = 0

    for zp in zips:
        demo_id = zp.stem
        out_dir = OUT_ROOT / demo_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "engagement_features.parquet"

        if out_path.exists() and not OVERWRITE:
            print(f"[SKIP] {demo_id} -> already exists")
            continue

        try:
            print(f"[BUILD] {demo_id}")
            df = build_for_zip(zp)
            df.write_parquet(out_path)
            print(f"[OK]   {demo_id} -> {out_path}")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {demo_id}: {e}")
            fail += 1

    print(f"Done. OK={ok}, FAIL={fail}, out_root={OUT_ROOT}")


if __name__ == "__main__":
    main()
