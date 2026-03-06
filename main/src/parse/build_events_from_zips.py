from __future__ import annotations

from pathlib import Path
import json
import zipfile
import io

import polars as pl

# =========================
# CONFIG (edit these only)
# =========================
ZIPS_DIR = Path(r"C:\NullCS\parsed_zips")
OUT_ROOT = Path(r"C:\NullCS\processed\demos")

MAX_ZIPS = None  # set to 5 for testing


# =========================
# HELPERS
# =========================
def read_parquet_from_zip(z: zipfile.ZipFile, name: str) -> pl.DataFrame:
    data = z.read(name)
    return pl.read_parquet(io.BytesIO(data))

def read_json_from_zip(z: zipfile.ZipFile, name: str) -> dict:
    return json.loads(z.read(name).decode("utf-8"))

def safe_get(d: dict, keys: list[str], default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def normalize_steamid(x) -> str | None:
    if x is None:
        return None
    return str(x)

def build_events_for_zip(zip_path: Path) -> bool:
    demo_id = zip_path.stem
    out_dir = OUT_ROOT / demo_id
    out_dir.mkdir(parents=True, exist_ok=True)

    events_path = out_dir / "events.parquet"
    meta_path = out_dir / "meta.json"

    # idempotent: skip if already built
    if events_path.exists() and events_path.stat().st_size > 0:
        return True

    with zipfile.ZipFile(zip_path, "r") as z:
        names = set(z.namelist())

        if "header.json" not in names:
            print(f"[SKIP] {zip_path.name} missing header.json")
            return False

        header = read_json_from_zip(z, "header.json")
        map_name = (
            header.get("map_name")
            or header.get("map")
            or safe_get(header, ["header", "map_name"])
            or safe_get(header, ["header", "map"])
            or None
        )

        # Kills are the core for v1
        if "kills.parquet" not in names:
            print(f"[WARN] {zip_path.name} has no kills.parquet (writing empty events)")
            empty = pl.DataFrame(
                {
                    "demo_id": pl.Series([], dtype=pl.Utf8),
                    "event_id": pl.Series([], dtype=pl.Int32),
                    "event_type": pl.Series([], dtype=pl.Utf8),
                    "tick": pl.Series([], dtype=pl.Int32),
                    "round_num": pl.Series([], dtype=pl.Int32),
                    "attacker_steamid": pl.Series([], dtype=pl.Utf8),
                    "victim_steamid": pl.Series([], dtype=pl.Utf8),
                    "is_headshot": pl.Series([], dtype=pl.Boolean),
                    "weapon": pl.Series([], dtype=pl.Utf8),
                    "map_name": pl.Series([], dtype=pl.Utf8),
                }
            )
            empty.write_parquet(str(events_path))
        else:
            kills = read_parquet_from_zip(z, "kills.parquet")

            # Common column name candidates (AWPy versions vary)
            tick_col = "tick" if "tick" in kills.columns else ("game_tick" if "game_tick" in kills.columns else None)
            round_col = "round_num" if "round_num" in kills.columns else ("round" if "round" in kills.columns else None)

            atk_col = None
            for c in ["attacker_steamid", "attackerSteamID", "attacker_steam_id", "attacker"]:
                if c in kills.columns:
                    atk_col = c
                    break

            vic_col = None
            for c in ["victim_steamid", "victimSteamID", "victim_steam_id", "victim"]:
                if c in kills.columns:
                    vic_col = c
                    break

            hs_col = None
            for c in ["is_headshot", "headshot", "isHeadshot", "hs"]:
                if c in kills.columns:
                    hs_col = c
                    break

            weapon_col = None
            for c in ["weapon", "weapon_name", "weaponName"]:
                if c in kills.columns:
                    weapon_col = c
                    break

            # Build canonical events
            df = kills

            # Create required columns, even if missing
            if tick_col is None:
                df = df.with_columns(pl.lit(None).cast(pl.Int32).alias("tick"))
            else:
                df = df.with_columns(pl.col(tick_col).cast(pl.Int32).alias("tick"))

            if round_col is None:
                df = df.with_columns(pl.lit(None).cast(pl.Int32).alias("round_num"))
            else:
                df = df.with_columns(pl.col(round_col).cast(pl.Int32).alias("round_num"))

            if atk_col is None:
                df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("attacker_steamid"))
            else:
                df = df.with_columns(pl.col(atk_col).cast(pl.Utf8).alias("attacker_steamid"))

            if vic_col is None:
                df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("victim_steamid"))
            else:
                df = df.with_columns(pl.col(vic_col).cast(pl.Utf8).alias("victim_steamid"))

            if hs_col is None:
                df = df.with_columns(pl.lit(None).cast(pl.Boolean).alias("is_headshot"))
            else:
                # some versions store 0/1
                df = df.with_columns(pl.col(hs_col).cast(pl.Int8).cast(pl.Boolean).alias("is_headshot"))

            if weapon_col is None:
                df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("weapon"))
            else:
                df = df.with_columns(pl.col(weapon_col).cast(pl.Utf8).alias("weapon"))

            df = df.with_columns(
                pl.lit(demo_id).alias("demo_id"),
                pl.lit("kill").alias("event_type"),
                pl.lit(map_name).cast(pl.Utf8).alias("map_name"),
            )

            # event_id = row index
            df = df.with_row_index(name="event_id")

            # Keep only canonical columns for now
            events = df.select([
                "demo_id",
                "event_id",
                "event_type",
                "tick",
                "round_num",
                "attacker_steamid",
                "victim_steamid",
                "is_headshot",
                "weapon",
                "map_name",
            ])

            events.write_parquet(str(events_path))

        # meta.json (store whatever you want)
        meta = {
            "demo_id": demo_id,
            "map_name": map_name,
            "zip_file": zip_path.name,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] {demo_id} -> {events_path}")
    return True


def main():
    zips = sorted(ZIPS_DIR.glob("*.zip"))
    if MAX_ZIPS is not None:
        zips = zips[:MAX_ZIPS]

    if not zips:
        print(f"No zips found in: {ZIPS_DIR}")
        return 1

    ok = 0
    fail = 0
    for i, zp in enumerate(zips, start=1):
        print(f"[{i}/{len(zips)}] {zp.name}")
        if build_events_for_zip(zp):
            ok += 1
        else:
            fail += 1

    print(f"Done. OK={ok}, FAIL={fail}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
