from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
import os

import numpy as np
import pandas as pd
import polars as pl
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../main
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_engagement_features import build_for_zip
import src.features.aggregate_player_features as agg_mod
from src.utils.scoring import (
    ensure_no_forbidden_features,
    compute_confidence_series,
    apply_rt_low_evidence_downweight,
    load_calibrator,
    maybe_calibrate,
    risk_band_series,
    top_signal_titles,
)
from src.utils.model_registry import resolve_model_artifacts


PROCESSED_ROOT = Path(r"C:\NullCS\main\data\processed")
PARSE_ZIPS_ROOT = PROCESSED_ROOT / "parse_zips"
DEMOS_ROOT = PROCESSED_ROOT / "demos"
REPORTS_ROOT = PROCESSED_ROOT / "reports"
MODELS_ROOT = PROCESSED_ROOT / "models"


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Infer cheat suspicion for one raw .dem path.")
    ap.add_argument("--dem", required=True, help="Path to raw .dem file (no rename required).")
    ap.add_argument("--steamid", default=None, help="Optional Steam64 to explain. Defaults to top-1 in lobby.")
    ap.add_argument(
        "--model-artifact",
        default=None,
        help="Optional model artifact filename/path. Default: newest model in processed/models.",
    )
    return ap.parse_args()


def make_demo_id(dem_path: Path) -> str:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    h = hashlib.sha1(str(dem_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"TEST_{now}_{h}"


def write_df_to_parquet(df, path: Path) -> None:
    df.write_parquet(str(path))


def parse_dem_to_zip(dem_path: Path, zip_path: Path) -> None:
    from awpy import Demo

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    _safe_print(f"[PARSE] {dem_path}")
    demo = Demo(str(dem_path))
    demo.parse()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tables_written: list[str] = []

        def try_write(name: str, getter) -> None:
            try:
                df = getter()
                write_df_to_parquet(df, tmpdir / f"{name}.parquet")
                tables_written.append(f"{name}.parquet")
            except KeyError:
                _safe_print(f"  [SKIP] {name} (missing required event in demo)")
            except Exception as e:
                _safe_print(f"  [WARN] {name} failed: {e}")

        try_write("kills", lambda: demo.kills)
        try_write("damages", lambda: demo.damages)
        try_write("shots", lambda: demo.shots)
        try_write("grenades", lambda: demo.grenades)
        try_write("smokes", lambda: demo.smokes)
        try_write("infernos", lambda: demo.infernos)
        try_write("bomb", lambda: demo.bomb)
        try_write("ticks", lambda: demo.ticks)
        try_write("rounds", lambda: demo.rounds)
        try_write("footsteps", lambda: demo.footsteps)

        header_path = tmpdir / "header.json"
        header_obj = demo.header
        if hasattr(header_obj, "model_dump"):
            header_obj = header_obj.model_dump()
        elif hasattr(header_obj, "dict"):
            header_obj = header_obj.dict()
        header_path.write_text(json.dumps(header_obj, indent=2), encoding="utf-8")
        tables_written.append("header.json")

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fname in tables_written:
                zf.write(tmpdir / fname, arcname=fname)

    _safe_print(f"[OK] parsed zip: {zip_path}")


def build_engagement_for_demo(demo_id: str, zip_path: Path) -> Path:
    out_dir = DEMOS_ROOT / demo_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "engagement_features.parquet"

    _safe_print(f"[BUILD] engagement features for {demo_id}")
    eng = build_for_zip(zip_path)
    if "attacker_steamid" in eng.columns:
        eng = eng.with_columns(pl.col("attacker_steamid").cast(pl.Utf8))
    if "victim_steamid" in eng.columns:
        eng = eng.with_columns(pl.col("victim_steamid").cast(pl.Utf8))
    eng.write_parquet(out_path)
    _safe_print(f"[OK] wrote engagement: {out_path}")
    return out_path


def aggregate_single_demo_features(demo_id: str, eng_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(eng_path)
    if df.empty:
        raise RuntimeError(f"No engagement rows found for demo {demo_id}.")

    df = df.copy()
    df["demo_id"] = str(demo_id)
    df["label"] = 0
    df["attacker_steamid"] = df["attacker_steamid"].astype(str).str.strip()
    if "attacker_name" not in df.columns:
        df["attacker_name"] = ""

    for col, default in [
        ("map_name", ""),
        ("rt_ticks", np.nan),
        ("distance", np.nan),
        ("headshot", False),
        ("is_thrusmoke", False),
        ("round_num", np.nan),
        ("kill_tick", np.nan),
        ("victim_steamid", ""),
        ("weapon", ""),
    ]:
        if col not in df.columns:
            df[col] = default

    key_cols = ["demo_id", "map_name", "attacker_steamid", "attacker_name", "label"]
    rows = [agg_mod.build_row(g) for _, g in df.groupby(key_cols, sort=False)]
    agg = pd.DataFrame(rows)
    agg = agg[agg["n_kills_with_rt"] >= agg_mod.MIN_KILLS].copy()
    if agg.empty:
        raise RuntimeError(
            f"No players pass MIN_KILLS={agg_mod.MIN_KILLS} in demo {demo_id}. "
            "Cannot score with training-compatible features."
        )

    n_players = df.groupby("demo_id")["attacker_steamid"].nunique().rename("n_players")
    agg = agg.merge(n_players, on="demo_id", how="left")
    agg["kills_per_player"] = agg["n_kills"] / agg["n_players"].replace(0, np.nan)

    agg["fast_rt_rate"] = [agg_mod.laplace(s, n) for s, n in zip(agg["fast_rt_count"], agg["rt_n"])]
    agg["headshot_rate"] = [agg_mod.laplace(s, n) for s, n in zip(agg["headshot_count"], agg["hs_n"])]
    agg["rt_le_2_rate"] = [agg_mod.laplace(s, n) for s, n in zip(agg["rt_le_2_count"], agg["rt_n"])]
    agg["rt_le_4_rate"] = [agg_mod.laplace(s, n) for s, n in zip(agg["rt_le_4_count"], agg["rt_n"])]
    agg["prefire_rate"] = [agg_mod.laplace(s, n) for s, n in zip(agg["prefire_count"], agg["n_kills"])]
    agg["prefire_long_range_rate"] = [agg_mod.laplace(s, n) for s, n in zip(agg["prefire_long_range_count"], agg["n_kills"])]
    agg["prefire_repeat_victims"] = [agg_mod.laplace(s, n) for s, n in zip(agg["prefire_victim_n"], agg["n_victims"])]
    agg["thrusmoke_kill_rate"] = [agg_mod.laplace(s, n) for s, n in zip(agg["thrusmoke_kills"], agg["n_kills"])]
    agg["thrusmoke_round_rate"] = [agg_mod.laplace(s, n) for s, n in zip(agg["thrusmoke_rounds"], agg["rounds_played"])]
    agg["long_range_fast_rt_rate"] = [agg_mod.laplace(s, n) for s, n in zip(agg["long_range_fast_rt_count"], agg["long_range_kills_with_rt"])]
    agg["long_range_fast_rt_rate_4"] = [agg_mod.laplace(s, n) for s, n in zip(agg["long_range_fast_rt_4_count"], agg["long_range_kills_with_rt"])]

    agg["rifle_kill_share"] = [agg_mod.laplace(s, n) for s, n in zip(agg["rifle_kills"], agg["n_kills"])]
    agg["pistol_kill_share"] = [agg_mod.laplace(s, n) for s, n in zip(agg["pistol_kills"], agg["n_kills"])]
    agg["awp_smg_kill_share"] = [agg_mod.laplace(s, n) for s, n in zip(agg["awp_smg_kills"], agg["n_kills"])]

    agg["rifle_fast_rt_rate"] = [agg_mod.laplace(s, n) for s, n in zip(agg["rifle_fast_rt_count"], agg["rifle_kills"])]
    agg["pistol_fast_rt_rate"] = [agg_mod.laplace(s, n) for s, n in zip(agg["pistol_fast_rt_count"], agg["pistol_kills"])]
    agg["awp_smg_fast_rt_rate"] = [agg_mod.laplace(s, n) for s, n in zip(agg["awp_smg_fast_rt_count"], agg["awp_smg_kills"])]

    agg["prefire_rate_rifle"] = [agg_mod.laplace(s, n) for s, n in zip(agg["rifle_prefire_count"], agg["rifle_kills"])]
    agg["prefire_rate_pistol"] = [agg_mod.laplace(s, n) for s, n in zip(agg["pistol_prefire_count"], agg["pistol_kills"])]
    agg["prefire_rate_awp_smg"] = [agg_mod.laplace(s, n) for s, n in zip(agg["awp_smg_prefire_count"], agg["awp_smg_kills"])]

    agg["thrusmoke_rate_rifle"] = [agg_mod.laplace(s, n) for s, n in zip(agg["rifle_thrusmoke_count"], agg["rifle_kills"])]
    agg["thrusmoke_rate_pistol"] = [agg_mod.laplace(s, n) for s, n in zip(agg["pistol_thrusmoke_count"], agg["pistol_kills"])]
    agg["thrusmoke_rate_awp_smg"] = [agg_mod.laplace(s, n) for s, n in zip(agg["awp_smg_thrusmoke_count"], agg["awp_smg_kills"])]

    agg["prefire_rate_w"] = agg["prefire_rate"] * np.log1p(agg["rt_n"])
    agg["thrusmoke_rate_w"] = agg["thrusmoke_kill_rate"] * np.log1p(agg["n_kills"])
    agg["fast_rt_rate_w"] = agg["fast_rt_rate"] * np.log1p(agg["rt_n"])
    agg["rt_iqr_80"] = agg["rt_p90"] - agg["rt_p10"]
    agg["dist_tail"] = agg["dist_p90"] - agg["dist_median"]

    global_rt_median = float(agg["rt_median"].median(skipna=True))
    global_rt_p10 = float(agg["rt_p10"].median(skipna=True))
    global_rt_p90 = float(agg["rt_p90"].median(skipna=True))
    global_dist_median = float(agg["dist_median"].median(skipna=True))

    agg["rt_median_shrunk"] = (agg["rt_median"] * agg["rt_n"] + global_rt_median * agg_mod.SHRINK_K) / (agg["rt_n"] + agg_mod.SHRINK_K)
    agg["rt_p10_shrunk"] = (agg["rt_p10"] * agg["rt_n"] + global_rt_p10 * agg_mod.SHRINK_K) / (agg["rt_n"] + agg_mod.SHRINK_K)
    agg["rt_p90_shrunk"] = (agg["rt_p90"] * agg["rt_n"] + global_rt_p90 * agg_mod.SHRINK_K) / (agg["rt_n"] + agg_mod.SHRINK_K)
    agg["dist_median_shrunk"] = (agg["dist_median"] * agg["n_kills"] + global_dist_median * agg_mod.SHRINK_K) / (agg["n_kills"] + agg_mod.SHRINK_K)

    rt_derived = [
        "rt_mean", "rt_median", "rt_p10", "rt_p90", "rt_std",
        "rt_median_shrunk", "rt_p10_shrunk", "rt_p90_shrunk",
        "fast_rt_rate", "fast_rt_rate_w", "rt_le_2_rate", "rt_le_4_rate",
        "prefire_rate", "prefire_rate_w", "prefire_long_range_rate", "prefire_repeat_victims",
        "long_range_fast_rt_rate", "long_range_fast_rt_rate_4",
        "max_fast_rt_streak", "max_prefire_streak",
        "rifle_fast_rt_rate", "pistol_fast_rt_rate", "awp_smg_fast_rt_rate",
        "prefire_rate_rifle", "prefire_rate_pistol", "prefire_rate_awp_smg",
    ]
    low_evidence = agg["rt_n"] < agg_mod.MIN_RT_EVIDENCE
    for c in rt_derived:
        if c in agg.columns:
            agg.loc[low_evidence, c] = np.nan

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
            agg = agg_mod.add_demo_norms(agg, base_col, pct_col, z_col)

    helper_cols = [
        "headshot_count", "fast_rt_count", "rt_le_2_count", "rt_le_4_count",
        "prefire_count", "prefire_long_range_count", "prefire_victim_n",
        "long_range_fast_rt_count", "long_range_fast_rt_4_count",
        "rifle_fast_rt_count", "pistol_fast_rt_count", "awp_smg_fast_rt_count",
        "rifle_prefire_count", "pistol_prefire_count", "awp_smg_prefire_count",
        "rifle_thrusmoke_count", "pistol_thrusmoke_count", "awp_smg_thrusmoke_count",
    ]
    agg = agg.drop(columns=[c for c in helper_cols if c in agg.columns])
    return agg


def score_demo(agg: pd.DataFrame, model_artifact: str | None = None) -> pd.DataFrame:
    requested_model = model_artifact or os.environ.get("CLARITY_MODEL_ARTIFACT")
    model_path, features_path = resolve_model_artifacts(MODELS_ROOT, requested_model)
    feature_cols = features_path.read_text(encoding="utf-8").strip().splitlines()
    ensure_no_forbidden_features(feature_cols, str(features_path))
    for c in feature_cols:
        if c not in agg.columns:
            agg[c] = np.nan

    X = agg[feature_cols].fillna(0.0).astype(float)
    model = XGBClassifier()
    model.load_model(str(model_path))
    proba = model.predict_proba(X)[:, 1]

    ranked = agg.copy()
    ranked["attacker_steamid"] = ranked["attacker_steamid"].astype(str).str.strip()
    ranked["proba_cheater_infer"] = proba
    ranked["confidence"] = compute_confidence_series(ranked)
    calibrator = load_calibrator()
    ranked["proba_calibrated"] = np.nan
    if calibrator is not None:
        ranked["proba_calibrated"] = maybe_calibrate(ranked["proba_cheater_infer"], calibrator)
    risk_base = ranked["proba_calibrated"].copy()
    miss = risk_base.isna()
    risk_base.loc[miss] = ranked.loc[miss, "proba_cheater_infer"].astype(float)
    ranked["risk"] = apply_rt_low_evidence_downweight(risk_base, ranked.get("rt_n", pd.Series([0] * len(ranked))))
    ranked["risk_band"] = risk_band_series(ranked["risk"])
    ranked["rt_reason_confidence"] = np.where(ranked.get("rt_n", pd.Series([0] * len(ranked))).fillna(0).astype(float) < 8, "low", "normal")
    ranked["ci_low"] = np.nan
    ranked["ci_high"] = np.nan
    ranked["top_reasons"] = ranked.apply(lambda r: json.dumps(top_signal_titles(r, top_k=3)), axis=1)
    ranked = ranked.sort_values("risk", ascending=False).reset_index(drop=True)
    return ranked


def write_ranked_outputs(demo_id: str, ranked: pd.DataFrame) -> tuple[Path, Path]:
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    demo_report_dir = REPORTS_ROOT / demo_id
    demo_report_dir.mkdir(parents=True, exist_ok=True)

    global_ranked = REPORTS_ROOT / "ranked_player_demo_suspicion_infer.csv"
    demo_ranked = demo_report_dir / "ranked_player_demo_suspicion_infer.csv"
    ranked.to_csv(global_ranked, index=False)
    ranked.to_csv(demo_ranked, index=False)
    return global_ranked, demo_ranked


def run_explain(demo_id: str, steamid: str | None) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "explain_demo.py"), "--demo", demo_id, "--mode", "infer"]
    if steamid:
        cmd.extend(["--steamid", str(steamid).strip()])
    _safe_print(f"[EXPLAIN] {' '.join(cmd)}")
    res = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if res.stdout:
        print(res.stdout)
    if res.stderr:
        print(res.stderr)
    if res.returncode != 0:
        raise RuntimeError(f"Explain command failed with exit code {res.returncode}")


def main() -> int:
    args = parse_args()
    dem_path = Path(args.dem).expanduser().resolve()
    if not dem_path.exists() or dem_path.suffix.lower() != ".dem":
        raise FileNotFoundError(f"Expected an existing .dem file path, got: {dem_path}")

    demo_id = make_demo_id(dem_path)
    zip_path = PARSE_ZIPS_ROOT / f"{demo_id}.zip"

    parse_dem_to_zip(dem_path, zip_path)
    eng_path = build_engagement_for_demo(demo_id, zip_path)
    agg = aggregate_single_demo_features(demo_id, eng_path)
    ranked = score_demo(agg, model_artifact=args.model_artifact)
    global_ranked, demo_ranked = write_ranked_outputs(demo_id, ranked)

    _safe_print(f"[OK] wrote inference ranked CSV: {demo_ranked}")
    _safe_print(f"[OK] refreshed explain source CSV: {global_ranked}")
    run_explain(demo_id, args.steamid)
    _safe_print(f"[DONE] reports at: {REPORTS_ROOT / demo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
