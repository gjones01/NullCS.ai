from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
import os

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# Ensure "main/" is importable.
MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from src.parse import parse_demos_awpy_api as parse_mod
from src.features import build_engagement_features as build_mod
from src.features import aggregate_player_features as agg_mod
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Inference-only pipeline for one demo.")
    ap.add_argument("--dem_path", required=True, help="Path to input .dem file")
    ap.add_argument("--demo_id", required=True, help="Demo ID, e.g. TEST_YYYYMMDD_HHMMSS_abcd1234")
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Processed root output directory, e.g. C:\\ClarityCS\\main\\data\\processed",
    )
    ap.add_argument(
        "--model-artifact",
        default=None,
        help="Optional model artifact filename/path. Default: newest model in processed/models.",
    )
    return ap.parse_args()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _feature_vector_hash(row: pd.Series, feature_cols: list[str]) -> str:
    vals = []
    for c in feature_cols:
        v = row.get(c, np.nan)
        if pd.isna(v):
            vals.append(None)
        else:
            vals.append(float(v))
    payload = json.dumps(vals, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _high_tag_flags(row: pd.Series) -> dict[str, bool]:
    out = {}
    for c in ["prefire_pct", "thrusmoke_pct", "hs_pct", "long_fast_rt_pct"]:
        v = row.get(c, np.nan)
        out[c] = bool(pd.notna(v) and float(v) >= 0.90)
    return out


def _ensure_demo_copy(src_dem: Path, demo_id: str, raw_uploads_root: Path) -> Path:
    if src_dem.suffix.lower() != ".dem":
        raise ValueError(f"Input must be a .dem file: {src_dem}")
    if not src_dem.exists():
        raise FileNotFoundError(f"Demo file not found: {src_dem}")

    dst_dir = raw_uploads_root / demo_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_dem = dst_dir / f"{demo_id}.dem"
    if src_dem.resolve() != dst_dem.resolve():
        shutil.copy2(src_dem, dst_dem)
    return dst_dem


def _build_player_features_for_demo(engagement_path: Path, demo_id: str) -> pd.DataFrame:
    df = pd.read_parquet(engagement_path)
    if df.empty:
        raise RuntimeError(f"Engagement features are empty for demo: {demo_id}")

    if "attacker_steamid" not in df.columns:
        raise ValueError(f"Missing attacker_steamid in engagement features: {engagement_path}")

    df = df.copy()
    df["demo_id"] = demo_id
    df["label"] = -1  # placeholder for inference-only rows
    df["attacker_steamid"] = df["attacker_steamid"].astype(str).str.strip()
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

    key_cols = ["demo_id", "map_name", "attacker_steamid", "attacker_name", "label"]
    rows = [agg_mod.build_row(g) for _, g in df.groupby(key_cols, sort=False)]
    agg = pd.DataFrame(rows)
    agg = agg[agg["n_kills_with_rt"] >= agg_mod.MIN_KILLS].copy()
    if agg.empty:
        raise RuntimeError(f"No eligible players (MIN_KILLS filter) for demo: {demo_id}")

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
        "headshot_count",
        "fast_rt_count",
        "rt_le_2_count",
        "rt_le_4_count",
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
    return agg


def _infer_scores(
    player_df: pd.DataFrame,
    processed_root: Path,
    model_artifact: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    models_dir = processed_root / "models"
    reports_dir = processed_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    requested_model = model_artifact or os.environ.get("CLARITY_MODEL_ARTIFACT")
    model_path, feats_path = resolve_model_artifacts(models_dir, requested_model)
    model_hash = _sha256_file(model_path)
    feats_hash = _sha256_file(feats_path)

    feature_cols = [x for x in feats_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    ensure_no_forbidden_features(feature_cols, str(feats_path))
    for c in feature_cols:
        if c not in player_df.columns:
            player_df[c] = np.nan

    X = player_df[feature_cols].fillna(0.0).astype(float)
    model = XGBClassifier()
    model.load_model(str(model_path))
    scores = model.predict_proba(X)[:, 1]

    out = player_df.copy()
    out["model_artifact_path"] = str(model_path)
    out["model_artifact_sha256"] = model_hash
    out["feature_list_path"] = str(feats_path)
    out["feature_list_sha256"] = feats_hash
    out["feature_list_version"] = feats_hash[:12]
    out["proba_cheater_infer"] = scores
    out["confidence"] = compute_confidence_series(out)
    calibrator = load_calibrator()
    out["proba_calibrated"] = np.nan
    if calibrator is not None:
        out["proba_calibrated"] = maybe_calibrate(out["proba_cheater_infer"], calibrator)
    risk_base = out["proba_calibrated"].copy()
    miss = risk_base.isna()
    risk_base.loc[miss] = out.loc[miss, "proba_cheater_infer"].astype(float)
    out["risk"] = apply_rt_low_evidence_downweight(risk_base, out.get("rt_n", pd.Series([0] * len(out))))
    out["risk_band"] = risk_band_series(out["risk"])
    out["rt_reason_confidence"] = np.where(out.get("rt_n", pd.Series([0] * len(out))).fillna(0).astype(float) < 8, "low", "normal")
    out["ci_low"] = np.nan
    out["ci_high"] = np.nan
    out["top_reasons"] = out.apply(lambda r: json.dumps(top_signal_titles(r, top_k=3)), axis=1)
    out = out.sort_values("risk", ascending=False)

    trace_players: list[dict[str, object]] = []
    for _, r in out.iterrows():
        pct_cols = [c for c in out.columns if c.endswith("_pct")]
        z_cols = [c for c in out.columns if c.endswith("_z")]
        lobby_pct = {c: (None if pd.isna(r.get(c)) else float(r.get(c))) for c in pct_cols}
        lobby_z = {c: (None if pd.isna(r.get(c)) else float(r.get(c))) for c in z_cols}
        feat_row = {}
        for c in feature_cols:
            v = r.get(c, np.nan)
            feat_row[c] = None if pd.isna(v) else float(v)

        raw = float(r.get("proba_cheater_infer", 0.0))
        cal = None if pd.isna(r.get("proba_calibrated")) else float(r.get("proba_calibrated"))
        risk = float(r.get("risk", 0.0))
        rt_n = float(r.get("rt_n", 0.0) if pd.notna(r.get("rt_n")) else 0.0)
        nkrt = float(r.get("n_kills_with_rt", 0.0) if pd.notna(r.get("n_kills_with_rt")) else 0.0)
        n_players = float(r.get("n_players", 0.0) if pd.notna(r.get("n_players")) else 0.0)
        low_evidence_fired = bool(rt_n < 8.0)
        high_tags = _high_tag_flags(r)
        any_high_tag = any(high_tags.values())
        why_low = None
        if any_high_tag and risk < 0.20:
            reasons = []
            if low_evidence_fired:
                reasons.append("low RT evidence downweight fired (rt_n < 8)")
            if cal is not None and cal < raw:
                reasons.append("calibration reduced probability")
            if not reasons:
                reasons.append("model base probability is low despite high percentile tags")
            why_low = "; ".join(reasons)

        trace_players.append(
            {
                "steamid": str(r.get("attacker_steamid", "")),
                "attacker_name": str(r.get("attacker_name", "")),
                "model_artifact_path": str(model_path),
                "model_version": model_hash[:12],
                "model_sha256": model_hash,
                "feature_list_path": str(feats_path),
                "feature_list_version": feats_hash[:12],
                "feature_list_sha256": feats_hash,
                "feature_row": feat_row,
                "feature_vector_hash": _feature_vector_hash(r, feature_cols),
                "raw_proba": raw,
                "calibrated_proba": cal,
                "lobby_percentile": lobby_pct,
                "lobby_z": lobby_z,
                "risk_display_value": risk,
                "risk_display_value_pct": risk * 100.0,
                "confidence_value": float(r.get("confidence", 0.0)),
                "confidence_value_pct": float(r.get("confidence", 0.0)) * 100.0,
                "confidence_method": (
                    "compute_confidence = 0.45*log1p(n_kills_with_rt)/log1p(30)"
                    " + 0.35*log1p(rt_n)/log1p(30)"
                    " + 0.20*log1p(rounds_played)/log1p(24), clipped to [0,1]"
                ),
                "ci_p05": (None if pd.isna(r.get("ci_low")) else float(r.get("ci_low"))),
                "ci_p95": (None if pd.isna(r.get("ci_high")) else float(r.get("ci_high"))),
                "ci_method": "not computed in current inference path (ci_low/ci_high are NaN)",
                "gating_rules": {
                    "low_evidence_downweight_fired": low_evidence_fired,
                    "n_kills_with_rt_thresholding_applied": True,
                    "n_kills_with_rt_value": nkrt,
                    "n_players_filter_applied": False,
                    "n_players_filter_note": "n_players>=8 filter is used in offline training/eval, not single-demo inference",
                    "n_players_value": n_players,
                },
                "evidence_counts": {
                    "rt_n": rt_n,
                    "smoke_n": float(r.get("smoke_n", 0.0) if pd.notna(r.get("smoke_n")) else 0.0),
                    "hs_n": float(r.get("hs_n", 0.0) if pd.notna(r.get("hs_n")) else 0.0),
                    "prefire_n": float(r.get("prefire_count", 0.0) if pd.notna(r.get("prefire_count")) else 0.0),
                    "long_range_n": float(
                        r.get("long_range_kills_with_rt", 0.0) if pd.notna(r.get("long_range_kills_with_rt")) else 0.0
                    ),
                },
                "high_tag_flags": high_tags,
                "why_risk_low_despite_high_tags": why_low,
            }
        )

    debug_trace = {
        "demo_id": str(player_df["demo_id"].iloc[0]),
        "model_artifact_path": str(model_path),
        "model_sha256": model_hash,
        "feature_list_path": str(feats_path),
        "feature_list_sha256": feats_hash,
        "risk_formula": "risk = calibrated_proba if available else raw_proba; then downweight by 0.85 when rt_n < 8",
        "confidence_formula": (
            "0.45*log1p(n_kills_with_rt)/log1p(30) + "
            "0.35*log1p(rt_n)/log1p(30) + "
            "0.20*log1p(rounds_played)/log1p(24), clamp to [0,1]"
        ),
        "calibration_used": bool(calibrator is not None),
        "ci_method": "not computed in current inference path",
        "players": trace_players,
    }

    infer_csv = reports_dir / "ranked_player_demo_suspicion_infer.csv"
    if infer_csv.exists():
        old = pd.read_csv(infer_csv)
        old = old[old["demo_id"].astype(str) != str(player_df["demo_id"].iloc[0])]
        out_all = pd.concat([old, out], ignore_index=True)
        sort_col = "risk" if "risk" in out_all.columns else "proba_cheater_infer"
        out_all = out_all.sort_values(sort_col, ascending=False)
    else:
        out_all = out
    out_all.to_csv(infer_csv, index=False)
    print(f"[OK] wrote {infer_csv}")
    return out, debug_trace


def main() -> int:
    args = parse_args()
    dem_path = Path(args.dem_path)
    demo_id = str(args.demo_id).strip()
    processed_root = Path(args.out_dir)
    if not demo_id:
        raise ValueError("demo_id cannot be empty")

    project_root = MAIN_ROOT.parent  # C:\NullCS
    raw_uploads_root = project_root / "main" / "data" / "raw_uploads"
    parsed_zips_root = project_root / "parsed_zips"
    demos_root = processed_root / "demos"
    reports_root = processed_root / "reports"

    demo_file = _ensure_demo_copy(dem_path, demo_id, raw_uploads_root)
    print(f"[INFO] demo file: {demo_file}")

    parse_mod.OUT_ROOT = parsed_zips_root
    ok = parse_mod.parse_one(demo_file)
    if not ok:
        raise RuntimeError(f"Parse failed for demo: {demo_file}")

    zip_path = parsed_zips_root / f"{demo_id}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Parsed zip not found: {zip_path}")
    print(f"[INFO] parsed zip: {zip_path}")

    eng_df = build_mod.build_for_zip(zip_path)
    demo_dir = demos_root / demo_id
    demo_dir.mkdir(parents=True, exist_ok=True)
    eng_path = demo_dir / "engagement_features.parquet"
    eng_df.write_parquet(eng_path)
    print(f"[OK] wrote {eng_path}")

    player_df = _build_player_features_for_demo(eng_path, demo_id=demo_id)
    per_demo_player_path = demo_dir / "player_features_infer.parquet"
    player_df.to_parquet(per_demo_player_path, index=False)
    print(f"[OK] wrote {per_demo_player_path}")

    infer_ranked, debug_trace = _infer_scores(player_df, processed_root=processed_root, model_artifact=args.model_artifact)
    per_demo_report_dir = reports_root / demo_id
    per_demo_report_dir.mkdir(parents=True, exist_ok=True)
    per_demo_csv = per_demo_report_dir / "ranked_players_infer.csv"
    infer_ranked.to_csv(per_demo_csv, index=False)
    print(f"[OK] wrote {per_demo_csv}")
    debug_trace_path = per_demo_report_dir / "debug_score_trace.json"
    debug_trace_path.write_text(json.dumps(debug_trace, indent=2), encoding="utf-8")
    print(f"[OK] wrote {debug_trace_path}")

    manifest = {
        "demo_id": demo_id,
        "demo_file": str(demo_file),
        "zip_path": str(zip_path),
        "engagement_features": str(eng_path),
        "player_features_infer": str(per_demo_player_path),
        "ranked_players_infer": str(per_demo_csv),
        "debug_score_trace": str(debug_trace_path),
        "model_artifact_path": str(debug_trace.get("model_artifact_path", "")),
        "model_sha256": str(debug_trace.get("model_sha256", "")),
    }
    manifest_path = per_demo_report_dir / "infer_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
