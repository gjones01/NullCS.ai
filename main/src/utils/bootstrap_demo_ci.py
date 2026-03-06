from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../main
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.features import aggregate_player_features as agg_mod
from src.utils.scoring import ensure_no_forbidden_features, load_calibrator


MODELS_ROOT = Path(r"C:\NullCS\main\data\processed\models")
FEATURES_PATH = MODELS_ROOT / "xgb_player_level_features.txt"
MODEL_PATH = MODELS_ROOT / "xgb_player_level_gridcv.json"


@dataclass
class BootstrapCIResult:
    risk_p05: float
    risk_p50: float
    risk_p95: float
    ci_width: float
    n_boot: int


def _aggregate_one_player_from_events(events: pd.DataFrame, demo_id: str, player_row: pd.Series) -> pd.DataFrame:
    d = events.copy()
    d["demo_id"] = demo_id
    d["label"] = int(player_row.get("label", 0))
    d["attacker_steamid"] = str(player_row.get("attacker_steamid", "")).strip()
    d["attacker_name"] = str(player_row.get("attacker_name", ""))

    for col, default in [
        ("map_name", player_row.get("map_name", "")),
        ("rt_ticks", np.nan),
        ("distance", np.nan),
        ("headshot", False),
        ("is_thrusmoke", False),
        ("round_num", np.nan),
        ("kill_tick", np.nan),
        ("victim_steamid", ""),
        ("weapon", ""),
    ]:
        if col not in d.columns:
            d[col] = default

    key_cols = ["demo_id", "map_name", "attacker_steamid", "attacker_name", "label"]
    rows = [agg_mod.build_row(g) for _, g in d.groupby(key_cols, sort=False)]
    agg = pd.DataFrame(rows)

    n_players = max(1, int(player_row.get("n_players", 10)))
    agg["n_players"] = n_players
    agg["kills_per_player"] = agg["n_kills"] / n_players

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

    # For bootstrap single-row draws, keep lobby-percentiles/static norms from base row.
    return agg


def bootstrap_player_demo_ci(
    demo_id: str,
    steamid: str,
    ranked_row: pd.Series,
    engagement_df: pd.DataFrame,
    n_boot: int = 100,
    random_state: int = 42,
) -> BootstrapCIResult:
    feature_cols = [x for x in FEATURES_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
    ensure_no_forbidden_features(feature_cols, str(FEATURES_PATH))

    xgb_model = XGBClassifier()
    xgb_model.load_model(str(MODEL_PATH))
    calibrator = load_calibrator()

    p = engagement_df.copy()
    p["attacker_steamid"] = p["attacker_steamid"].astype(str).str.strip()
    p = p[(p["demo_id"].astype(str) == str(demo_id)) & (p["attacker_steamid"] == str(steamid).strip())].copy()
    if p.empty:
        raise ValueError(f"No engagement rows for demo={demo_id} steamid={steamid}.")

    rng = np.random.default_rng(seed=int(random_state))
    risks: list[float] = []
    for _ in range(int(max(10, n_boot))):
        idx = rng.integers(0, len(p), size=len(p))
        sampled = p.iloc[idx].copy()
        agg = _aggregate_one_player_from_events(sampled, demo_id=demo_id, player_row=ranked_row)
        row = ranked_row.to_frame().T.copy()
        for c in agg.columns:
            row[c] = agg.iloc[0][c]
        for c in feature_cols:
            if c not in row.columns:
                row[c] = np.nan
        X = row[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
        raw = float(xgb_model.predict_proba(X)[:, 1][0])
        if calibrator is None:
            cal = raw
        elif isinstance(calibrator, dict):
            cal_model = calibrator.get("model")
            method = str(calibrator.get("method", "")).lower()
            if method == "sigmoid" and hasattr(cal_model, "predict_proba"):
                cal = float(np.clip(cal_model.predict_proba(np.asarray([[raw]], dtype=float))[:, 1][0], 0.0, 1.0))
            elif hasattr(cal_model, "predict"):
                cal = float(np.clip(cal_model.predict(np.asarray([raw], dtype=float))[0], 0.0, 1.0))
            else:
                cal = raw
        else:
            cal = float(np.clip(calibrator.predict(np.asarray([raw], dtype=float))[0], 0.0, 1.0))
        risks.append(cal)

    arr = np.asarray(risks, dtype=float)
    p05 = float(np.quantile(arr, 0.05))
    p50 = float(np.quantile(arr, 0.50))
    p95 = float(np.quantile(arr, 0.95))
    return BootstrapCIResult(
        risk_p05=p05,
        risk_p50=p50,
        risk_p95=p95,
        ci_width=float(p95 - p05),
        n_boot=int(max(10, n_boot)),
    )
