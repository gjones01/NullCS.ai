from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from xgboost import XGBClassifier
import sys

DATA_PATH = Path(r"C:\NullCS\main\data\processed\player_features.parquet")
MODEL_PATH = Path(r"C:\NullCS\main\data\processed\models\xgb_player_level_gridcv.json")
FEATS_PATH = Path(r"C:\NullCS\main\data\processed\models\xgb_player_level_features.txt")
BEST_PARAMS_PATH = Path(r"C:\NullCS\main\data\processed\models\xgb_player_level_best_params.json")
GRID_RESULTS_PATH = Path(r"C:\NullCS\main\data\processed\models\xgb_gridcv_results.csv")

OUT_DIR = Path(r"C:\NullCS\main\data\processed\reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../main
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.scoring import (
    ensure_no_forbidden_features,
    compute_confidence_series,
    apply_rt_low_evidence_downweight,
    load_calibrator,
    maybe_calibrate,
    risk_band_series,
    top_signal_titles,
)

FALLBACK_PARAMS = dict(
    colsample_bytree=0.8,
    gamma=0.0,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=3,
    n_estimators=400,
    reg_alpha=0.0,
    reg_lambda=5.0,
    subsample=0.8,
)

THRESHOLDS = [0.2, 0.3, 0.4, 0.5]
TOPK = [1, 2, 3, 5]
FORBIDDEN_FEATURE_COLS = {
    "label",
    "demo_id",
    "attacker_steamid",
    "cheater_steamid",
    "is_cdemo",
    "is_cheater_demo",
    "is_target_player",
    "demo_label_any_cheater",
}


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--use-saved-model",
        action="store_true",
        help="Use saved model for in-sample inference on provided rows (not grouped OOF evaluation).",
    )
    ap.add_argument(
        "--debug-miss",
        default=None,
        help="Print full lobby ranking for the given demo_id from current eval output.",
    )
    ap.add_argument(
        "--write-oof-parquet",
        action="store_true",
        help="Also write player_oof_predictions.parquet (CSV is always written).",
    )
    return ap.parse_args()


def _normalize_model_params(params: dict) -> dict:
    int_keys = {"max_depth", "min_child_weight", "n_estimators"}
    out = dict(params)
    out.pop("scale_pos_weight", None)
    for k in list(out.keys()):
        if k in int_keys:
            out[k] = int(out[k])
        else:
            out[k] = float(out[k])
    return out


def load_best_params_from_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        params = payload.get("best_params", payload)
        if not isinstance(params, dict) or not params:
            print(f"[WARN] best params JSON at {path} has no usable params.")
            return None
        params = _normalize_model_params(params)
        print(f"[INFO] loaded best params from: {path}")
        print(f"[INFO] eval params: {params}")
        return params
    except Exception as e:
        print(f"[WARN] failed to parse best params JSON at {path}: {e}")
        return None


def load_best_params_from_gridcv(results_path: Path) -> dict:
    if not results_path.exists():
        print(f"[WARN] grid results not found at {results_path}; using fallback params.")
        return dict(FALLBACK_PARAMS)

    df = pd.read_csv(results_path)
    if df.empty:
        print(f"[WARN] grid results at {results_path} are empty; using fallback params.")
        return dict(FALLBACK_PARAMS)

    if "rank_test_score" in df.columns:
        row = df.sort_values("rank_test_score", ascending=True).iloc[0]
    elif "mean_test_score" in df.columns:
        row = df.sort_values("mean_test_score", ascending=False).iloc[0]
    else:
        print(f"[WARN] grid results missing ranking columns; using fallback params.")
        return dict(FALLBACK_PARAMS)

    params = {}
    for col in df.columns:
        if not col.startswith("param_"):
            continue
        key = col[len("param_") :]
        value = row[col]
        if pd.isna(value):
            continue
        params[key] = value

    params = _normalize_model_params(params)

    if not params:
        print(f"[WARN] no params parsed from {results_path}; using fallback params.")
        return dict(FALLBACK_PARAMS)

    print(f"[INFO] loaded best params from: {results_path}")
    print(f"[INFO] eval params: {params}")
    return params


def load_best_params() -> dict:
    params = load_best_params_from_json(BEST_PARAMS_PATH)
    if params is not None:
        return params
    return load_best_params_from_gridcv(GRID_RESULTS_PATH)


def predict_with_retrained_folds(X, y, groups, scale_pos_weight, model_params: dict) -> tuple[np.ndarray, np.ndarray]:
    # out-of-fold predictions
    oof = np.zeros(len(y), dtype=float)
    fold_ids = np.full(len(y), -1, dtype=int)

    gkf = GroupKFold(n_splits=N_SPLITS)
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_te = X.iloc[te_idx]

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            **model_params,
        )
        model.fit(X_tr, y_tr)

        oof[te_idx] = model.predict_proba(X_te)[:, 1]
        fold_ids[te_idx] = fold
        print(f"[FOLD {fold}] done")

    return oof, fold_ids


def predict_with_saved_model(X) -> np.ndarray:
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))
    print(f"[INFO] loaded model from: {MODEL_PATH}")
    return model.predict_proba(X)[:, 1]


def main():
    args = parse_args()
    df = pd.read_parquet(DATA_PATH)
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    if "n_players" in df.columns:
        before_rows = len(df)
        before_demos = df["demo_id"].nunique()
        df = df[df["n_players"] >= 8].copy()
        print(
            f"[INFO] n_players>=8 filter: rows {before_rows}->{len(df)} "
            f"demos {before_demos}->{df['demo_id'].nunique()}"
        )

    # Load feature list from file (so it matches training)
    feature_cols = FEATS_PATH.read_text(encoding="utf-8").strip().splitlines()
    ensure_no_forbidden_features(feature_cols, str(FEATS_PATH))

    X = df[feature_cols].fillna(0.0).astype(float)
    y = df["label"].values
    groups = df["demo_id"].values

    # imbalance weight
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_pos_weight = n_neg / max(1, n_pos)
    model_params = load_best_params()

    if args.use_saved_model:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Saved model not found at: {MODEL_PATH}")
        print("[WARN] Using saved model on provided rows (in-sample/inference mode, not grouped OOF).")
        scores = predict_with_saved_model(X)
        fold_ids = np.full(len(y), -1, dtype=int)
        mode = "insample"
        raw_score_col = "proba_cheater_insample"
    else:
        print("[INFO] Default mode: computing grouped OOF predictions with fold retraining.")
        scores, fold_ids = predict_with_retrained_folds(X, y, groups, scale_pos_weight, model_params)
        mode = "oof"
        raw_score_col = "proba_raw_oof"

    # metrics
    pr = average_precision_score(y, scores)
    roc = roc_auc_score(y, scores)
    print(f"\n[{mode.upper()}] PR-AUC={pr:.4f}  ROC-AUC={roc:.4f}")

    # threshold reports
    print("\n[THRESHOLDS]")
    for thr in THRESHOLDS:
        pred = (scores >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        precision = tp / max(1, (tp + fp))
        recall = tp / max(1, (tp + fn))
        print(f"  thr={thr:.2f}  TP={tp} FP={fp} FN={fn} TN={tn}  prec={precision:.3f} rec={recall:.3f}")

    # attach predictions
    keep_cols = ["demo_id", "map_name", "attacker_steamid", "attacker_name", "label"]
    keep_cols.extend([c for c in feature_cols if c not in keep_cols])
    out = df[keep_cols].copy()
    out["attacker_steamid"] = out["attacker_steamid"].astype(str).str.strip()
    out[raw_score_col] = scores
    if mode == "oof":
        out["proba_cheater_oof"] = scores  # backward compatibility
    out["fold_id"] = fold_ids
    out["y_true"] = out["label"].astype(int)
    out["confidence"] = compute_confidence_series(out)

    calibrator = load_calibrator()
    if calibrator is not None:
        out["proba_calibrated"] = maybe_calibrate(out[raw_score_col], calibrator)
        print("[INFO] applied saved calibrator to evaluation output.")
    else:
        out["proba_calibrated"] = np.nan

    risk_base = out["proba_calibrated"].copy()
    missing = risk_base.isna()
    risk_base.loc[missing] = out.loc[missing, raw_score_col].astype(float)
    out["risk"] = apply_rt_low_evidence_downweight(risk_base, out.get("rt_n", pd.Series([0] * len(out))))
    out["risk_band"] = risk_band_series(out["risk"])
    out["rt_reason_confidence"] = np.where(out.get("rt_n", pd.Series([0] * len(out))).fillna(0).astype(float) < 8, "low", "normal")
    out["top_reasons"] = out.apply(lambda r: json.dumps(top_signal_titles(r, top_k=3)), axis=1)

    # save player-level ranked list
    out_sorted = out.sort_values("risk", ascending=False)
    player_csv = OUT_DIR / f"ranked_player_demo_suspicion_{mode}.csv"
    out_sorted.to_csv(player_csv, index=False)
    print(f"\n[OK] wrote {player_csv}")

    # OOF experiment dataset for calibration/CI work.
    if mode == "oof":
        base_cols = [
            "demo_id",
            "attacker_steamid",
            "y_true",
            "proba_raw_oof",
            "fold_id",
            "n_kills",
            "rt_n",
            "hs_n",
            "smoke_n",
            "rounds_played",
            "map_name",
            "attacker_name",
        ]
        numeric_feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(out[c])]
        oof_cols = [c for c in base_cols if c in out.columns] + [c for c in numeric_feature_cols if c not in base_cols]
        oof_rows = out[oof_cols].copy()
        oof_csv = OUT_DIR / "player_oof_predictions.csv"
        oof_rows.to_csv(oof_csv, index=False)
        print(f"[OK] wrote {oof_csv}")
        if args.write_oof_parquet:
            oof_parquet = OUT_DIR / "player_oof_predictions.parquet"
            oof_rows.to_parquet(oof_parquet, index=False)
            print(f"[OK] wrote {oof_parquet}")

    # demo-level aggregation (top-k and max)
    demo_rows = []
    for demo_id, g in out.groupby("demo_id"):
        probs = np.sort(g["risk"].values)[::-1]
        labels_sorted = g.sort_values("risk", ascending=False)["label"].values
        row = {"demo_id": demo_id, "map_name": g["map_name"].iloc[0]}

        row["demo_label_any_cheater"] = int(g["label"].max())  # since CDemo demos are all cheater-labeled
        row["n_players"] = len(g)

        row["max_proba"] = float(probs[0]) if len(probs) else 0.0
        row["top1_has_cheater"] = int(labels_sorted[0] == 1) if len(labels_sorted) else 0
        row["top2_has_cheater"] = int((labels_sorted[:2] == 1).any()) if len(labels_sorted) else 0
        row["top3_has_cheater"] = int((labels_sorted[:3] == 1).any()) if len(labels_sorted) else 0
        for k in TOPK:
            row[f"top{k}_mean"] = float(probs[:k].mean()) if len(probs) >= k else float(probs.mean()) if len(probs) else 0.0

        demo_rows.append(row)

    demo_df = pd.DataFrame(demo_rows)
    demo_df = demo_df.sort_values("top3_mean", ascending=False)

    demo_csv = OUT_DIR / f"ranked_demo_suspicion_{mode}.csv"
    demo_df.to_csv(demo_csv, index=False)
    print(f"[OK] wrote {demo_csv}")

    # CDemo Top1 misses for debugging.
    misses = []
    cdemo_ids = [d for d in demo_df["demo_id"].tolist() if str(d).lower().startswith("cdemo")]
    for demo_id in cdemo_ids:
        g = out[out["demo_id"] == demo_id].sort_values("risk", ascending=False).copy()
        if g.empty:
            continue
        if int((g["label"] == 1).any()) == 0:
            continue
        top1 = g.iloc[0]
        if int(top1["label"]) == 1:
            continue
        cheater_rows = g[g["label"] == 1]
        cheater = cheater_rows.iloc[0]
        misses.append(
            {
                "demo_id": str(demo_id),
                "cheater_steamid": str(cheater["attacker_steamid"]),
                "pred_top1_steamid": str(top1["attacker_steamid"]),
                "pred_top1_risk": float(top1["risk"]),
                "cheater_risk": float(cheater["risk"]),
                "pred_top1_proba": float(top1["risk"]),
                "cheater_proba": float(cheater["risk"]),
            }
        )
    misses_df = pd.DataFrame(misses).sort_values("pred_top1_risk", ascending=False) if misses else pd.DataFrame(
        columns=["demo_id", "cheater_steamid", "pred_top1_steamid", "pred_top1_risk", "cheater_risk"]
    )
    misses_csv = OUT_DIR / "top1_misses.csv"
    misses_df.to_csv(misses_csv, index=False)
    print(f"[OK] wrote {misses_csv}")

    cdemo_mask = demo_df["demo_id"].str.lower().str.startswith("cdemo")
    cdemo_df = demo_df[cdemo_mask].copy()
    if len(cdemo_df):
        cheater_ranks = []
        for demo_id in cdemo_df["demo_id"].tolist():
            g = out[out["demo_id"] == demo_id].sort_values("risk", ascending=False).reset_index(drop=True)
            idx = g.index[g["label"] == 1].tolist()
            if idx:
                cheater_ranks.append(int(idx[0] + 1))
        top1_acc = cdemo_df["top1_has_cheater"].mean()
        top2_acc = cdemo_df["top2_has_cheater"].mean()
        top3_acc = cdemo_df["top3_has_cheater"].mean()
        mean_rank = float(np.mean(cheater_ranks)) if cheater_ranks else float("nan")
        print(
            f"\n[CDemo ranking] n={len(cdemo_df)} "
            f"Top1={top1_acc:.3f} Top2={top2_acc:.3f} Top3={top3_acc:.3f} "
            f"MeanCheaterRank={mean_rank:.3f}"
        )
    else:
        print("\n[CDemo ranking] no CDemo rows present.")

    # False-positive report for non-CDemo players.
    non_cdemo = out[~out["demo_id"].astype(str).str.lower().str.startswith("cdemo")].copy()
    fp_hi = non_cdemo[non_cdemo["risk"] > 0.8].copy()
    print(f"\n[NON-CDemo] count risk>0.8: {len(fp_hi)}")

    def summarize_row_reason(r: pd.Series) -> str:
        reasons = []
        if r.get("prefire_pct", 0) >= 0.9:
            reasons.append("high_lobby_prefire_pct")
        if r.get("thrusmoke_pct", 0) >= 0.9:
            reasons.append("high_lobby_thrusmoke_pct")
        if r.get("long_fast_rt_pct", 0) >= 0.9:
            reasons.append("high_lobby_long_fast_rt_pct")
        if r.get("hs_pct", 0) >= 0.9:
            reasons.append("high_lobby_hs_pct")
        if pd.notna(r.get("max_fast_rt_streak")) and r.get("max_fast_rt_streak", 0) >= 3:
            reasons.append("fast_rt_streak>=3")
        if not reasons:
            reasons.append("mixed_signal")
        return "|".join(reasons)

    top_non_cdemo = non_cdemo.sort_values("risk", ascending=False).head(30).copy()
    top_non_cdemo["reason_summary"] = top_non_cdemo.apply(summarize_row_reason, axis=1)
    fp_cols = [c for c in [
        "demo_id", "map_name", "attacker_name", "attacker_steamid", "label", raw_score_col, "proba_calibrated", "risk", "confidence", "risk_band",
        "prefire_rate", "thrusmoke_kill_rate", "long_range_fast_rt_rate_4", "headshot_rate",
        "prefire_pct", "thrusmoke_pct", "long_fast_rt_pct", "hs_pct", "max_fast_rt_streak",
        "reason_summary",
    ] if c in top_non_cdemo.columns]
    fp_csv = OUT_DIR / f"top_non_cdemo_false_positives_{mode}.csv"
    top_non_cdemo[fp_cols].to_csv(fp_csv, index=False)
    print(f"[OK] wrote {fp_csv}")

    # show top 15 demos
    print("\n[TOP 15 DEMOS by top3_mean]")
    print(demo_df.head(15).to_string(index=False))

    if "proba_calibrated" in out.columns:
        has_cal = out["proba_calibrated"].notna()
        if has_cal.any():
            print(
                "[DIST] raw_mean={:.4f} raw_std={:.4f} calibrated_mean={:.4f} calibrated_std={:.4f}".format(
                    float(out[raw_score_col].mean()),
                    float(out[raw_score_col].std(ddof=1)),
                    float(out.loc[has_cal, "proba_calibrated"].mean()),
                    float(out.loc[has_cal, "proba_calibrated"].std(ddof=1)),
                )
            )

    if args.debug_miss:
        dbg = out[out["demo_id"].astype(str) == str(args.debug_miss)].copy()
        if dbg.empty:
            print(f"\n[DEBUG MISS] demo_id={args.debug_miss} not found in evaluation rows.")
        else:
            dbg = dbg.sort_values("risk", ascending=False)
            cols = ["demo_id", "attacker_name", "attacker_steamid", "label", raw_score_col, "proba_calibrated", "risk", "confidence"]
            print(f"\n[DEBUG MISS] Full lobby ranking for {args.debug_miss}")
            _safe_print(dbg[cols].to_string(index=False))


if __name__ == "__main__":
    main()
