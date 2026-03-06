from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import sys

from sklearn.model_selection import GroupKFold, GridSearchCV
from xgboost import XGBClassifier

DATA_PATH = Path(r"C:\NullCS\main\data\processed\player_features.parquet")
OUT_DIR = Path(r"C:\NullCS\main\data\processed\models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../main
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.scoring import ensure_no_forbidden_features


def main():
    df = pd.read_parquet(DATA_PATH)

    # labeled only
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

    groups = df["demo_id"].values

    # numeric features only
    exclude = {"label", "demo_id", "map_name", "attacker_name", "attacker_steamid"}
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    ensure_no_forbidden_features(feature_cols, "training feature list")

    X = df[feature_cols].fillna(0.0)
    y = df["label"].values

    # imbalance weight
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_pos_weight = n_neg / max(1, n_pos)
    print(f"[INFO] rows={len(df)} neg={n_neg} pos={n_pos} scale_pos_weight={scale_pos_weight:.3f}")
    print(f"[INFO] demos={df['demo_id'].nunique()}  features={len(feature_cols)}")

    base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )

    # Tight grid (36 combos). Expand later if needed.
    param_grid = {
        "n_estimators": [400, 800, 1200],
        "max_depth": [3, 4],
        "learning_rate": [0.03, 0.05, 0.08],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [3, 8],
        "reg_lambda": [1.0, 5.0],
        # keep gamma/reg_alpha fixed for first grid
        "gamma": [0.0],
        "scale_pos_weight": [scale_pos_weight],
        "reg_alpha": [0.0],
    }

    cv = GroupKFold(n_splits=N_SPLITS)


    search = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring="average_precision",
        cv=cv.split(X, y, groups=groups),
        verbose=2,
        n_jobs=-1,
        return_train_score=True,
    )

    search.fit(X, y)

    print("\n[RESULT] Best PR-AUC (CV):", search.best_score_)
    print("[RESULT] Best params:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    # Save best model + features + cv results
    best_model = search.best_estimator_
    model_path = OUT_DIR / "xgb_player_level_gridcv.json"
    best_model.save_model(model_path)

    feat_path = OUT_DIR / "xgb_player_level_features.txt"
    Path(feat_path).write_text("\n".join(feature_cols), encoding="utf-8")

    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    results_path = OUT_DIR / "xgb_gridcv_results.csv"
    results.to_csv(results_path, index=False)
    best_params_path = OUT_DIR / "xgb_player_level_best_params.json"
    payload = {
        "best_score_average_precision": float(search.best_score_),
        "best_params": search.best_params_,
    }
    best_params_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n[OK] saved model: {model_path}")
    print(f"[OK] saved features: {feat_path}")
    print(f"[OK] saved results: {results_path}")
    print(f"[OK] saved best params: {best_params_path}")


if __name__ == "__main__":
    main()
