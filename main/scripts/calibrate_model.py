from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score


REPORTS_DIR = Path(r"C:\NullCS\main\data\processed\reports")
MODELS_DIR = Path(r"C:\NullCS\main\data\processed\models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

OOF_CSV = REPORTS_DIR / "player_oof_predictions.csv"
OOF_PARQUET = REPORTS_DIR / "player_oof_predictions.parquet"
CALIBRATOR_OUT = MODELS_DIR / "calibration_isotonic.pkl"
SUMMARY_OUT = MODELS_DIR / "calibration_summary.json"
CURVE_OUT = MODELS_DIR / "calibration_curve.csv"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fit probability calibrator from OOF-only predictions.")
    ap.add_argument("--bins", type=int, default=12, help="Number of bins for reliability curve.")
    ap.add_argument("--min-isotonic-positives", type=int, default=25, help="Fallback to sigmoid if positives below this.")
    return ap.parse_args()


def load_oof() -> pd.DataFrame:
    if OOF_PARQUET.exists():
        return pd.read_parquet(OOF_PARQUET)
    if OOF_CSV.exists():
        return pd.read_csv(OOF_CSV)
    raise FileNotFoundError(f"Missing OOF predictions file. Expected {OOF_PARQUET} or {OOF_CSV}.")


def binned_curve(y: np.ndarray, p_raw: np.ndarray, p_cal: np.ndarray, bins: int) -> pd.DataFrame:
    q = pd.qcut(p_raw, q=bins, duplicates="drop")
    df = pd.DataFrame({"y_true": y, "proba_raw": p_raw, "proba_cal": p_cal, "bin": q})
    g = df.groupby("bin", dropna=True)
    rows = []
    for idx, d in enumerate(g, start=1):
        b, x = d
        rows.append(
            {
                "bin_id": idx,
                "bin": str(b),
                "n": int(len(x)),
                "mean_pred_raw": float(x["proba_raw"].mean()),
                "mean_pred_calibrated": float(x["proba_cal"].mean()),
                "empirical_pos_rate": float(x["y_true"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    df = load_oof()
    if "y_true" not in df.columns or "proba_raw_oof" not in df.columns:
        raise ValueError("OOF file must contain y_true and proba_raw_oof columns.")

    y = df["y_true"].astype(int).to_numpy()
    p_raw = np.clip(df["proba_raw_oof"].astype(float).to_numpy(), 1e-6, 1.0 - 1e-6)

    n = len(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n == 0 or n_pos == 0 or n_neg == 0:
        raise ValueError("Need both positive and negative OOF rows for calibration.")

    method = "isotonic"
    if n_pos < int(args.min_isotonic_positives) or n < 200:
        method = "sigmoid"

    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(p_raw, y)
        payload = {"method": "isotonic", "model": cal}
        p_cal = np.clip(cal.predict(p_raw), 0.0, 1.0)
    else:
        lr = LogisticRegression(solver="lbfgs", max_iter=2000)
        lr.fit(p_raw.reshape(-1, 1), y)
        payload = {"method": "sigmoid", "model": lr}
        p_cal = np.clip(lr.predict_proba(p_raw.reshape(-1, 1))[:, 1], 0.0, 1.0)

    dump(payload, CALIBRATOR_OUT)
    print(f"[OK] wrote calibrator: {CALIBRATOR_OUT}")

    summary = {
        "n_rows": int(n),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "method": method,
        "raw_brier": float(brier_score_loss(y, p_raw)),
        "calibrated_brier": float(brier_score_loss(y, p_cal)),
        "raw_logloss": float(log_loss(y, p_raw)),
        "calibrated_logloss": float(log_loss(y, p_cal)),
        "raw_roc_auc": float(roc_auc_score(y, p_raw)),
        "calibrated_roc_auc": float(roc_auc_score(y, p_cal)),
        "raw_pr_auc": float(average_precision_score(y, p_raw)),
        "calibrated_pr_auc": float(average_precision_score(y, p_cal)),
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] wrote summary: {SUMMARY_OUT}")

    curve = binned_curve(y, p_raw, p_cal, bins=max(4, int(args.bins)))
    curve.to_csv(CURVE_OUT, index=False)
    print(f"[OK] wrote calibration curve: {CURVE_OUT}")

    print(
        "[CAL] method={} raw_brier={:.5f} cal_brier={:.5f}".format(
            summary["method"], summary["raw_brier"], summary["calibrated_brier"]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
