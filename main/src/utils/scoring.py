from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


MODELS_DIR = Path(r"C:\NullCS\main\data\processed\models")
CALIBRATOR_PATH = MODELS_DIR / "calibration_isotonic.pkl"
CALIBRATION_SUMMARY_PATH = MODELS_DIR / "calibration_summary.json"

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


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def compute_confidence(
    n_kills_with_rt: float,
    rt_n: float,
    rounds_played: float,
    w1: float = 0.45,
    w2: float = 0.35,
    w3: float = 0.20,
) -> float:
    nk = max(0.0, float(n_kills_with_rt if pd.notna(n_kills_with_rt) else 0.0))
    rn = max(0.0, float(rt_n if pd.notna(rt_n) else 0.0))
    rp = max(0.0, float(rounds_played if pd.notna(rounds_played) else 0.0))
    score = (
        w1 * (math.log1p(nk) / math.log1p(30.0))
        + w2 * (math.log1p(rn) / math.log1p(30.0))
        + w3 * (math.log1p(rp) / math.log1p(24.0))
    )
    return clamp01(score)


def compute_confidence_series(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [
            compute_confidence(nk, rn, rp)
            for nk, rn, rp in zip(
                df.get("n_kills_with_rt", pd.Series([0] * len(df))),
                df.get("rt_n", pd.Series([0] * len(df))),
                df.get("rounds_played", pd.Series([0] * len(df))),
            )
        ],
        index=df.index,
        dtype=float,
    )


def apply_rt_low_evidence_downweight(risk: pd.Series, rt_n: pd.Series, min_rt: int = 8, factor: float = 0.85) -> pd.Series:
    out = risk.astype(float).copy()
    low = rt_n.fillna(0).astype(float) < float(min_rt)
    out.loc[low] = out.loc[low] * float(factor)
    return out


def risk_band(r: float) -> str:
    x = float(r)
    if x < 0.20:
        return "low"
    if x < 0.50:
        return "medium"
    return "high"


def risk_band_series(risk: pd.Series) -> pd.Series:
    return risk.astype(float).map(risk_band)


def top_signal_titles(row: pd.Series, top_k: int = 3) -> list[dict]:
    # Positive means "more suspicious", and rt_median_pct is inverted.
    candidates: list[tuple[str, float]] = []
    for key, title, invert in [
        ("prefire_pct", "High prefire percentile", False),
        ("thrusmoke_pct", "High through-smoke percentile", False),
        ("hs_pct", "High headshot percentile", False),
        ("long_fast_rt_pct", "High long-range fast-RT percentile", False),
        ("rt_median_pct", "Low RT-median percentile", True),
    ]:
        v = row.get(key, np.nan)
        if pd.isna(v):
            continue
        val = float(v)
        suspicious = (1.0 - val) if invert else val
        candidates.append((title, suspicious))
    candidates.sort(key=lambda x: x[1], reverse=True)
    out = []
    for title, sev in candidates[:top_k]:
        out.append({"title": title, "severity": _severity_from_score(sev)})
    return out


def _severity_from_score(s: float) -> str:
    if s >= 0.90:
        return "high"
    if s >= 0.75:
        return "medium"
    return "low"


def load_calibrator(path: Path = CALIBRATOR_PATH):
    if not path.exists():
        return None
    from joblib import load

    return load(path)


def maybe_calibrate(raw: pd.Series, calibrator) -> pd.Series:
    if calibrator is None:
        return raw.astype(float)
    arr = np.asarray(raw.fillna(0.0), dtype=float)
    # Backward/forward compatibility:
    # - direct sklearn calibrator with .predict(arr)
    # - payload dict {"method": "...", "model": ...}
    if isinstance(calibrator, dict):
        model = calibrator.get("model")
        method = str(calibrator.get("method", "")).lower()
        if method == "sigmoid" and hasattr(model, "predict_proba"):
            cal = model.predict_proba(arr.reshape(-1, 1))[:, 1]
        elif hasattr(model, "predict"):
            cal = model.predict(arr)
        else:
            cal = arr
    else:
        cal = calibrator.predict(arr)
    return pd.Series(np.clip(cal, 0.0, 1.0), index=raw.index, dtype=float)


def ensure_no_forbidden_features(feature_cols: list[str], where: str) -> None:
    bad = sorted(set(feature_cols) & FORBIDDEN_FEATURE_COLS)
    for c in feature_cols:
        lc = str(c).lower()
        if (
            "steamid" in lc
            or lc == "demo_id"
            or lc == "label"
            or "cheater_map" in lc
            or ("cheater" in lc and "rate" not in lc and "pct" not in lc and "score" not in lc)
        ):
            if c not in bad:
                bad.append(c)
    bad = sorted(set(bad))
    if bad:
        raise ValueError(
            f"Leakage guardrail: forbidden feature columns detected in {where}: {bad}. "
            "Remove identity/label-derived columns before scoring."
        )
