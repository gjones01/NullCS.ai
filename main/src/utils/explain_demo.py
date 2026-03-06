from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../main
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.scoring import top_signal_titles
from src.utils.bootstrap_demo_ci import bootstrap_player_demo_ci


@dataclass(frozen=True)
class ExplainConfig:
    processed_root: Path
    demos_root: Path
    models_root: Path
    reports_root: Path
    ranked_csv: Path
    proba_col: str
    mode: str


def default_config(mode: str = "oof") -> ExplainConfig:
    mode = str(mode).strip().lower()
    if mode not in {"oof", "insample", "infer"}:
        raise ValueError(f"Unsupported mode '{mode}'. Expected one of: oof, insample, infer.")

    processed_root = Path(r"C:\NullCS\main\data\processed")
    ranked_csv = processed_root / "reports" / f"ranked_player_demo_suspicion_{mode}.csv"
    if mode == "oof":
        proba_col = "proba_raw_oof"
    elif mode == "insample":
        proba_col = "proba_cheater_insample"
    else:
        proba_col = "proba_cheater_infer"

    return ExplainConfig(
        processed_root=processed_root,
        demos_root=processed_root / "demos",
        models_root=processed_root / "models",
        reports_root=processed_root / "reports",
        ranked_csv=ranked_csv,
        proba_col=proba_col,
        mode=mode,
    )


def _read_ranked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Ranked file not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported ranked file type: {path.suffix}")


def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def _normalize_steamid_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def _severity_rank(s: str) -> int:
    mp = {"high": 3, "medium": 2, "low": 1, "context": 0}
    return mp.get(str(s).lower(), 0)


def _parse_top_reasons_field(x) -> list[dict]:
    if isinstance(x, list):
        return x
    if isinstance(x, str) and x.strip():
        try:
            val = json.loads(x)
            if isinstance(val, list):
                return val
        except Exception:
            return []
    return []


def pick_top_player_in_demo(
    cfg: ExplainConfig,
    demo_id: str,
    steamid: str | None = None,
    name: str | None = None,
) -> pd.Series:
    df = _read_ranked(cfg.ranked_csv)
    required = {"demo_id", "attacker_steamid"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Ranked file missing columns: {sorted(missing)}")

    df = df.copy()
    df["attacker_steamid"] = _normalize_steamid_series(df["attacker_steamid"])
    d = df[df["demo_id"].astype(str) == str(demo_id)].copy()
    if d.empty:
        raise ValueError(f"No rows for demo_id={demo_id} in {cfg.ranked_csv}")

    if steamid is not None:
        steamid = str(steamid).strip()
        d = d[d["attacker_steamid"] == steamid].copy()
        if d.empty:
            raise ValueError(f"steamid {steamid} not found in demo {demo_id}.")
    elif name is not None:
        name = str(name).strip()
        if "attacker_name" not in d.columns:
            raise ValueError("attacker_name column missing from ranked file.")
        d = d[d["attacker_name"].astype(str) == name].copy()
        if d.empty:
            raise ValueError(f'name "{name}" not found in demo {demo_id}.')

    sort_col = "risk" if "risk" in d.columns else (cfg.proba_col if cfg.proba_col in d.columns else d.columns[0])
    d = d.sort_values(sort_col, ascending=False)
    return d.iloc[0]


def load_engagement_features(cfg: ExplainConfig, demo_id: str) -> pd.DataFrame:
    p = cfg.demos_root / demo_id / "engagement_features.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing engagement features: {p}")
    df = pd.read_parquet(p)
    if "demo_id" not in df.columns:
        df["demo_id"] = str(demo_id)
    return df


def build_reasons(top_row: pd.Series, eng: pd.DataFrame) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    steamid = str(top_row["attacker_steamid"]).strip()
    player = eng.copy()
    player["attacker_steamid"] = _normalize_steamid_series(player["attacker_steamid"])
    player = player[player["attacker_steamid"] == steamid].copy()

    reasons: list[dict] = []
    top = top_row.to_dict()
    rt_low_conf = int(top.get("rt_n", 0) or 0) < 8

    def _f(key: str, default=None):
        return _safe_float(top.get(key), default=default)

    if "rt_ticks" in player.columns:
        rt = player["rt_ticks"].dropna()
        if len(rt) > 0:
            med = float(rt.median())
            p05 = float(rt.quantile(0.05))
            frac_le_4 = float((rt <= 4).mean())
            frac_le_6 = float((rt <= 6).mean())

            severity = "low"
            if frac_le_4 >= 0.20 or (frac_le_6 >= 0.35 and med <= 8):
                severity = "high"
            elif frac_le_4 >= 0.10 or frac_le_6 >= 0.25:
                severity = "medium"
            if rt_low_conf and severity in {"high", "medium"}:
                severity = "low"

            reasons.append(
                {
                    "title": "Reaction-time tail",
                    "severity": severity,
                    "summary": f"median={med:.1f} ticks | p05={p05:.1f} | %<=4={frac_le_4:.2%} | %<=6={frac_le_6:.2%}",
                    "why_it_matters": "Fast first-shot tails can indicate assistive aim in context.",
                    "evidence_file": "evidence_fast_rt.csv",
                    "confidence_note": "low confidence (rt_n<8)" if rt_low_conf else "normal",
                }
            )

    if "is_prefire" in player.columns:
        pre = player["is_prefire"].fillna(False).astype(bool)
        prefire_rate = float(pre.mean()) if len(pre) else 0.0
        severity = "high" if prefire_rate >= 0.12 else "medium" if prefire_rate >= 0.06 else "low"
        if rt_low_conf and severity in {"high", "medium"}:
            severity = "low"
        reasons.append(
            {
                "title": "Prefire rate",
                "severity": severity,
                "summary": f"prefire_rate={prefire_rate:.2%}",
                "why_it_matters": "Repeated first shots before visibility can be suspicious with corroborating signals.",
                "evidence_file": "evidence_prefire.csv",
                "confidence_note": "low confidence (rt_n<8)" if rt_low_conf else "normal",
            }
        )

    if "is_thrusmoke" in player.columns:
        ts = player["is_thrusmoke"].fillna(False).astype(bool)
        ts_rate = float(ts.mean()) if len(ts) else 0.0
        ts_kills = int(ts.sum()) if len(ts) else 0
        severity = "high" if ts_rate >= 0.10 else "medium" if ts_rate >= 0.05 else "low"
        reasons.append(
            {
                "title": "Through-smoke kill pattern",
                "severity": severity,
                "summary": f"thrusmoke_kill_rate={ts_rate:.2%} | thrusmoke_kills={ts_kills}",
                "why_it_matters": "Sustained through-smoke outcomes can be suspicious alongside timing and aiming signals.",
                "evidence_file": "evidence_thrusmoke.csv",
                "confidence_note": "normal",
            }
        )

    if {"distance", "rt_ticks"}.issubset(player.columns):
        lr = player[(player["distance"].fillna(-1) >= 1500.0) & (player["rt_ticks"].notna())].copy()
        lr_rate = float((lr["rt_ticks"] <= 4).mean()) if len(lr) else 0.0
        severity = "high" if len(lr) >= 3 and lr_rate >= 0.40 else "medium" if len(lr) >= 2 and lr_rate >= 0.25 else "low"
        if rt_low_conf and severity in {"high", "medium"}:
            severity = "low"
        reasons.append(
            {
                "title": "Long-range fast reaction (<=4 ticks)",
                "severity": severity,
                "summary": f"long_range_fast_rt_rate_4={lr_rate:.2%} over n={len(lr)} long-range kills",
                "why_it_matters": "Very fast first-shot reactions at long range are comparatively rare.",
                "evidence_file": "evidence_long_range_fast_rt_4.csv",
                "confidence_note": "low confidence (rt_n<8)" if rt_low_conf else "normal",
            }
        )

    # Add UI short reasons from ranked row when available.
    ranked_short = _parse_top_reasons_field(top.get("top_reasons"))
    for r in ranked_short:
        reasons.append(
            {
                "title": str(r.get("title", "Model signal")),
                "severity": str(r.get("severity", "low")),
                "summary": "High relative lobby signal.",
                "why_it_matters": "Ranked by within-lobby percentiles used during model scoring.",
                "evidence_file": "",
                "confidence_note": "normal",
            }
        )

    cols = [
        c
        for c in [
            "round_num",
            "kill_tick",
            "t0_visible",
            "first_shot_tick",
            "rt_ticks",
            "distance",
            "weapon",
            "headshot",
            "is_prefire",
            "is_thrusmoke",
            "attacker_name",
            "victim_name",
            "victim_steamid",
        ]
        if c in player.columns
    ]
    evidence_fast_rt = player.sort_values("rt_ticks", ascending=True).head(30)[cols] if "rt_ticks" in player.columns else player.head(30)[cols]
    evidence_prefire = player[player["is_prefire"].fillna(False)].sort_values("kill_tick").head(30)[cols] if "is_prefire" in player.columns else pd.DataFrame()
    evidence_thrusmoke = player[player["is_thrusmoke"].fillna(False)].sort_values("kill_tick").head(30)[cols] if "is_thrusmoke" in player.columns else pd.DataFrame()
    evidence_long_range_fast_rt_4 = (
        player[(player["distance"].fillna(-1) >= 1500.0) & (player["rt_ticks"].notna()) & (player["rt_ticks"] <= 4)].sort_values("kill_tick").head(30)[cols]
        if {"distance", "rt_ticks"}.issubset(player.columns)
        else pd.DataFrame()
    )

    evidence = {
        "evidence_fast_rt.csv": evidence_fast_rt,
        "evidence_prefire.csv": evidence_prefire,
        "evidence_thrusmoke.csv": evidence_thrusmoke,
        "evidence_long_range_fast_rt_4.csv": evidence_long_range_fast_rt_4,
    }
    reasons.sort(key=lambda x: _severity_rank(x["severity"]), reverse=True)
    return reasons, evidence


def _build_signals(top: pd.Series) -> dict:
    def val(k):
        v = top.get(k)
        if pd.isna(v):
            return None
        try:
            return float(v)
        except Exception:
            return None

    return {
        "raw_values": {
            "prefire_rate": val("prefire_rate"),
            "headshot_rate": val("headshot_rate"),
            "thrusmoke_kill_rate": val("thrusmoke_kill_rate"),
            "rt_median": val("rt_median"),
            "long_range_fast_rt_rate_4": val("long_range_fast_rt_rate_4"),
        },
        "lobby_percentiles": {
            "prefire_pct": val("prefire_pct"),
            "hs_pct": val("hs_pct"),
            "thrusmoke_pct": val("thrusmoke_pct"),
            "rt_median_pct": val("rt_median_pct"),
            "long_fast_rt_pct": val("long_fast_rt_pct"),
        },
        "top_contributing_signals": top_signal_titles(top, top_k=5),
    }


def explain_demo(
    cfg: ExplainConfig,
    demo_id: str,
    steamid: str | None = None,
    name: str | None = None,
    with_ci: bool = False,
    n_boot: int = 100,
):
    top = pick_top_player_in_demo(cfg, demo_id, steamid=steamid, name=name)
    sid = str(top["attacker_steamid"]).strip()
    name = str(top.get("attacker_name", ""))

    eng = load_engagement_features(cfg, demo_id)
    reasons, evidence = build_reasons(top, eng)

    out_dir = cfg.reports_root / demo_id / sid
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname, df in evidence.items():
        df.to_csv(out_dir / fname, index=False)

    # CI is optional due runtime cost.
    ci = None
    if with_ci:
        ci_obj = bootstrap_player_demo_ci(
            demo_id=demo_id,
            steamid=sid,
            ranked_row=top,
            engagement_df=eng,
            n_boot=int(max(20, n_boot)),
        )
        ci = {
            "risk_p05": ci_obj.risk_p05,
            "risk_p50": ci_obj.risk_p50,
            "risk_p95": ci_obj.risk_p95,
            "ci_width": ci_obj.ci_width,
            "n_boot": ci_obj.n_boot,
        }
        (out_dir / "ci.json").write_text(json.dumps(ci, indent=2), encoding="utf-8")

    risk = _safe_float(top.get("risk"), _safe_float(top.get("proba_calibrated"), _safe_float(top.get(cfg.proba_col), 0.0)))
    confidence = _safe_float(top.get("confidence"), 0.0)
    report = {
        "mode": cfg.mode,
        "demo_id": str(demo_id),
        "player": {
            "attacker_name": name,
            "attacker_steamid": sid,
        },
        "risk": {
            "score": risk,
            "band": str(top.get("risk_band", "unknown")),
            "raw_probability": _safe_float(top.get(cfg.proba_col), None),
            "calibrated_probability": _safe_float(top.get("proba_calibrated"), None),
        },
        "confidence": {
            "score": confidence,
            "rt_reason_confidence": "low" if int(top.get("rt_n", 0) or 0) < 8 else "normal",
        },
        "uncertainty_ci": ci,
        "reasons": reasons,
        "signals": _build_signals(top),
        "evidence_files": [k for k in evidence.keys()],
    }

    (out_dir / "top_player_row.json").write_text(top.to_json(indent=2), encoding="utf-8")
    (out_dir / "reasons.json").write_text(json.dumps(reasons, indent=2), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n=== DEMO REPORT: {demo_id} ===")
    _safe_print(f"player: {name} ({sid})")
    _safe_print(f"risk: {risk:.6f} confidence: {confidence:.3f}")
    if ci is not None:
        _safe_print(f"ci: [{ci['risk_p05']:.6f}, {ci['risk_p95']:.6f}] n={ci['n_boot']}")
    print(f"[OK] wrote: {out_dir}")
    return out_dir
