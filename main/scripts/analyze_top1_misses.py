from __future__ import annotations

from pathlib import Path
import pandas as pd


REPORTS_DIR = Path(r"C:\NullCS\main\data\processed\reports")
PLAYER_FEATURES_PATH = Path(r"C:\NullCS\main\data\processed\player_features.parquet")
MISSES_PATH = REPORTS_DIR / "top1_misses.csv"
RANKED_PATH = REPORTS_DIR / "ranked_player_demo_suspicion_oof.csv"
OUT_DIR = REPORTS_DIR / "miss_analysis"


def _norm_sid(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def main() -> None:
    if not MISSES_PATH.exists():
        raise FileNotFoundError(f"Missing: {MISSES_PATH}")
    if not RANKED_PATH.exists():
        raise FileNotFoundError(f"Missing: {RANKED_PATH}")
    if not PLAYER_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing: {PLAYER_FEATURES_PATH}")

    misses = pd.read_csv(MISSES_PATH)
    ranked = pd.read_csv(RANKED_PATH)
    feats = pd.read_parquet(PLAYER_FEATURES_PATH)

    if misses.empty:
        print("[INFO] top1_misses.csv is empty; nothing to analyze.")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        return

    ranked["attacker_steamid"] = _norm_sid(ranked["attacker_steamid"])
    feats["attacker_steamid"] = _norm_sid(feats["attacker_steamid"])

    # all feature columns from player_features except identifiers.
    id_cols = {"demo_id", "map_name", "attacker_steamid", "attacker_name", "label"}
    feature_cols = [c for c in feats.columns if c not in id_cols]
    numeric_feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(feats[c])]

    cols_for_join = [
        "demo_id",
        "map_name",
        "attacker_steamid",
        "attacker_name",
        "label",
        *feature_cols,
    ]
    feats_small = feats[cols_for_join].copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for _, miss in misses.iterrows():
        demo_id = str(miss["demo_id"])
        cheater_sid = str(miss["cheater_steamid"]).strip()
        pred_sid = str(miss["pred_top1_steamid"]).strip()

        c = feats_small[(feats_small["demo_id"].astype(str) == demo_id) & (feats_small["attacker_steamid"] == cheater_sid)]
        p = feats_small[(feats_small["demo_id"].astype(str) == demo_id) & (feats_small["attacker_steamid"] == pred_sid)]

        if c.empty or p.empty:
            print(
                f"[WARN] {demo_id}: missing rows in player_features for "
                f"cheater={cheater_sid} or pred={pred_sid}; skipping"
            )
            continue

        crow = c.iloc[0]
        prow = p.iloc[0]

        out_row: dict[str, object] = {
            "demo_id": demo_id,
            "cheater_steamid": cheater_sid,
            "pred_top1_steamid": pred_sid,
            "pred_top1_proba": float(miss.get("pred_top1_proba", float("nan"))),
            "cheater_proba": float(miss.get("cheater_proba", float("nan"))),
            "delta_proba_pred_minus_cheater": float(miss.get("pred_top1_proba", float("nan"))) - float(miss.get("cheater_proba", float("nan"))),
            "cheater_name": crow.get("attacker_name"),
            "pred_top1_name": prow.get("attacker_name"),
            "cheater_label": crow.get("label"),
            "pred_top1_label": prow.get("label"),
        }

        for col in feature_cols:
            cval = crow.get(col)
            pval = prow.get(col)
            out_row[f"cheater__{col}"] = cval
            out_row[f"pred_top1__{col}"] = pval
            if col in numeric_feature_cols:
                cnum = pd.to_numeric(pd.Series([cval]), errors="coerce").iloc[0]
                pnum = pd.to_numeric(pd.Series([pval]), errors="coerce").iloc[0]
                if pd.notna(cnum) and pd.notna(pnum):
                    out_row[f"delta__{col}"] = float(pnum) - float(cnum)
                else:
                    out_row[f"delta__{col}"] = float("nan")

        out_path = OUT_DIR / f"{demo_id}.csv"
        pd.DataFrame([out_row]).to_csv(out_path, index=False)
        print(f"[OK] wrote {out_path}")

    print(f"[DONE] miss analysis written to {OUT_DIR}")


if __name__ == "__main__":
    main()
