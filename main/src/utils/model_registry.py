from __future__ import annotations

from pathlib import Path


def _candidate_models(models_dir: Path) -> list[Path]:
    bad_tokens = ("best_params", "gridcv_results", "calibrator")
    out = []
    for p in models_dir.glob("*.json"):
        name = p.name.lower()
        if not name.startswith("xgb_"):
            continue
        if any(tok in name for tok in bad_tokens):
            continue
        out.append(p)
    return sorted(out, key=lambda x: x.stat().st_mtime, reverse=True)


def resolve_model_artifacts(models_dir: Path, model_artifact: str | None = None) -> tuple[Path, Path]:
    models_dir = Path(models_dir)
    if model_artifact:
        model_path = Path(model_artifact)
        if not model_path.is_absolute():
            model_path = models_dir / model_path
    else:
        candidates = _candidate_models(models_dir)
        if not candidates:
            raise FileNotFoundError(f"No model artifacts found in {models_dir}")
        model_path = candidates[0]

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    stem = model_path.stem
    sibling_feat = model_path.with_name(f"{stem}_features.txt")
    default_feat = models_dir / "xgb_player_level_features.txt"
    if sibling_feat.exists():
        feat_path = sibling_feat
    elif default_feat.exists():
        feat_path = default_feat
    else:
        feature_candidates = sorted(models_dir.glob("*features*.txt"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not feature_candidates:
            raise FileNotFoundError(f"No feature list artifact found in {models_dir}")
        feat_path = feature_candidates[0]

    return model_path, feat_path
