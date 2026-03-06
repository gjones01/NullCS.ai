from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(r"C:\NullCS\huggfacedata")
SPLITS = ("no_cheater_present", "with_cheater_present")
REPORT_PATH = Path(r"C:\NullCS\main\data\processed\reports\cs2cd_schema_report.md")
LABEL_TERMS = ("cheater", "is_cheater", "banned", "vac", "label", "target", "flag")


@dataclass
class SplitAudit:
    split: str
    parquet_files: list[Path]
    json_files: list[Path]
    sampled_parquet: list[Path]
    sampled_json: list[Path]
    parquet_schema: dict[str, str]
    parquet_label_cols: list[str]
    json_top_keys: dict[str, list[str]]
    json_nested_examples: dict[str, dict[str, list[str]]]
    json_label_keys: dict[str, list[str]]
    cheaters_shapes: dict[str, str]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Inspect CS2CD schema and label availability.")
    ap.add_argument("--sample-n", type=int, default=5, help="Number of parquet/json files to sample per split.")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def _is_label_like(name: str) -> bool:
    n = name.lower()
    return any(term in n for term in LABEL_TERMS)


def _sorted_rglob(root: Path, pattern: str) -> list[Path]:
    return sorted([p for p in root.rglob(pattern) if p.is_file()])


def _sample_paths(paths: list[Path], n: int, rng: random.Random) -> list[Path]:
    if len(paths) <= n:
        return paths
    return sorted(rng.sample(paths, n))


def _dtype_map(df: pd.DataFrame) -> dict[str, str]:
    return {k: str(v) for k, v in df.dtypes.to_dict().items()}


def _json_nested_example(obj: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for k, v in obj.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            out[k] = sorted(v[0].keys())[:20]
    return out


def _find_json_label_keys(obj: dict[str, Any]) -> list[str]:
    hits = [k for k in obj.keys() if _is_label_like(k)]
    for k, v in obj.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            for nk in v[0].keys():
                if _is_label_like(nk):
                    hits.append(f"{k}.{nk}")
    return sorted(set(hits))


def audit_split(split: str, sample_n: int, rng: random.Random) -> SplitAudit:
    split_root = ROOT / split
    parquet_files = _sorted_rglob(split_root, "*.parquet")
    json_files = _sorted_rglob(split_root, "*.json")
    sampled_parquet = _sample_paths(parquet_files, sample_n, rng)
    sampled_json = _sample_paths(json_files, sample_n, rng)

    parquet_schema: dict[str, str] = {}
    for p in sampled_parquet:
        df = pd.read_parquet(p)
        parquet_schema.update(_dtype_map(df))
    parquet_label_cols = sorted([c for c in parquet_schema if _is_label_like(c)])

    json_top_keys: dict[str, list[str]] = {}
    json_nested_examples: dict[str, dict[str, list[str]]] = {}
    json_label_keys: dict[str, list[str]] = {}
    cheaters_shapes: dict[str, str] = {}

    for p in sampled_json:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            json_top_keys[p.name] = [f"<non-dict:{type(obj).__name__}>"]
            json_nested_examples[p.name] = {}
            json_label_keys[p.name] = []
            continue

        json_top_keys[p.name] = sorted(obj.keys())
        json_nested_examples[p.name] = _json_nested_example(obj)
        json_label_keys[p.name] = _find_json_label_keys(obj)

        if "cheaters" in obj:
            val = obj["cheaters"]
            if isinstance(val, list) and val and isinstance(val[0], dict):
                keys = sorted(val[0].keys())
                cheaters_shapes[p.name] = f"list[dict] keys={keys}"
            else:
                cheaters_shapes[p.name] = f"{type(val).__name__}"

    return SplitAudit(
        split=split,
        parquet_files=parquet_files,
        json_files=json_files,
        sampled_parquet=sampled_parquet,
        sampled_json=sampled_json,
        parquet_schema=parquet_schema,
        parquet_label_cols=parquet_label_cols,
        json_top_keys=json_top_keys,
        json_nested_examples=json_nested_examples,
        json_label_keys=json_label_keys,
        cheaters_shapes=cheaters_shapes,
    )


def infer_answers(audits: list[SplitAudit]) -> tuple[bool, str, str]:
    has_per_player = False
    steamid_field = "unknown"
    detail = "No explicit per-player label field detected in sampled files."

    for audit in audits:
        for _, shape in audit.cheaters_shapes.items():
            if "keys=['steamid']" in shape or "keys=[\'steamid\']" in shape:
                has_per_player = True
                steamid_field = "cheaters[].steamid (JSON top-level field)"
                detail = (
                    "Per-player labels are present in sampled files: each match JSON contains "
                    "a top-level `cheaters` list with per-player `steamid` entries."
                )
                break
        if has_per_player:
            break

    if has_per_player:
        return True, steamid_field, detail
    return False, steamid_field, "Only match-level split labels detected (no per-player boolean field found)."


def write_report(
    audits: list[SplitAudit],
    has_per_player: bool,
    steamid_field: str,
    explicit_answer: str,
    out_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# CS2CD Schema Audit")
    lines.append("")
    lines.append(f"- Dataset root: `{ROOT}`")
    lines.append(f"- Report generated: `{pd.Timestamp.utcnow().isoformat()}` UTC")
    lines.append("")
    lines.append("## Explicit Answers")
    lines.append("")
    lines.append(f"- Is there a per-player cheater boolean/label? **{'Yes' if has_per_player else 'No'}**")
    lines.append(f"- If yes: what field maps to steamid? **{steamid_field}**")
    lines.append(f"- If labels are only match-level: {'Yes' if not has_per_player else 'No'}")
    lines.append(f"- Notes: {explicit_answer}")
    lines.append("")

    for a in audits:
        lines.append(f"## Split: `{a.split}`")
        lines.append("")
        lines.append(f"- Files discovered: parquet={len(a.parquet_files)}, json={len(a.json_files)}")
        lines.append(f"- Sampled parquet files: {[p.name for p in a.sampled_parquet]}")
        lines.append(f"- Sampled json files: {[p.name for p in a.sampled_json]}")
        lines.append("")
        lines.append("### Parquet Schema (union of sampled files)")
        lines.append("")
        for col, dtype in sorted(a.parquet_schema.items()):
            lines.append(f"- `{col}`: `{dtype}`")
        lines.append("")
        lines.append(f"- Label-like parquet fields: {a.parquet_label_cols if a.parquet_label_cols else 'none'}")
        lines.append("")
        lines.append("### JSON Keys (sampled)")
        lines.append("")
        for fname in sorted(a.json_top_keys.keys()):
            lines.append(f"- `{fname}` top-level keys: {a.json_top_keys[fname]}")
            nested = a.json_nested_examples.get(fname, {})
            if nested:
                one_key = sorted(nested.keys())[0]
                lines.append(f"- `{fname}` nested example: `{one_key}` -> {nested[one_key]}")
            labels = a.json_label_keys.get(fname, [])
            lines.append(f"- `{fname}` label-like keys: {labels if labels else 'none'}")
            if fname in a.cheaters_shapes:
                lines.append(f"- `{fname}` `cheaters` shape: {a.cheaters_shapes[fname]}")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    audits = [audit_split(split, args.sample_n, rng) for split in SPLITS]
    has_per_player, steamid_field, explicit_answer = infer_answers(audits)

    for a in audits:
        print(f"\n=== {a.split} ===")
        print(f"[INFO] files: parquet={len(a.parquet_files)} json={len(a.json_files)}")
        print(f"[INFO] sampled parquet: {[p.name for p in a.sampled_parquet]}")
        print(f"[INFO] sampled json: {[p.name for p in a.sampled_json]}")
        print(f"[INFO] label-like parquet fields: {a.parquet_label_cols if a.parquet_label_cols else 'none'}")
        for fname in sorted(a.json_label_keys.keys()):
            hits = a.json_label_keys[fname]
            if hits:
                print(f"[INFO] {fname} label-like json keys: {hits}")
            if fname in a.cheaters_shapes:
                print(f"[INFO] {fname} cheaters shape: {a.cheaters_shapes[fname]}")

    print("\n=== Explicit Answer ===")
    print(f"Per-player cheater label present: {'YES' if has_per_player else 'NO'}")
    print(f"SteamID mapping field: {steamid_field}")
    print(explicit_answer)
    if not has_per_player:
        print("Labels appear to be match-level only.")

    write_report(audits, has_per_player, steamid_field, explicit_answer, REPORT_PATH)
    print(f"\n[OK] wrote report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
