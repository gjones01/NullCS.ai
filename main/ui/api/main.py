from __future__ import annotations

import datetime as dt
import json
import subprocess
import threading
import uuid
from pathlib import Path
import sys

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    INFER_CSV,
    JOBS_PATH,
    MODEL_ARTIFACT,
    PROCESSED_DIR,
    PROJECT_ROOT,
    PYTHON_EXE,
    RAW_UPLOADS_DIR,
    REPORTS_DIR,
    SCRIPTS_DIR,
    STATE_DIR,
)

app = FastAPI(title="ClarityCS UI API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    # Vite dev server may auto-increment ports (5173, 5174, 5175, ...).
    # Allow localhost loopback origins on any port to prevent browser fetch failures.
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_jobs_lock = threading.Lock()
PIPELINE_STEPS = ["Uploading", "Parsing", "Feature Build", "Model", "Explanation"]


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _demo_id() -> str:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"TEST_{ts}_{uuid.uuid4().hex[:8]}"


def _tail(path: Path, n: int = 120) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def _load_jobs() -> dict:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if not JOBS_PATH.exists():
        return {}
    try:
        return json.loads(JOBS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_jobs(jobs: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_PATH.write_text(json.dumps(jobs, indent=2), encoding="utf-8")


def _set_job(demo_id: str, **fields) -> dict:
    with _jobs_lock:
        jobs = _load_jobs()
        cur = jobs.get(demo_id, {})
        cur.update(fields)
        cur["updated_at"] = _now()
        jobs[demo_id] = cur
        _save_jobs(jobs)
        return cur


def _get_job(demo_id: str) -> dict | None:
    with _jobs_lock:
        return _load_jobs().get(demo_id)


def _infer_pipeline_step(state: str, logs_tail: str) -> dict:
    stage_idx = 0
    txt = (logs_tail or "").lower()
    if "[info] parsed zip:" in txt:
        stage_idx = 2
    if "engagement_features.parquet" in txt:
        stage_idx = 3
    if "ranked_players_infer.csv" in txt:
        stage_idx = 4
    if state == "done":
        stage_idx = 4
    if state == "error":
        stage_idx = max(stage_idx, 1)
    stage_idx = max(0, min(stage_idx, len(PIPELINE_STEPS) - 1))
    return {"stage_index": stage_idx, "stage": PIPELINE_STEPS[stage_idx], "steps": PIPELINE_STEPS}


def _run_pipeline_background(demo_id: str, dem_path: Path) -> None:
    log_dir = REPORTS_DIR / demo_id / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "pipeline.log"
    cmd = [
        str(PYTHON_EXE),
        str(SCRIPTS_DIR / "run_infer_pipeline.py"),
        "--dem_path",
        str(dem_path),
        "--demo_id",
        demo_id,
        "--out_dir",
        str(PROCESSED_DIR),
    ]
    if MODEL_ARTIFACT:
        cmd.extend(["--model-artifact", MODEL_ARTIFACT])
    _set_job(demo_id, state="running", cmd=cmd, log_path=str(log_path), error="")
    with log_path.open("a", encoding="utf-8", errors="replace") as lf:
        lf.write(f"[{_now()}] START {' '.join(cmd)}\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _set_job(demo_id, pid=proc.pid)
        code = proc.wait()
        lf.write(f"\n[{_now()}] EXIT_CODE {code}\n")
    if code == 0:
        _set_job(demo_id, state="done", return_code=code)
    else:
        _set_job(demo_id, state="error", return_code=code, error=f"pipeline failed with exit code {code}")


def _ensure_demo_dem_path(demo_id: str) -> Path:
    p = RAW_UPLOADS_DIR / demo_id / f"{demo_id}.dem"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Uploaded demo not found for {demo_id}")
    return p


def _safe_report_dir(demo_id: str, steamid: str) -> Path:
    base = REPORTS_DIR.resolve()
    target = (REPORTS_DIR / str(demo_id).strip() / str(steamid).strip()).resolve()
    if base != target and base not in target.parents:
        raise HTTPException(status_code=400, detail="Invalid report path")
    return target


def _safe_evidence_path(demo_id: str, steamid: str, filename: str) -> Path:
    name = Path(filename).name
    if name != filename or not name.lower().endswith(".csv") or not name.startswith("evidence_"):
        raise HTTPException(status_code=400, detail="Invalid evidence filename")
    p = _safe_report_dir(demo_id, steamid) / name
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Evidence file not found: {name}")
    return p


def _load_debug_trace(demo_id: str) -> dict:
    debug_path = REPORTS_DIR / str(demo_id) / "debug_score_trace.json"
    if not debug_path.exists():
        raise HTTPException(status_code=404, detail=f"Debug trace not found: {debug_path}")
    try:
        trace = json.loads(debug_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse debug trace: {e}")
    return {"path": str(debug_path), "trace": trace}


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/upload-demo")
async def upload_demo(file: UploadFile = File(...), demo_id: str | None = Form(default=None)) -> dict:
    demo_id = (demo_id or "").strip() or _demo_id()
    if not str(file.filename or "").lower().endswith(".dem"):
        raise HTTPException(status_code=400, detail="Only .dem uploads are supported")
    out_dir = RAW_UPLOADS_DIR / demo_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{demo_id}.dem"
    content = await file.read()
    out_path.write_bytes(content)
    _set_job(demo_id, state="queued", log_path=str(REPORTS_DIR / demo_id / "logs" / "pipeline.log"), error="")
    return {"demo_id": demo_id}


@app.post("/demo/{demo_id}/run")
def run_demo(demo_id: str) -> dict:
    dem_path = _ensure_demo_dem_path(demo_id)
    cur = _get_job(demo_id) or {}
    if cur.get("state") == "running":
        return {"demo_id": demo_id, "state": "running"}
    _set_job(demo_id, state="queued", error="")
    t = threading.Thread(target=_run_pipeline_background, args=(demo_id, dem_path), daemon=True)
    t.start()
    return {"demo_id": demo_id, "state": "queued"}


@app.get("/demo/{demo_id}/status")
def demo_status(demo_id: str) -> dict:
    cur = _get_job(demo_id)
    if not cur:
        logs_tail = ""
        state = "queued"
        step = _infer_pipeline_step(state, logs_tail)
        return {"demo_id": demo_id, "state": state, "logs_tail": logs_tail, "error": "", **step}

    log_path = Path(cur.get("log_path", ""))
    logs_tail = _tail(log_path)
    state = cur.get("state", "queued")
    step = _infer_pipeline_step(state, logs_tail)
    return {"demo_id": demo_id, "state": state, "logs_tail": logs_tail, "error": cur.get("error", ""), **step}


def _row_summary(row: pd.Series) -> dict:
    def f(name: str) -> float | None:
        if name not in row or pd.isna(row[name]):
            return None
        return float(row[name])

    return {
        "rt_median": f("rt_median"),
        "prefire_rate": f("prefire_rate"),
        "thrusmoke_kill_rate": f("thrusmoke_kill_rate"),
        "headshot_rate": f("headshot_rate"),
        "long_range_fast_rt_rate_4": f("long_range_fast_rt_rate_4"),
    }


def _parse_top_reasons(val) -> list[dict]:
    if isinstance(val, list):
        return val
    if isinstance(val, str) and val.strip():
        try:
            x = json.loads(val)
            if isinstance(x, list):
                return x
        except Exception:
            return []
    return []


@app.get("/demo/{demo_id}/players")
def demo_players(demo_id: str, debug: int = Query(default=0)) -> dict:
    if not INFER_CSV.exists():
        raise HTTPException(status_code=404, detail=f"Inference CSV not found: {INFER_CSV}")
    df = pd.read_csv(INFER_CSV)
    d = df[df["demo_id"].astype(str) == str(demo_id)].copy()
    if d.empty:
        return {"demo_id": demo_id, "players": []}
    sort_col = "risk" if "risk" in d.columns else "proba_cheater_infer"
    d = d.sort_values(sort_col, ascending=False)
    players = []
    for _, r in d.iterrows():
        risk = float(r.get("risk", r.get("proba_calibrated", r.get("proba_cheater_infer", 0.0))))
        players.append(
            {
                "steamid": str(r.get("attacker_steamid", "")),
                "attacker_name": str(r.get("attacker_name", "")),
                "proba_cheater_infer": float(r.get("proba_cheater_infer", 0.0)),
                "risk": risk,
                "confidence": float(r.get("confidence", 0.0)),
                "ci_low": (None if pd.isna(r.get("ci_low")) else float(r.get("ci_low"))),
                "ci_high": (None if pd.isna(r.get("ci_high")) else float(r.get("ci_high"))),
                "risk_band": str(r.get("risk_band", "")),
                "top_reasons": _parse_top_reasons(r.get("top_reasons")),
                "features_summary": _row_summary(r),
            }
        )
    resp: dict[str, object] = {"demo_id": demo_id, "players": players}

    if int(debug) == 1:
        try:
            loaded = _load_debug_trace(demo_id)
            debug_trace = loaded["trace"]
            debug_path = loaded["path"]
        except HTTPException as e:
            debug_trace = {"error": str(e.detail)}
            debug_path = str(REPORTS_DIR / str(demo_id) / "debug_score_trace.json")
        resp["debug"] = {
            "enabled": True,
            "path": debug_path,
            "trace": debug_trace,
        }

    return resp


@app.get("/demo/{demo_id}/player/{steamid}/score-trace")
def player_score_trace(demo_id: str, steamid: str) -> dict:
    loaded = _load_debug_trace(demo_id)
    trace = loaded["trace"]
    players = trace.get("players", []) if isinstance(trace, dict) else []
    s = str(steamid).strip()
    hit = None
    for p in players:
        if str(p.get("steamid", "")).strip() == s:
            hit = p
            break
    if hit is None:
        raise HTTPException(status_code=404, detail=f"No score trace found for steamid {steamid} in demo {demo_id}")
    return {"demo_id": demo_id, "steamid": s, "debug_path": loaded["path"], "trace": hit}


@app.post("/demo/{demo_id}/player/{steamid}/explain")
def explain_player(demo_id: str, steamid: str) -> dict:
    log_dir = REPORTS_DIR / demo_id / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"explain_{steamid}.log"
    cmd = [
        str(PYTHON_EXE),
        str(SCRIPTS_DIR / "explain_demo.py"),
        "--demo",
        str(demo_id),
        "--steamid",
        str(steamid),
        "--mode",
        "infer",
    ]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    log_path.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Explain failed. See {log_path}")
    out_dir = _safe_report_dir(demo_id, steamid)
    return {
        "demo_id": demo_id,
        "steamid": steamid,
        "report_dir": str(out_dir),
        "report_json": str(out_dir / "report.json"),
        "reasons_json": str(out_dir / "reasons.json"),
        "evidence_files": [p.name for p in sorted(out_dir.glob("evidence_*.csv"))],
    }


@app.get("/api/explain")
def api_explain(
    demo_id: str = Query(...),
    steamid: str = Query(...),
    mode: str = Query(default="infer"),
    ci: int = Query(default=0),
    n_boot: int = Query(default=100),
) -> dict:
    mode = str(mode).strip().lower()
    if mode not in {"oof", "insample", "infer"}:
        raise HTTPException(status_code=400, detail="mode must be one of: oof, insample, infer")

    try:
        if str(PROJECT_ROOT / "main") not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT / "main"))
        from src.utils.explain_demo import default_config, explain_demo

        cfg = default_config(mode=mode)
        explain_demo(
            cfg,
            demo_id=str(demo_id),
            steamid=str(steamid),
            with_ci=bool(int(ci)),
            n_boot=int(max(20, n_boot)),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explain failed: {e}") from e

    out_dir = _safe_report_dir(str(demo_id), str(steamid))
    report_path = out_dir / "report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"Missing report.json at {report_path}")
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse report.json: {e}") from e


@app.get("/demo/{demo_id}/player/{steamid}/report/files")
def player_report_files(demo_id: str, steamid: str) -> dict:
    out_dir = _safe_report_dir(demo_id, steamid)
    if not out_dir.exists():
        raise HTTPException(status_code=404, detail=f"Report folder not found: {out_dir}")
    return {
        "demo_id": demo_id,
        "steamid": steamid,
        "report_dir": str(out_dir),
        "report_exists": (out_dir / "report.json").exists(),
        "reasons_exists": (out_dir / "reasons.json").exists(),
        "top_row_exists": (out_dir / "top_player_row.json").exists(),
        "evidence_files": [p.name for p in sorted(out_dir.glob("evidence_*.csv"))],
    }


@app.get("/demo/{demo_id}/player/{steamid}/report/reasons")
def player_report_reasons(demo_id: str, steamid: str) -> dict:
    out_dir = _safe_report_dir(demo_id, steamid)
    reasons_path = out_dir / "reasons.json"
    if not reasons_path.exists():
        raise HTTPException(status_code=404, detail=f"Reasons file not found: {reasons_path}")
    try:
        reasons = json.loads(reasons_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse reasons.json: {e}") from e
    if not isinstance(reasons, list):
        reasons = []
    return {"demo_id": demo_id, "steamid": steamid, "reasons": reasons}


@app.get("/demo/{demo_id}/player/{steamid}/report/evidence/{filename}")
def player_report_evidence(demo_id: str, steamid: str, filename: str, limit: int = 500) -> dict:
    path = _safe_evidence_path(demo_id, steamid, filename)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse CSV {filename}: {e}") from e

    lim = max(1, min(int(limit), 5000))
    if len(df) > lim:
        df = df.head(lim)
    return {
        "demo_id": demo_id,
        "steamid": steamid,
        "filename": filename,
        "columns": [str(c) for c in df.columns.tolist()],
        "rows": df.where(pd.notna(df), None).to_dict(orient="records"),
        "row_count": len(df),
    }


@app.get("/demo/{demo_id}/player/{steamid}/report")
def player_report(demo_id: str, steamid: str) -> dict:
    out_dir = _safe_report_dir(demo_id, steamid)
    if not out_dir.exists():
        raise HTTPException(status_code=404, detail=f"Report folder not found: {out_dir}")
    reasons_path = out_dir / "reasons.json"
    reasons = json.loads(reasons_path.read_text(encoding="utf-8")) if reasons_path.exists() else []
    evidence = []
    for p in sorted(out_dir.glob("evidence_*.csv")):
        df = pd.read_csv(p)
        evidence.append(
            {
                "filename": p.name,
                "columns": [str(c) for c in df.columns.tolist()],
                "row_count": int(len(df)),
                "preview": df.head(50).where(pd.notna(df), None).to_dict(orient="records"),
            }
        )
    return {"demo_id": demo_id, "steamid": steamid, "report_dir": str(out_dir), "reasons": reasons, "evidence": evidence}
