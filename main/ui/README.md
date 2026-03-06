# ClarityCS Local App (`main/ui`)

This is an isolated local app that runs your real ClarityCS inference pipeline (no retraining, no mock data).
The UI is evidence-first: it focuses on player identity, reason signals, and evidence tables.

## Structure

- `main/ui/api` FastAPI backend
- `main/ui/web` React + Vite + TypeScript + Tailwind + Framer Motion frontend
- Inference entrypoint: `main/scripts/run_infer_pipeline.py`

## Backend Config

Edit `main/ui/api/config.py` (or set env vars):

- `PROJECT_ROOT`
- `PYTHON_EXE`
- `RAW_UPLOADS_DIR`
- `PROCESSED_DIR`
- `SCRIPTS_DIR`
- `MODEL_ARTIFACT` (optional; model filename/path, e.g. `xgb_player_level_gridcv.json`)

Defaults:

- Prefer `C:\NullCS\NewAnubisTri\.venv\Scripts\python.exe` if it exists
- Else use `python` from PATH
- If `CLARITY_MODEL_ARTIFACT` is set, API pipeline runs pass `--model-artifact` to inference.

## Run Backend (Windows PowerShell)

From repo root:

```powershell
pip install -r main/ui/api/requirements.txt
python -m uvicorn main.ui.api.main:app --host 127.0.0.1 --port 8000 --reload
```

## Run Frontend (Windows PowerShell)

From repo root:

```powershell
cd main/ui/web
npm install
npm run dev
```

Frontend defaults to API base `http://127.0.0.1:8000`.
Override with:

```powershell
$env:VITE_API_BASE="http://127.0.0.1:8000"
npm run dev
```

## API Endpoints

- `POST /upload-demo` form-data file upload (`.dem`) and optional `demo_id`
- `POST /demo/{demo_id}/run` run inference pipeline in background
- `GET /demo/{demo_id}/status` job state + log tail
- `GET /demo/{demo_id}/players` players from infer CSV
- `POST /demo/{demo_id}/player/{steamid}/explain` run explain script (`--mode infer`)
- `GET /demo/{demo_id}/player/{steamid}/report` reasons + evidence CSV previews

## Outputs

- Raw uploads: `main/data/raw_uploads/<demo_id>/<demo_id>.dem`
- Engagement: `main/data/processed/demos/<demo_id>/engagement_features.parquet`
- Per-demo features: `main/data/processed/demos/<demo_id>/player_features_infer.parquet`
- Global infer output: `main/data/processed/reports/ranked_player_demo_suspicion_infer.csv`
- Per-demo output: `main/data/processed/reports/<demo_id>/ranked_players_infer.csv`
- Explain outputs: `main/data/processed/reports/<demo_id>/<steamid>/...`
- Logs: `main/data/processed/reports/<demo_id>/logs/`

## Direct CLI Dry Run (No Retraining)

```powershell
python main/scripts/run_infer_pipeline.py --dem_path "<path-to-demo.dem>" --demo_id "TEST_20260226_120000_abcd1234" --out_dir "C:\NullCS\main\data\processed"
```

## Smoke Test (Pipeline + UI)

From repo root:

```powershell
powershell -ExecutionPolicy Bypass -File main/scripts/smoke_ui_pipeline.ps1 -DemoPath "<path-to-demo.dem>"
```

What this does:
- Runs inference pipeline once on your sample demo path.
- Prints commands to start backend/frontend.
- Lists manual checks for the upload, player list, analysis page, reason signals, severity legend, and evidence tables.

## Public Repo Note

Data, model artifacts, uploads, and generated reports are intentionally not included in source control for public pushes.
