# ClarityCS / NullCS Code Ownership Map

## 1) High-level architecture (who calls what)

```text
React Web UI (main/ui/web)
  |
  | HTTP (upload, run, status, players, explain, report)
  v
FastAPI Backend (main/ui/api/main.py)
  |
  | subprocess: python main/scripts/run_infer_pipeline.py
  v
Inference Pipeline (single demo)
  1) parse_demos_awpy_api.parse_one(.dem -> parsed_zips/<demo>.zip)
  2) build_engagement_features.build_for_zip(zip -> demos/<demo>/engagement_features.parquet)
  3) aggregate logic (in run_infer_pipeline.py; training-compatible player features)
  4) XGBoost load + score (model_registry + scoring utils)
  5) reports/<demo>/ranked_players_infer.csv + debug_score_trace.json
  |
  | optional subprocess: python main/scripts/explain_demo.py --mode infer
  v
Explainability (main/src/utils/explain_demo.py)
  -> report.json, reasons.json, evidence_*.csv
```

Offline training/eval path (not called by UI automatically):

```text
Raw demos -> parse_demos_awpy_api.py -> parsed_zips/*.zip
          -> build_engagement_features.py -> main/data/processed/demos/*/engagement_features.parquet
          -> aggregate_player_features.py -> player_features.parquet
          -> train_xgb_gridcv.py -> models/*.json + feature list
          -> evaluate_xgb_gridcv.py -> reports/ranked_*_oof.csv (or insample)
          -> explain_demo.py --mode oof|insample
```

## 2) Key folders/files and ownership

### Product surfaces
- `main/ui/web/`: React + Vite frontend. API client is in `main/ui/web/src/api.ts`.
- `main/ui/api/`: FastAPI backend. Orchestrates upload/run/status/report endpoints and shells out to pipeline scripts.

### Inference pipeline (single uploaded demo)
- `main/scripts/run_infer_pipeline.py`: Primary UI-serving inference entrypoint. Copies uploaded `.dem`, parses, builds engagement features, aggregates per-player features, loads model, scores risk, writes report artifacts.
- `main/src/utils/model_registry.py`: Resolves which model JSON + feature list text file to use.
- `main/src/utils/scoring.py`: Risk post-processing logic (confidence score, optional calibration, low-evidence downweight, risk bands, top reason tags).
- `main/scripts/explain_demo.py`: CLI wrapper for explain output generation.
- `main/src/utils/explain_demo.py`: Core explanation writer (`report.json`, `reasons.json`, `evidence_*.csv`).

### Feature extraction / data engineering
- `main/src/parse/parse_demos_awpy_api.py`: AWPy parse of `.dem` files into zip bundles (`kills.parquet`, `shots.parquet`, `ticks.parquet`, etc. + `header.json`).
- `main/src/features/build_engagement_features.py`: Kill/visibility/reaction-time feature builder per demo (`engagement_features.parquet`).
- `main/src/features/aggregate_player_features.py`: Aggregates engagement rows to player-demo feature rows + labels (`player_features.parquet`).
- `main/src/parse/build_events_from_zips.py`: Alternate event builder (`events.parquet`), currently writes to `C:\NullCS\processed\demos` (different tree than training/inference defaults).

### Model training/evaluation
- `main/scripts/train_xgb_gridcv.py`: Grouped CV grid search training; writes model + feature list + CV results.
- `main/scripts/evaluate_xgb_gridcv.py`: OOF evaluation and ranking outputs (or optional in-sample via `--use-saved-model`).
- `main/scripts/calibrate_model.py`: Fits calibration artifact used by scoring when present.
- `main/scripts/bootstrap_demo_ci.py`: CI estimation for player-demo risk.

### Legacy/alternate branches (not the main owner path)
- `Tick Level/`, `cheat_training/`, `Post 100 Scripts New/`, top-level scripts in repo root: historical/experimental pipelines.

## 3) Environment variables and what they control

Primary runtime vars:
- `CLARITY_PROJECT_ROOT`: Overrides backend project root used by API config.
- `CLARITY_PYTHON_EXE`: Python interpreter path backend uses to run scripts.
- `CLARITY_RAW_UPLOADS_DIR`: Upload storage root for `.dem` files.
- `CLARITY_PROCESSED_DIR`: Processed data root (`demos/`, `reports/`, etc.).
- `CLARITY_SCRIPTS_DIR`: Where backend expects runnable scripts.
- `CLARITY_MODEL_ARTIFACT`: Model filename/path override for inference scripts.
- `VITE_API_BASE`: Frontend API base URL (default `http://127.0.0.1:8000`).

AWPy asset var:
- `AWPY_HOME`: Used by visibility/map-asset utility scripts (`main/src/utils/visibility_awpy.py`, `awpy_map_assets.py`, `los_sanity_test.py`) to locate `tris` assets.

Notes:
- Most core pipeline paths are still hardcoded to `C:\NullCS\...` in scripts.
- No central `.env` loader is present; values are read directly with `os.getenv`/`os.environ`.

## 4) Exact commands (dev + build + tests)

Run from repo root `C:\NullCS` unless noted.

### UI/API development
1. Backend dependencies:
```powershell
pip install -r main/ui/api/requirements.txt
```
2. Start backend:
```powershell
python -m uvicorn main.ui.api.main:app --host 127.0.0.1 --port 8000 --reload
```
3. Start frontend (new terminal):
```powershell
cd main/ui/web
npm install
npm run dev
```
4. Frontend production build:
```powershell
cd main/ui/web
npm run build
```

### Offline pipeline (training path)
```powershell
python main/src/parse/parse_demos_awpy_api.py
python main/src/features/build_engagement_features.py
python main/src/features/aggregate_player_features.py
python main/scripts/train_xgb_gridcv.py
python main/scripts/evaluate_xgb_gridcv.py
python main/scripts/explain_demo.py --demo CDemo3 --mode oof
```

### Inference-only single demo (CLI)
```powershell
python main/scripts/run_infer_pipeline.py --dem_path "<path-to-demo.dem>" --demo_id "TEST_YYYYMMDD_HHMMSS_xxxxxxxx" --out_dir "C:\NullCS\main\data\processed"
```

### Tests / smoke checks in this repo
There is no formal `pytest`/`npm test` suite configured.
Use these script-based checks:
1. Determinism test:
```powershell
python main/scripts/check_infer_determinism.py --dem_path "<path-to-demo.dem>"
```
2. Evaluation + explain smoke:
```powershell
python main/scripts/evaluate_xgb_gridcv.py; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; python main/scripts/explain_demo.py --demo CDemo3 --mode oof
```

## 5) If X breaks, check Y first

- If UI cannot talk to backend, check `VITE_API_BASE`, backend URL/port, and CORS (`main/ui/api/main.py`).
- If upload works but run never starts, check `main/ui/api/state/jobs.json` and `main/data/processed/reports/<demo>/logs/pipeline.log`.
- If inference fails at parse step, check AWPy availability in selected Python (`CLARITY_PYTHON_EXE`) and parser output zip at `parsed_zips/<demo>.zip`.
- If feature build fails, check required zip members (`ticks.parquet`, `kills.parquet`, `shots.parquet`) and map LOS assets (`AWPY_HOME` + tris files).
- If model load fails, check `main/data/processed/models` for model JSON + matching feature list, and `CLARITY_MODEL_ARTIFACT` override value.
- If player list is empty, check MIN_KILLS filtering in aggregation logic (`n_kills_with_rt >= 5`).
- If explanation fails, check that ranked infer CSV exists (`reports/ranked_player_demo_suspicion_infer.csv`) and that the requested steamid exists for that demo.
- If offline pipeline scripts disagree, verify path mismatches: `build_events_from_zips.py` writes to `C:\NullCS\processed\demos`, while training/inference uses `C:\NullCS\main\data\processed\...`.
- If `run_pipeline.py` fails immediately, check script paths inside it; it currently references missing files under `main/scripts`.
