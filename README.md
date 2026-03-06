# NullCS

NullCS is a research project for analyzing suspicious Counter-Strike 2 behavior from structured demo data. It leverages statistics and machine learning to flag gameplay moments and provide signals that align with cheating in Counter-Strike 2. Any output produced by NullCS should not be taken as definitive proof of cheating. It is still in Pre-Alpha and still being trained on gameplay data.

## Quickstart (Backend + UI)

From repo root:

```powershell
pip install -r main/ui/api/requirements.txt
python -m uvicorn main.ui.api.main:app --host 127.0.0.1 --port 8000 --reload
```

In a second terminal:

```powershell
cd main/ui/web
npm install
npm run dev
```

UI/API details: `main/ui/README.md`

## Smoke Test

```powershell
powershell -ExecutionPolicy Bypass -File main/scripts/smoke_ui_pipeline.ps1 -DemoPath "<path-to-demo.dem>"
```

## Scope

- This repo is for research/tooling and evidence review workflows.
- It is not standalone proof of cheating or an anti-cheat product.

## Public Repository Safety

Raw demos, processed outputs, reports, evidence CSVs, and model artifacts are excluded from source control.
