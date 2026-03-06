# ClarityCS

Evidence-first demo analysis UI and pipeline for CS demo forensics research.

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

UI doc and API details: `main/ui/README.md`

## Smoke Test

```powershell
powershell -ExecutionPolicy Bypass -File main/scripts/smoke_ui_pipeline.ps1 -DemoPath "<path-to-demo.dem>"
```

## Public Repository Safety

Training data, raw demos, processed outputs, reports, evidence CSVs, and model artifacts are excluded from version control.
