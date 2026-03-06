# NullCS

Evidence-first demo analysis UI and pipeline for CS demo forensics research.

## Prerequisites

- Python 3.10+ with `python`/`pip` on PATH
- Node.js 18+ with `npm` on PATH
- Git

## First-Time Setup (New User)

From a normal terminal (for example `C:\Users\<you>`), clone the repo first:

```powershell
cd $HOME
git clone https://github.com/gjones01/NullCS.ai.git
cd .\NullCS.ai
```

## Quickstart (Backend + UI)

From repo root (`...\NullCS.ai`):

```powershell
python -m pip install -r .\main\ui\api\requirements.txt
python -m uvicorn main.ui.api.main:app --host 127.0.0.1 --port 8000 --reload
```

In a second terminal:

```powershell
cd $HOME\NullCS.ai\main\ui\web
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

## Troubleshooting

If you see:

`ERROR: Could not open requirements file: ... main/ui/api/requirements.txt`

you are not in the repo directory yet. Run:

```powershell
cd $HOME\NullCS.ai
```
