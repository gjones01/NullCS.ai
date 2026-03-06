# NullCS

NullCS is a research project for analyzing suspicious Counter-Strike 2 behavior from structured demo data. It leverages statistics and machine learning to flag gameplay moments and provide signals that align with cheating in Counter-Strike 2. Any output produced by NullCS should not be taken as definitive proof of cheating. It is still in Pre-Alpha and still being trained on gameplay data.

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

## Troubleshooting

If you see:

`ERROR: Could not open requirements file: ... main/ui/api/requirements.txt`

you are not in the repo directory yet. Run:

```powershell
cd $HOME\NullCS.ai
```
