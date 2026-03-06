# Status checkpoint

## What changed (today)
- Fixed data leakage:
  - attacker_steamid removed from model features
  - evaluation now defaults to grouped OOF (true eval)
  - added guardrails to fail if forbidden columns appear
  - separated outputs: *_oof.csv vs *_insample.csv
- Added engagement-level signals:
  - `is_thrusmoke` from kills data
  - `is_prefire` (`first_shot_tick < t0_visible`)
- Added player aggregation features:
  - `thrusmoke_kills`, `thrusmoke_kill_rate`
  - `thrusmoke_rounds`, `thrusmoke_max_round_streak`
  - `prefire_rate`, `rt_le_2_rate`, `rt_le_4_rate`, `long_range_fast_rt_rate`
  - weapon-family splits:
    - `rifle_kill_share`, `pistol_kill_share`, `awp_smg_kill_share`
    - `rifle_fast_rt_rate`, `pistol_fast_rt_rate`, `awp_smg_fast_rt_rate`
- Added CDemo ranking metrics in eval:
  - prints Top1/Top3 cheater hit rates across CDemo demos

## What to run
1) Rebuild engagement features:
   - python main/src/features/build_engagement_features.py
2) Rebuild player features:
   - python main/src/features/aggregate_player_features.py
3) Train:
   - python main/scripts/train_xgb_gridcv.py
4) Evaluate (OOF):
   - python main/scripts/evaluate_xgb_gridcv.py
Outputs:
- main/data/processed/reports/ranked_player_demo_suspicion_oof.csv
- main/data/processed/reports/ranked_demo_suspicion_oof.csv

## Current reality
- OOF results are muddy; pros can still get flagged.
- RT/HS/distance alone behaves like a skill detector.

## Next feature work (highest ROI)
- Tune thresholds:
  - `LONG_RANGE_DIST` in aggregation is currently 1500.0
  - consider map-normalized distance bins instead of one global cutoff
- Expand weapon families:
  - verify full awpy weapon naming coverage and map unknowns
- Add robustness metrics:
  - per-map Top1/Top3 CDemo hit rates
  - calibration plots or Brier score on OOF predictions

## Notes
- Only trust *_oof.csv for evaluation.

---

## 2026-03-06 UI Evidence-First Refactor

### What changed
- Demo players page simplified to evidence workflow only:
  - removed Risk / Confidence / CI columns
  - removed risk/probability-driven ordering in UI
  - kept Player, SteamID, Analyze, and search
- Player analysis page kept:
  - Reason Signals with severity chips (`LOW`, `MEDIUM`, `HIGH`, `CONTEXT`)
  - Severity Legend panel on right
  - Evidence Tables section (search + table)
- Evidence table tab labels are now human-friendly (frontend mapping):
  - `evidence_fast_rt.csv` -> `Fast Reaction Time`
  - `evidence_fast_rt_streak.csv` -> `Fast RT Streaks`
  - `evidence_prefire.csv` -> `Prefires`
  - `evidence_prefire_streak.csv` -> `Prefire Streaks`
  - `evidence_thrusmoke.csv` -> `Through-Smoke Kills`
  - `evidence_headshot_streak.csv` -> `Headshot Streaks`
  - `evidence_long_range_fast_rt_4.csv` -> `Long-Range Fast RT`

### Verification status
- Added smoke helper script:
  - `main/scripts/smoke_ui_pipeline.ps1`
- Updated `main/ui/README.md` with smoke-test command and manual checks for:
  - upload
  - processing
  - players list
  - Analyze page content
  - evidence tabs/table
  - browser console sanity

### Reminder
- This tool is for research/forensics triage and evidence review.
- It is not standalone proof of cheating.

---

## UI/API Fix Note (TypeError: Failed to fetch)

### Symptom
- In the web UI, clicking `Analyze Demo` showed:
  - `TypeError: Failed to fetch`
- This happened even though the backend process could be running.

### Root cause (plain-English)
- The browser blocks cross-origin requests unless the backend explicitly allows the frontend origin (CORS policy).
- The backend previously allowed only:
  - `http://localhost:5173`
  - `http://127.0.0.1:5173`
- Vite dev server can auto-switch ports if 5173 is busy (for example 5175, 5177).
- When frontend ran on `http://localhost:5175`, backend rejected origin at browser layer, so `fetch` failed before app-level JSON error handling.

### What was changed
1) Backend CORS policy made robust for local dev ports:
- File changed: `main/ui/api/main.py`
- Old: fixed allowlist for `:5173`
- New: regex allow for localhost loopback on any port:
  - `^https?://(localhost|127\.0\.0\.1)(:\d+)?$`
- Why: prevents random breakage when Vite picks a different local port.

2) Frontend network error message made more actionable:
- File changed: `main/ui/web/src/api.ts`
- Added try/catch around `fetch(...)` to throw a clearer message:
  - confirms backend URL expected
  - hints CORS/origin mismatch
- Why: quicker diagnosis than raw `TypeError: Failed to fetch`.

### Verification performed (successful)
1) CORS preflight test against backend:
- Sent `OPTIONS /upload-demo` with:
  - `Origin: http://localhost:5175`
  - `Access-Control-Request-Method: POST`
- Result:
  - `HTTP 200`
  - `access-control-allow-origin: http://localhost:5175`
- Interpretation: browser is now allowed to call API from dynamic Vite ports.

2) Existing pipeline/API smoke still good after CORS change:
- Backend boot and endpoint checks still returned valid JSON (health + players + explain).
- Inference and explain flows still produced `risk/confidence/report.json` artifacts.

### Practical run guidance
- Start API:
  - `python -m uvicorn main.ui.api.main:app --host 127.0.0.1 --port 8000 --reload`
- Start UI:
  - `cd main/ui/web`
  - `npm run dev`
- If Vite starts on non-5173 (e.g., 5175/5177), it is now supported without code changes.
