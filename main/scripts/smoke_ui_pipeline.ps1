param(
  [Parameter(Mandatory = $true)]
  [string]$DemoPath,
  [string]$DemoId = ""
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $DemoPath)) {
  throw "Demo path not found: $DemoPath"
}

if (-not $DemoId) {
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $DemoId = "SMOKE_$stamp"
}

Write-Host "Running inference pipeline on demo:" $DemoPath
python main/scripts/run_infer_pipeline.py --dem_path "$DemoPath" --demo_id "$DemoId" --out_dir "C:\NullCS\main\data\processed"
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Pipeline complete for demo_id=$DemoId"
Write-Host ""
Write-Host "Start backend (terminal 1):"
Write-Host "  python -m uvicorn main.ui.api.main:app --host 127.0.0.1 --port 8000 --reload"
Write-Host ""
Write-Host "Start frontend (terminal 2):"
Write-Host "  cd main/ui/web"
Write-Host "  npm run dev"
Write-Host ""
Write-Host "Manual UI checks:"
Write-Host "  1) Upload demo works"
Write-Host "  2) Processing reaches done"
Write-Host "  3) Players list loads with Player, SteamID, Analyze"
Write-Host "  4) Analyze shows Reason Signals + severity chips (LOW/MEDIUM/HIGH/CONTEXT)"
Write-Host "  5) Severity Legend is visible on the right and not clipped"
Write-Host "  6) Evidence tabs and table render"
Write-Host "  7) Browser console has no errors"
