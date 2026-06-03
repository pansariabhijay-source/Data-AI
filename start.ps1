# start.ps1 — launch the whole Axiom stack (FastAPI backend + Next.js frontend).
#
# Usage (from the project root):
#   ./start.ps1
#
# This runs `npm run dev` in ./frontend, which in turn starts BOTH:
#   • api  — FastAPI/uvicorn backend (api.py) on http://127.0.0.1:8000
#   • web  — Next.js dev server      on http://localhost:3000
#
# Open http://localhost:3000 — you'll be sent to the /auth sign-in page first,
# then to the landing page after you sign in. Press Ctrl+C to stop both.

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$frontend = Join-Path $root "frontend"

if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
    Write-Host "[start] Installing frontend dependencies (first run)..." -ForegroundColor Yellow
    Push-Location $frontend
    npm install
    Pop-Location
}

Write-Host "[start] Launching Axiom - backend + frontend. Browser: http://localhost:3000" -ForegroundColor Cyan
Push-Location $frontend
try {
    npm run dev
} finally {
    Pop-Location
}
