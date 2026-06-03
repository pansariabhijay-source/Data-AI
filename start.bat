@echo off
REM start.bat - launch the whole Axiom stack (FastAPI backend + Next.js frontend).
REM Double-click this file, or run `start.bat` from the project root.
REM It runs `npm run dev` in .\frontend, which starts BOTH the backend (port 8000)
REM and the Next.js frontend (port 3000). Open http://localhost:3000 - you land on
REM the /auth sign-in page first, then the landing page after signing in.

cd /d "%~dp0frontend"
if not exist "node_modules" (
  echo [start] Installing frontend dependencies (first run)...
  call npm install
)
echo [start] Launching Axiom - backend + frontend. Browser: http://localhost:3000
call npm run dev
