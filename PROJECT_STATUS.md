# Axiom — Project Status & Handoff

> **Purpose of this file:** a living "where we are" document so that after a reboot
> (or a new chat session) you don't have to re-explain the project. Update the
> **Done** and **Next** sections as work progresses. Last updated: **2026-06-03**.

---

## What this project is

**Axiom** — an autonomous, multi-agent AI data-science platform. Upload a CSV and
8 specialized agents run the full ML lifecycle (ingest → preprocess → feature
engineering → split → train → error detection → improvement → finalize) and
produce trained models, SHAP explainability, visualizations, and reports.

- **Backend:** FastAPI (`api.py`) on `http://127.0.0.1:8000`, SQLite DB
  (`data/axiom.db`), CrewAI agents, Cerebras LLM via LiteLLM.
- **Frontend:** Next.js 16 + TypeScript + Tailwind in `frontend/`, on
  `http://localhost:3000`. Dev server proxies `/api/*` → backend (`next.config.ts`).
- **Auth:** email/password → bcrypt hash + bearer session token (`UserSession`),
  mirrored into an `axiom-auth` cookie so the server-side gate can read it.

---

## How to run everything (backend + frontend + auth)

From the **project root** (`C:\Users\ACER\Desktop\Data-AI`):

```powershell
./start.ps1          # PowerShell — installs frontend deps on first run, then starts both
```
or just double-click **`start.bat`**.

Both launchers run `npm run dev` in `frontend/`, which executes `frontend/dev-all.mjs`
and starts **both** processes together:
- `[api]` FastAPI/uvicorn backend on `:8000`
- `[web]` Next.js dev server on `:3000`

Then open **http://localhost:3000**. Press **Ctrl+C** to stop both.

**Backend prerequisites:** Python venv at `venv/` with deps from `requirements.txt`,
and a `CEREBRAS_API_KEY` in `.env`. If the venv is missing the launcher falls back
to system `python` on PATH.

---

## Auth / routing flow (current behaviour)

The gate lives in **`frontend/src/proxy.ts`** (Next 16 renamed `middleware` → `proxy`).

```
open app (any URL) ──unauthenticated──▶ /auth  (sign in / sign up)
                                          │
                              sign in / sign up
                                          ▼
                                        /  (landing page)
                                          │  user clicks a CTA
                              ┌───────────┴───────────┐
                              ▼                       ▼
                           /free                 /enterprise
```

- **Everything is gated**, including the landing page `/`. An unauthenticated
  visitor is redirected to `/auth` first — the auth page always shows before the
  landing page.
- After sign in / sign up the user is sent to `/` (the landing page). Their saved
  workspace mode is still restored into the store in the background.
- A deep link to a protected page (e.g. `/free`) is remembered via `?next=` and
  honored after auth.
- An already-authenticated user who hits `/auth` is bounced to `/`.

Key files for this flow:
- `frontend/src/proxy.ts` — the server-side auth gate (route protection).
- `frontend/src/app/auth/page.tsx` — sign-in/sign-up UI + post-auth redirect.
- `frontend/src/store/useAuthStore.ts` + `frontend/src/lib/cookies.ts` — token
  persistence and the `axiom-auth` cookie mirror the gate reads.

---

## API endpoints (all verified consistent with the frontend client)

Backend routes in `api.py` ↔ frontend calls in `frontend/src/lib/api.ts`:

| Endpoint | Method | Used by (frontend fn) |
|---|---|---|
| `/api/auth/signup` | POST | `authSignup` |
| `/api/auth/login` | POST | `authLogin` |
| `/api/auth/logout` | POST | `authLogout` |
| `/api/auth/me` | GET | (available; not yet called) |
| `/api/auth/history` | GET | `getPromptHistory` |
| `/api/workspace` | GET | `getWorkspace` |
| `/api/workspace/mode` | POST | `setWorkspaceMode` |
| `/api/upload` | POST | `uploadDataset` |
| `/api/init-run` | POST | `initRun` |
| `/api/run` | POST | `startPipeline` |
| `/api/workflow/run` | POST | `runWorkflow` |
| `/api/agent/{name}/run` | POST | `runSingleAgent` |
| `/api/agent/{name}/output/{run_id}` | GET | `getAgentOutput` |
| `/api/status/{run_id}` | GET | `getStatus` |
| `/api/results/{run_id}` | GET | `getResults` |
| `/api/report/{run_id}` | GET | `getReport` |
| `/api/report/{run_id}/pdf` | GET | `downloadReportPdf` |
| `/api/shap/{run_id}` | GET | `getShapData` |
| `/api/runs` | GET | `listExperiments` |
| `/api/visualizations/{run_id}` | GET | `getVisualizations` |
| `/api/visualizations/{run_id}/generate` | POST | `generateVisualization` |
| `/api/export/{run_id}/excel` | GET | `exportRunExcel` |

Security notes (do **not** regress — see memory `production_hardening.md`):
- Every run-scoped endpoint takes `user = Depends(get_current_user)` and checks
  ownership via `_require_run` / `_authorize_run_id`.
- CORS is locked to `ALLOWED_ORIGINS` env (never wildcard with credentials).
- Run state persists to `artifacts/<run_id>/run_state.json` and survives restarts.

---

## Done (most recent first)

- **2026-06-03 — Auth-first routing.** Made the landing page `/` require auth so
  the `/auth` page always shows before the landing page. Updated `proxy.ts`
  (gate `/`, send authed users on `/auth` → `/`) and `auth/page.tsx` (post-auth
  redirect now lands on `/`). Added root launchers `start.ps1` / `start.bat`.
  Verified `tsc --noEmit` clean. Created this status file.
- **2026-06-03 — Production hardening.** Authz on all run endpoints, env-driven
  CORS, run-state disk persistence, secret scrub, secure session tokens, path
  safety. (See `production_hardening.md`.)
- **Frontend V2 redesign (phase 1).** Near-black + electric-blue palette,
  `AgentNetwork` component, Mission Control pipeline page. (See
  `frontend_v2_redesign.md`.)

---

## Next / TODO

- [ ] **Manual smoke test the new flow:** run `./start.ps1`, confirm
      `localhost:3000` → `/auth`, sign up, confirm you land on `/`, then `/free`.
- [ ] Decide whether an authed user on the landing page should see a "Go to
      workspace / Logout" affordance in the Navbar (currently CTAs go to
      `/free` / `/enterprise`).
- [ ] Results pages (`app/free/results/[runId]`, `app/report/[runId]`) — executive
      summary, model leaderboard, SHAP, export center not yet redesigned.
- [ ] Enterprise surfaces (agent console, workflow builder, analytics,
      experiments) — restyled via tokens but not structurally rebuilt.
- [ ] Wire `/api/auth/me` for a token-refresh / session-validation check on load.

---

## Gotchas / things to remember

- This is **Next.js 16** with breaking changes — read `frontend/node_modules/next/dist/docs/`
  before writing Next code (see `frontend/AGENTS.md`). The auth gate file is
  `proxy.ts`, **not** `middleware.ts`.
- The backend also defines `GET /` serving a legacy dashboard HTML, but in dev the
  Next.js frontend owns `/` (only `/api/*` is proxied to the backend), so there's
  no conflict.
- `.env` holds the real `CEREBRAS_API_KEY`; `.env.example` has a placeholder only.
