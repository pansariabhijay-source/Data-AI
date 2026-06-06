# Axiom Frontend — Context & Spec

A detailed map of the frontend: every route, section, and button, what each does,
the data flow, and what we want the UI to be. Read this with the repo-root
`currentdoingandtodo.md` (project-wide context).

---

## 1. What the frontend is

A **Next.js 16 (App Router, Turbopack) + React 19 + TypeScript + Tailwind**
single-page-feel app that drives the Axiom autonomous ML pipeline. Dark, premium
"AI OS" aesthetic (near-black surfaces, electric-blue/teal accent, glass panels,
framer-motion micro-animations).

- **Dev:** `npm run dev` **from the `frontend/` dir** → launches the FastAPI
  backend (:8000) + Next.js (:3000) together via `dev-all.mjs`.
- **API:** the frontend calls `/api/*`; `next.config.ts` rewrites `/api/*` →
  `http://127.0.0.1:8000`. **Exception:** large file **uploads go directly to the
  backend** (`src/lib/api.ts` `backendOrigin()`) to bypass the dev proxy's body
  cap. All requests carry `Authorization: Bearer <token>`.
- **Auth token** is kept in `useAuthStore` (persisted to localStorage) and mirrored
  to an `axiom-auth` cookie so the route guard can read it.

## 2. The two modes

Axiom has two tiers, chosen on the **welcome** screen and persisted per-user via
`/api/workspace`:

- **Free** — one-click, zero-config: upload a CSV → the full pipeline runs → a
  results report. Routes under `/free`.
- **Pro / Enterprise** — full control: workflow builder, step-by-step agent
  console, run history, reports, visualizations, settings. Routes under
  `/enterprise`, wrapped in the `AppShell` (sidebar layout).

`Mode = "free" | "enterprise"` (`src/lib/types.ts`). `useAppStore` holds the
selected mode + `hasSelectedMode`.

---

## 3. Routes / pages (what each section + button does)

### Public

**`/` — Landing (`app/page.tsx`)**
Marketing hero: "Agentic Orchestration", a live **Model Leaderboard** showcase,
the **AGENTS** network animation (`AgentNetwork`). CTA buttons route to `/auth`
(sign in) or into the product.

**`/auth` — Sign in / Sign up (`app/auth/page.tsx`)**
Toggle between login and signup. Fields: username (`ada_lovelace`), email
(`you@company.com`), password (`••••••••`). Calls `authLogin` / `authSignup` →
stores token in `useAuthStore` → routes to `/welcome` (or the last mode).

**`/welcome` — Onboarding / mode picker (`app/welcome/page.tsx`)**
First-run choice between **Free** and **Pro**. Calls `setWorkspaceMode` (persists
server-side), sets `hasSelectedMode`, routes to `/free` or `/enterprise`.

### Free flow

**`/free` — Free home (`app/free/page.tsx`)**
- **Step indicator**: 1 Upload → 2 Set Target → 3 Launch.
- **Upload zone** (drag/drop or "browse files"): client-side guard (≤1 GB, not
  empty) → `uploadDataset()` → shows filename, rows, columns, chart count.
- **Dataset Intelligence** accordion: the auto-generated charts (base64 images).
- **Target Variable** dropdown: pick the column to predict (or "None — unsupervised
  clustering").
- **Data Preview** table (first rows).
- **"Run Full AI Pipeline"** button → `startPipeline()` → routes to
  `/free/pipeline/[runId]`.
- Below: a **"Unlock Pro"** marketing section with a "Switch to Pro Mode" button →
  `/enterprise`.

**`/free/pipeline/[runId]` — Live pipeline (`app/free/pipeline/[runId]/page.tsx`)**
- Polls `/api/status` every 1.5 s.
- **Left rail**: the 8 agents as a vertical timeline — icon, label, and (NEW) each
  completed agent's **real summary facts** (`agentFacts()`, e.g. "rows after:
  200,000 · duplicates removed: 20,000"), plus per-agent duration.
- **Center**: an "Execution Graph" (serpentine 4+4 circuit) showing the active
  stage.
- **Logs** panel (toggle): streamed log lines.
- On completion → auto-redirects to `/free/results/[runId]`.

**`/free/results/[runId]` — Free results (`app/free/results/[runId]/page.tsx`)**
Tabbed report (Overview / Leaderboard / Visualizations / Quality Audit /
Narrative). Header buttons: **Notebook** (`downloadReportNotebook`), **Download
PDF** (`downloadReportPdf`). Narrative renders via `ReportMarkdown`.

### Pro / Enterprise (wrapped in `AppShell` = sidebar + content)

Sidebar (`components/layout/Sidebar.tsx`) links: **Home · Runs · Reports ·
Visualizations · Agents · Workflow Builder · Settings**. `ModeSwitch` toggles
Free/Pro.

**`/enterprise` — Pro dashboard (`app/enterprise/page.tsx`)**
- Greeting + "Continue where you left off".
- **Upload dataset** drop-zone (`uploadDataset`); on success routes to
  `/enterprise/workflow`. Shows an inline error if upload fails.
- Suggested-action cards → Workflow Builder, Agent Console, Run History, Reports.
- Recent runs list ("View all" → `/enterprise/runs`).

**`/enterprise/workflow` — Workflow Builder (`app/enterprise/workflow/page.tsx`)**
Design a **Custom Pipeline**: pick which agents run and in what order, set target,
then launch (`runWorkflow` → `/api/workflow/run`). Upload entry point if no dataset
loaded yet.

**`/enterprise/agents` — Agent Console (`app/enterprise/agents/page.tsx`)**
Run agents **one at a time** and inspect each output live (`initRun` →
`runSingleAgent` → `getAgentOutput`). Shares one `PipelineState` across calls so
each agent builds on the previous.

**`/enterprise/runs` — Run history (`app/enterprise/runs/page.tsx`)**
Table/list of every run (`listExperiments` → `/api/runs`): status, dataset, mode,
champion, metric, timing. Links into each run's report.

**`/enterprise/reports` — Reports (`app/enterprise/reports/page.tsx`)**
Grid of completed-run cards. Each card headlines the **dataset name**, with
**date · time**, a Free/Pro badge, target, champion + metric, and a small run_id
reference. Buttons: **View report** (→ `/report` or `/free/results`) and a PDF
**download** icon. (Recently redesigned for clarity.)

**`/enterprise/visualizations` — Visualizations (`app/enterprise/visualizations/page.tsx`)**
Browse/generate charts for a run (`getVisualizations` / `generateVisualization`).

**`/enterprise/settings` — Settings (`app/enterprise/settings/page.tsx`)**
Account / workspace preferences (theme, mode, etc.).

### Shared report

**`/report/[runId]` — Full report (`app/report/[runId]/page.tsx`)**
The richest report view. Header: run badge + **Markdown / Notebook / Reproduce /
Download PDF** buttons. **Champion hero card** (model + metric + KPIs). Tabs:
- **Overview** — executive summary, dataset + preprocessing cards, feature-
  engineering timeline, selected-feature chips.
- **Leaderboard** — bar chart + ranked model table.
- **Visualizations** — chart grid with on-demand "Generate Chart" buttons.
- **Explainability (SHAP)** — top features bar chart + table.
- **Quality Audit** — leakage / overfitting / drift / imbalance findings (incl.
  the prominent **leakage-removed** callout).
- **Full Narrative** — the markdown report via `ReportMarkdown`.

**`/pipeline/[runId]`** — generic (mode-agnostic) live pipeline view, sibling of
the free one.

---

## 4. Components

- **`layout/Navbar`** — top bar for public/free pages (logo → `/`, auth link).
- **`layout/Sidebar`** — Pro nav (the 7 links above).
- **`layout/AppShell`** — wraps Pro pages (sidebar + content + user footer).
- **`layout/ModeSwitch`** — Free ⇄ Pro toggle.
- **`ui/AgentNetwork`** — animated 8-agent network/graph (landing + hero).
- **`ui/AnimatedCounter`** — count-up number animation for stats.
- **`ReportMarkdown`** — premium GFM renderer (react-markdown + remark-gfm): themed
  tables (zebra, numeric right-align), accent headings, code, callouts. Used by
  both report pages.

## 5. State (`src/store`, Zustand)

- **`useAuthStore`** — `user`, `token`, `isAuthenticated`; `setAuth` (persists to
  localStorage + `axiom-auth` cookie), logout.
- **`useAppStore`** — `mode`, `hasSelectedMode`, `dataset` (last upload preview),
  `activeRunId`, plus an `experiments` cache.
- **`useAgentStore`** — per-agent status/output for the Agent Console, helpers like
  `getRunningAgent`, `isAllCompleted`.

## 6. `src/lib`

- **`api.ts`** — all backend calls + TS types. Auth (`authLogin/Signup/logout`,
  `getWorkspace`, `setWorkspaceMode`), `uploadDataset` (direct-to-backend, size
  guarded), `startPipeline`, `initRun`, `runWorkflow`, `runSingleAgent`,
  `getStatus`, `getResults`, `getReport`, `getShapData`, `getVisualizations`,
  `generateVisualization`, `listExperiments`, `exportRunExcel`,
  `downloadReportPdf`, `downloadReportNotebook(kind)`.
- **`types.ts`** — `Mode`, `AgentId`, **`AGENT_META`** (label + description per
  agent), `AGENT_ORDER`, `DatasetPreview`, `Experiment`, output types.
- **`cookies.ts`** — `axiom-auth` / workspace cookie helpers.
- **`animations.ts`** — shared framer-motion variants (`fadeUp`, `stagger`).
- **`proxy.ts` / `next.config.ts`** — `/api/*` rewrite + dev-proxy body cap.

## 7. The 8 agents (shown across the UI)

`data_collection → preprocessing → feature_engineering → data_splitting →
model_training → error_detection → improvement → finalization` — labels +
descriptions live in `AGENT_META`. The IDs MUST match the backend
`core/agent_runner.py AGENT_REGISTRY` keys.

## 8. End-to-end data flow

```
/auth (login)  →  /welcome (pick mode)
  Free:  /free (upload + target + launch)  →  /free/pipeline/[runId] (poll status)
         →  /free/results/[runId]  (+ Notebook / PDF)
  Pro:   /enterprise (upload)  →  /enterprise/workflow OR /enterprise/agents
         →  /report/[runId]  (tabs + Markdown/Notebook/Reproduce/PDF)
  Anytime: /enterprise/runs · /reports · /visualizations · /settings
```
Upload → `uploadDataset` (direct to backend) returns columns/dtypes/preview/charts.
Run → returns `run_id`; the pipeline executes in a **child process** server-side;
the UI **polls `/api/status`** (1.5 s) for stage/log/agent-output updates; results
come from `/api/results`, the narrative from `/api/report`, charts/SHAP from their
endpoints, and downloads (PDF / .ipynb report / .ipynb reproduction) from
`/api/report/{id}/...`.

---

## 9. What we want from the frontend (vision + principles)

- **Two clear tiers**: Free = effortless one-click ML; Pro = full transparency and
  control. The mode choice should persist and never be confusing.
- **Trust through transparency**: every stage should visibly show *what it did*
  (we now surface real per-agent facts), and the report must make honest scores
  understandable — e.g. the **leakage-removed callout** explains why an honest 0.91
  beats a leaked 0.99.
- **Premium, calm aesthetic**: dark glass UI, subtle motion, clear hierarchy; the
  dataset/date/time should always identify a run at a glance.
- **Multiple export formats**: PDF (shareable), report `.ipynb`, and a true
  **reproduction `.ipynb`** that re-runs the pipeline.
- **Responsive + resilient**: friendly errors (e.g. upload too large / backend
  down), no silent failures, near-real-time progress.

## 10. Known gaps / wishlist (frontend)

- **Live logs** are polled (1.5 s), not streamed. SSE was evaluated and deferred
  (browser `EventSource` can't send the Bearer header). Revisit if true real-time
  is needed.
- **Settings** page is thin — could expose the env knobs (sampling caps, calibration
  toggle, artifact retention) as user-facing controls.
- **Visualizations** page could match the report's chart-studio polish.
- **Agent Console** could show the per-agent "facts" the same way the live pipeline
  view now does.
- Surface the **inference manifest** (expected features, threshold) on the report
  for users wiring up their own scoring.
- Mobile layout is secondary — primarily designed for desktop.

> Keep this file updated when routes, buttons, or flows change — it's the frontend
> source of truth for a fresh session.
