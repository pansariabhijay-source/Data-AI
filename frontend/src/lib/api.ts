const API_BASE = "/api";

// Hard cap on client-side upload size (mirror of backend MAX_UPLOAD_MB).
const MAX_UPLOAD_MB = 1024;

/**
 * Origin of the FastAPI backend for requests that must bypass the Next.js dev
 * proxy. The proxy buffers the entire request body in memory and resets the
 * connection once it exceeds its size cap (ECONNRESET, surfaced to the user as
 * a misleading "backend unavailable"). Large dataset uploads therefore go
 * straight to the backend, which streams them to disk and has CORS configured
 * for this origin. In production (single gateway) this is "" => same-origin.
 */
function backendOrigin(): string {
  if (process.env.NEXT_PUBLIC_BACKEND_URL) return process.env.NEXT_PUBLIC_BACKEND_URL;
  if (typeof window !== "undefined") {
    const h = window.location.hostname;
    if (h === "localhost" || h === "127.0.0.1") return `http://${h}:8000`;
  }
  return "";
}

// ── Types ──────────────────────────────────────────────────────────────────

export interface UploadResponse {
  status: string;
  filename: string;
  path: string;
  columns: string[];
  dtypes: Record<string, string>;
  n_rows: number;
  preview: Record<string, unknown>[];
  visualizations?: VizResult[];
}

export interface RunResponse {
  status: string;
  run_id: string;
}

export interface StatusResponse {
  run_id: string;
  status: "starting" | "running" | "completed" | "failed";
  current_stage: string | null;
  completed_stages: string[];
  started_at: string;
  logs: { time: string; msg: string }[];
  error: string | null;
  agent_outputs?: Record<string, AgentOutputResponse>;
}

export interface AgentOutputResponse {
  agent_id: string;
  status: string;
  started_at?: string;
  completed_at?: string;
  duration_seconds?: number;
  memory_mb?: number;
  summary: Record<string, unknown>;
  metrics: Record<string, number>;
  logs: { time: string; msg: string }[];
  artifacts: Record<string, string>;
  visualizations: VizResult[];
  error?: string;
}

export interface VizResult {
  name: string;
  type: string;
  description: string;
  base64_png: string;
  category: "basic" | "advanced";
}

export interface ModelResult {
  name: string;
  status: string;
  metrics: Record<string, number>;
  time_s: number;
  is_best: boolean;
}

export interface ResultsResponse {
  run_id: string;
  status: string;
  problem_type: string;
  target_column: string;
  best_model: string;
  best_metric_name: string;
  best_metric_value: number;
  retry_count: number;
  dataset: { rows: number; columns: number; quality_score: number };
  preprocessing: { rows_before: number; rows_after: number; quality_score: number; duplicates_removed: number };
  features: { before: number; after: number; selected: string[] };
  models: ModelResult[];
  errors: { severity: string; type: string; cause: string; fix: string }[];
  artifacts: Record<string, string>;
  reports: Record<string, string>;
  experiment_history: { iteration: number; best_model: string; best_value: number }[];
  agent_outputs?: Record<string, AgentOutputResponse>;
}

// ── API Functions ──────────────────────────────────────────────────────────

import { useAuthStore } from "@/store/useAuthStore";

async function authFetch(url: string, options: RequestInit = {}) {
  const token = useAuthStore.getState().token;
  const headers = new Headers(options.headers);
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  return fetch(url, { ...options, headers });
}

// ── Auth Endpoints ─────────────────────────────────────────────────────────


async function handleResponse(res: Response, defaultError: string) {
  if (!res.ok) {
    const isJson = res.headers.get('content-type')?.includes('application/json');
    if (isJson) {
      const data = await res.json();
      throw new Error(data.detail || data.message || defaultError);
    } else {
      const text = await res.text();
      if (res.status === 502 || res.status === 504 || res.status === 500) {
        throw new Error('Backend server is unavailable or encountered an error. Please ensure the Python backend is running.');
      }
      throw new Error(`Server error: ${res.status} ${res.statusText}`);
    }
  }
  return res.json();
}

export async function authSignup(email: string, username: string, password: string) {
  const res = await fetch(`${API_BASE}/auth/signup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, username, password }),
  });
  return handleResponse(res, "Signup failed");
}

export async function authLogin(email: string, password: string) {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  return handleResponse(res, "Login failed");
}

export async function authLogout(token: string) {
  await fetch(`${API_BASE}/auth/logout`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
  });
}

export async function getPromptHistory() {
  const res = await authFetch(`${API_BASE}/auth/history`);
  const data = await handleResponse(res, "Failed to fetch history");
  return data.history;
}

export interface WorkspacePrefs {
  selected_mode: "free" | "enterprise" | null;
  theme: string;
  onboarding_completed: boolean;
  updated_at: string | null;
}

/** Read the user's durable workspace preferences so they resume exactly where they left off. */
export async function getWorkspace(): Promise<WorkspacePrefs> {
  const res = await authFetch(`${API_BASE}/workspace`);
  const data = await handleResponse(res, "Failed to load workspace");
  return data.workspace as WorkspacePrefs;
}

/** Persist the user's selected workspace mode. Best-effort; UI does not block on it. */
export async function setWorkspaceMode(mode: "free" | "enterprise"): Promise<void> {
  await authFetch(`${API_BASE}/workspace/mode`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  }).catch(() => {});
}

// ── Platform Endpoints ─────────────────────────────────────────────────────

export async function uploadDataset(file: File): Promise<UploadResponse> {
  // Fail fast with a clear message instead of streaming a huge body only to be
  // rejected (or, worse, truncated by a proxy → confusing connection error).
  if (file.size === 0) throw new Error("That file is empty.");
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
    throw new Error(
      `File is ${(file.size / 1024 / 1024).toFixed(0)} MB — the limit is ${MAX_UPLOAD_MB} MB.`,
    );
  }
  const form = new FormData();
  form.append("file", file);
  // Bypass the Next.js dev proxy for the (potentially large) upload body.
  const url = `${backendOrigin()}${API_BASE}/upload`;
  let res: Response;
  try {
    res = await authFetch(url, { method: "POST", body: form });
  } catch {
    throw new Error(
      "Could not reach the backend. Make sure it is running (it starts with the frontend via `npm run dev`), then try again.",
    );
  }
  return handleResponse(res, "Upload failed");
}

export async function startPipeline(dataPath: string, targetColumn?: string): Promise<RunResponse> {
  const form = new FormData();
  form.append("data_path", dataPath);
  if (targetColumn) form.append("target_column", targetColumn);
  const res = await authFetch(`${API_BASE}/run`, { method: "POST", body: form });
  return handleResponse(res, "Failed to start");
}

/** Initialize a run context without starting any pipeline. Used by the Enterprise Agent Console. */
export async function initRun(dataPath: string, targetColumn?: string): Promise<RunResponse> {
  const form = new FormData();
  form.append("data_path", dataPath);
  if (targetColumn) form.append("target_column", targetColumn);
  const res = await authFetch(`${API_BASE}/init-run`, { method: "POST", body: form });
  return handleResponse(res, "Failed to initialize run");
}

/** Download all agent outputs for a run as an Excel file. */
export async function exportRunExcel(runId: string): Promise<void> {
  const res = await authFetch(`${API_BASE}/export/${runId}/excel`);
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error((data as { detail?: string }).detail || "Export failed");
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `axiom-${runId.slice(0, 8)}.xlsx`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export async function runSingleAgent(
  agentName: string,
  runId: string,
  config?: Record<string, unknown>
): Promise<{ status: string; agent_output: AgentOutputResponse }> {
  const res = await authFetch(`${API_BASE}/agent/${agentName}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_id: runId, config }),
  });
  return handleResponse(res, "Agent run failed");
}

export async function runWorkflow(
  agentList: string[],
  dataPath: string,
  targetColumn?: string,
  config?: Record<string, unknown>
): Promise<RunResponse> {
  const res = await authFetch(`${API_BASE}/workflow/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      agents: agentList,
      data_path: dataPath,
      target_column: targetColumn,
      config,
    }),
  });
  return handleResponse(res, "Workflow failed");
}

export async function getStatus(runId: string): Promise<StatusResponse> {
  const res = await authFetch(`${API_BASE}/status/${runId}`);
  return handleResponse(res, "Status fetch failed");
}

export async function getResults(runId: string): Promise<ResultsResponse> {
  const res = await authFetch(`${API_BASE}/results/${runId}`);
  return handleResponse(res, "Results fetch failed");
}

export async function getReport(runId: string): Promise<string> {
  const res = await authFetch(`${API_BASE}/report/${runId}`);
  const data = await handleResponse(res, "Report not found");
  return data.report;
}

/** Download the styled PDF report for a run. */
export async function downloadReportPdf(runId: string): Promise<void> {
  const res = await authFetch(`${API_BASE}/report/${runId}/pdf`);
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error((data as { detail?: string }).detail || "PDF download failed");
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `axiom-report-${runId.slice(0, 8)}.pdf`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Download the run as a Jupyter notebook (.ipynb).
 * - "report": narrative + charts + code to load the champion (shareable).
 * - "reproduce": standalone notebook that re-runs Axiom's actual pipeline to
 *   reproduce the result from scratch.
 */
export async function downloadReportNotebook(
  runId: string,
  kind: "report" | "reproduce" = "report",
): Promise<void> {
  const res = await authFetch(`${API_BASE}/report/${runId}/notebook?kind=${kind}`);
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error((data as { detail?: string }).detail || "Notebook download failed");
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `axiom-${kind === "reproduce" ? "reproduce" : "report"}-${runId.slice(0, 8)}.ipynb`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Download the trained champion model as a zip bundle: the .joblib model file,
 * the inference manifest (feature order + decision threshold), and a
 * predict_example.py starting point for scoring new data offline.
 */
export async function downloadModel(runId: string): Promise<void> {
  const res = await authFetch(`${API_BASE}/report/${runId}/model`);
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error((data as { detail?: string }).detail || "Model download failed");
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `axiom-model-${runId.slice(0, 8)}.zip`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export async function getShapData(runId: string): Promise<Record<string, number>> {
  const res = await authFetch(`${API_BASE}/shap/${runId}`);
  const data = await handleResponse(res, "SHAP data not found");
  return data.shap_importance as Record<string, number>;
}

export async function getAgentOutput(
  runId: string,
  agentName: string
): Promise<AgentOutputResponse> {
  const res = await authFetch(`${API_BASE}/agent/${agentName}/output/${runId}`);
  return handleResponse(res, "Agent output not found");
}

export async function getVisualizations(
  runId: string,
  category?: "basic" | "advanced"
): Promise<VizResult[]> {
  const url = category
    ? `${API_BASE}/visualizations/${runId}?category=${category}`
    : `${API_BASE}/visualizations/${runId}`;
  const res = await authFetch(url);
  const data = await handleResponse(res, "Visualizations not found");
  return data.visualizations;
}

export async function generateVisualization(
  runId: string,
  vizType: string,
  params?: Record<string, unknown>
): Promise<VizResult> {
  const res = await authFetch(`${API_BASE}/visualizations/${runId}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ type: vizType, params }),
  });
  return handleResponse(res, "Visualization generation failed");
}

export async function listExperiments(): Promise<{
  runs: {
    run_id: string;
    status: string;
    started_at: string;
    completed_at: string | null;
    duration_seconds: number | null;
    current_stage: string | null;
    mode?: string;
    dataset?: string | null;
    target?: string | null;
    best_model?: string;
    best_metric_value?: number;
    completed_stages: string[];
    agent_count: number;
  }[];
}> {
  const res = await authFetch(`${API_BASE}/runs`);
  return handleResponse(res, "Failed to list experiments");
}
