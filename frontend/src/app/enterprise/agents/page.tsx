"use client";

import { useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Cpu, Play, Upload, ChevronDown, Database, Rows3,
  Columns, CheckCircle2, XCircle, Clock, MemoryStick,
  FileSpreadsheet, Loader2, ChevronRight, AlertTriangle,
  BarChart3, ScrollText,
} from "lucide-react";
import {
  uploadDataset, initRun, runSingleAgent, getAgentOutput,
  exportRunExcel, UploadResponse, AgentOutputResponse,
} from "@/lib/api";
import { AGENT_ORDER, AGENT_META } from "@/lib/types";
import type { AgentId } from "@/lib/types";
import { fadeUp, stagger } from "@/lib/animations";

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Poll until agent status is no longer "running", up to 5 minutes. */
async function pollUntilDone(
  runId: string,
  agentId: string,
): Promise<AgentOutputResponse> {
  for (let i = 0; i < 150; i++) {
    await new Promise((r) => setTimeout(r, 2000));
    try {
      const out = await getAgentOutput(runId, agentId);
      if (out.status !== "running") return out;
    } catch {
      // 404 - background thread hasn't written the output yet, keep polling
    }
  }
  throw new Error("Agent timed out after 5 minutes");
}

function fmt(v: unknown): string {
  if (v === null || v === undefined) return "-";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(4);
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

// ── Sub-components ────────────────────────────────────────────────────────────

function KVTable({ data, title }: { data: Record<string, unknown>; title: string }) {
  const entries = Object.entries(data);
  if (!entries.length) return null;
  return (
    <div className="mt-4">
      <div className="text-[10px] font-semibold uppercase tracking-[1.5px] text-text-muted mb-2">
        {title}
      </div>
      <div className="rounded-xl border border-glass-border overflow-hidden">
        {entries.map(([k, v], i) => (
          <div
            key={k}
            className={`flex justify-between items-start gap-4 px-4 py-2.5 text-[12px] ${
              i !== entries.length - 1 ? "border-b border-glass-border/50" : ""
            } ${i % 2 === 0 ? "" : "bg-white/[0.01]"}`}
          >
            <span className="text-text-muted shrink-0">{k}</span>
            <span className="text-text-primary font-mono text-right break-all">{fmt(v)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function LogsPanel({ logs }: { logs: { time: string; msg: string }[] }) {
  if (!logs.length) return null;
  return (
    <div className="mt-4">
      <div className="text-[10px] font-semibold uppercase tracking-[1.5px] text-text-muted mb-2">
        Logs
      </div>
      <div className="rounded-xl border border-glass-border bg-void/40 p-3 max-h-48 overflow-y-auto space-y-1">
        {logs.map((l, i) => (
          <div key={i} className="flex gap-2 text-[11px] font-mono">
            <span className="text-text-ghost shrink-0">{l.time?.slice(11, 19) ?? ""}</span>
            <span className="text-text-secondary">{l.msg}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function AgentConsolePage() {
  const [uploadData, setUploadData]   = useState<UploadResponse | null>(null);
  const [target, setTarget]           = useState("");
  const [uploading, setUploading]     = useState(false);
  const [runId, setRunId]             = useState<string | null>(null);
  const [initializing, setInit]       = useState(false);

  const [agentOutputs, setAgentOutputs] = useState<
    Partial<Record<AgentId, AgentOutputResponse>>
  >({});
  const [runningAgent, setRunningAgent] = useState<AgentId | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<AgentId | null>(null);
  const [exporting, setExporting]     = useState(false);
  const [error, setError]             = useState<string | null>(null);

  // ── File upload ────────────────────────────────────────────────────────────

  const handleFile = useCallback(async (file: File) => {
    setUploading(true);
    setError(null);
    setRunId(null);
    setAgentOutputs({});
    setSelectedAgent(null);
    try {
      const data = await uploadDataset(file);
      setUploadData(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }, []);

  const openFilePicker = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".csv,.tsv,.txt";
    input.onchange = (e) => {
      const f = (e.target as HTMLInputElement).files?.[0];
      if (f) handleFile(f);
    };
    input.click();
  };

  // ── Initialize run ─────────────────────────────────────────────────────────

  const handleInit = async () => {
    if (!uploadData) return;
    setInit(true);
    setError(null);
    try {
      const res = await initRun(uploadData.path, target || undefined);
      setRunId(res.run_id);
      setAgentOutputs({});
      setSelectedAgent(null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Initialization failed");
    } finally {
      setInit(false);
    }
  };

  // ── Run a single agent ─────────────────────────────────────────────────────

  const runAgent = useCallback(
    async (agentId: AgentId) => {
      if (!runId || runningAgent) return;
      setRunningAgent(agentId);
      setError(null);
      setSelectedAgent(agentId);

      // Optimistically mark as running
      setAgentOutputs((prev) => ({
        ...prev,
        [agentId]: {
          agent_id: agentId,
          status: "running",
          summary: {}, metrics: {}, logs: [], artifacts: {}, visualizations: [],
        } as AgentOutputResponse,
      }));

      try {
        await runSingleAgent(agentId, runId);
        const output = await pollUntilDone(runId, agentId);
        setAgentOutputs((prev) => ({ ...prev, [agentId]: output }));
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : "Agent failed";
        setError(msg);
        setAgentOutputs((prev) => ({
          ...prev,
          [agentId]: {
            ...(prev[agentId] ?? {}),
            status: "failed",
            error: msg,
          } as AgentOutputResponse,
        }));
      } finally {
        setRunningAgent(null);
      }
    },
    [runId, runningAgent],
  );

  // ── Export ─────────────────────────────────────────────────────────────────

  const handleExport = useCallback(async () => {
    if (!runId) return;
    setExporting(true);
    setError(null);
    try {
      await exportRunExcel(runId);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Export failed");
    } finally {
      setExporting(false);
    }
  }, [runId]);

  // ── Derived ────────────────────────────────────────────────────────────────

  const completedCount = Object.values(agentOutputs).filter(
    (o) => o?.status === "completed",
  ).length;

  const selectedOutput = selectedAgent ? agentOutputs[selectedAgent] : null;

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="p-8 max-w-[1400px] mx-auto">
      {/* Header */}
      <motion.div variants={stagger} initial="hidden" animate="visible" className="mb-8">
        <motion.div variants={fadeUp} className="badge badge-accent mb-3">
          <Cpu size={10} /> Agent Console
        </motion.div>
        <motion.h1
          variants={fadeUp}
          className="font-display text-3xl md:text-4xl font-bold text-text-primary tracking-tight"
        >
          Independent Agents
        </motion.h1>
        <motion.p variants={fadeUp} className="text-text-secondary mt-2 font-light">
          Run any agent in isolation. Each agent builds on the state left by the previous one.
        </motion.p>
      </motion.div>

      {/* ── Step 1: Upload ──────────────────────────────────────────────────── */}
      <div className="glass p-6 mb-5">
        <div className="text-[10px] font-semibold uppercase tracking-[2px] text-text-muted mb-4">
          Step 1 - Upload Dataset
        </div>

        {!uploadData ? (
          <div
            onClick={openFilePicker}
            className="p-8 border border-dashed border-glass-border rounded-2xl text-center cursor-pointer hover:border-glass-hover hover:bg-glass-hover transition-all"
          >
            {uploading ? (
              <Loader2 size={24} className="text-accent animate-spin mx-auto" />
            ) : (
              <>
                <Upload size={24} className="text-text-muted mx-auto mb-2" strokeWidth={1.5} />
                <p className="text-[13px] text-text-muted">
                  Click to upload a CSV / TSV file
                </p>
              </>
            )}
          </div>
        ) : (
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-xl bg-success/[0.08] border border-success/20 flex items-center justify-center shrink-0">
              <Database size={18} className="text-success" strokeWidth={1.5} />
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-[13px] font-semibold text-text-primary truncate">
                {uploadData.filename}
              </div>
              <div className="flex items-center gap-3 mt-0.5">
                <span className="flex items-center gap-1 text-[11px] text-text-muted">
                  <Rows3 size={10} /> {uploadData.n_rows.toLocaleString()} rows
                </span>
                <span className="flex items-center gap-1 text-[11px] text-text-muted">
                  <Columns size={10} /> {uploadData.columns.length} cols
                </span>
              </div>
            </div>
            <button
              onClick={openFilePicker}
              className="text-[11px] text-accent font-medium hover:underline shrink-0"
            >
              Replace
            </button>
          </div>
        )}
      </div>

      {/* ── Step 2: Configure & Init ────────────────────────────────────────── */}
      <AnimatePresence>
        {uploadData && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3, ease: [0.25, 0.4, 0, 1] }}
            className="overflow-hidden"
          >
            <div className="glass p-6 mb-5">
              <div className="text-[10px] font-semibold uppercase tracking-[2px] text-text-muted mb-4">
                Step 2 - Configure &amp; Initialize
              </div>
              <div className="flex items-end gap-4">
                <div className="flex-1">
                  <label className="text-[11px] text-text-muted block mb-1.5">
                    Target Column
                  </label>
                  <div className="relative">
                    <select
                      value={target}
                      onChange={(e) => setTarget(e.target.value)}
                      className="w-full appearance-none bg-void/50 border border-glass-border rounded-xl px-3 py-2.5 text-[13px] text-text-primary focus:outline-none focus:border-accent/40 transition-colors pr-8"
                    >
                      <option value="">None - unsupervised</option>
                      {uploadData.columns.map((c) => (
                        <option key={c} value={c}>{c}</option>
                      ))}
                    </select>
                    <ChevronDown
                      size={13}
                      className="absolute right-2.5 top-1/2 -translate-y-1/2 text-text-muted pointer-events-none"
                    />
                  </div>
                </div>
                <button
                  onClick={handleInit}
                  disabled={initializing || !!runId}
                  className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-accent text-black font-semibold text-[13px] hover:bg-accent-bright transition-colors disabled:opacity-40 disabled:cursor-not-allowed shrink-0"
                >
                  {initializing ? (
                    <Loader2 size={14} className="animate-spin" />
                  ) : runId ? (
                    <CheckCircle2 size={14} />
                  ) : (
                    <Cpu size={14} />
                  )}
                  {runId ? "Initialized" : "Initialize Run"}
                </button>
              </div>

              {runId && (
                <div className="mt-3 flex items-center gap-2 text-[11px] text-text-muted">
                  <CheckCircle2 size={12} className="text-success" />
                  Run ID:
                  <span className="font-mono text-accent">{runId}</span>
                  <button
                    onClick={() => { setRunId(null); setAgentOutputs({}); setSelectedAgent(null); }}
                    className="ml-auto text-[11px] text-text-ghost hover:text-text-muted transition-colors"
                  >
                    Reset
                  </button>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {error && (
        <div className="flex items-start gap-3 p-4 rounded-xl border border-destructive/20 bg-destructive/[0.04] mb-5">
          <XCircle size={15} className="text-destructive shrink-0 mt-0.5" />
          <span className="text-destructive text-[12px]">{error}</span>
        </div>
      )}

      {/* ── Step 3: Agent Grid ──────────────────────────────────────────────── */}
      {uploadData && (
        <>
          <div className="text-[10px] font-semibold uppercase tracking-[2px] text-text-muted mb-4">
            Step 3 - Run Agents
            {completedCount > 0 && (
              <span className="ml-2 text-success">
                ({completedCount}/8 completed)
              </span>
            )}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-5">
            {AGENT_ORDER.map((agentId, i) => {
              const meta    = AGENT_META[agentId];
              const output  = agentOutputs[agentId];
              const status  = output?.status;
              const isRun   = runningAgent === agentId;
              const isSel   = selectedAgent === agentId;

              const borderCls =
                status === "completed" ? "border-success/25" :
                status === "failed"    ? "border-destructive/25" :
                status === "running"   ? "border-accent/25" :
                isSel                  ? "border-glass-hover" : "";

              return (
                <motion.div
                  key={agentId}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.04 }}
                  className={`glass-xs overflow-hidden transition-all duration-300 ${borderCls}`}
                >
                  <div className="p-4">
                    {/* Icon + name */}
                    <div className="flex items-center gap-2.5 mb-3">
                      <div
                        className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0"
                        style={{ background: `${meta.color}18` }}
                      >
                        <Cpu size={14} style={{ color: meta.color }} strokeWidth={1.5} />
                      </div>
                      <div className="min-w-0">
                        <div className="text-[12px] font-semibold leading-tight truncate">
                          {meta.shortLabel}
                        </div>
                        <div className="text-[10px] text-text-ghost truncate">{meta.label}</div>
                      </div>
                    </div>

                    {/* Status */}
                    {status === "running" && (
                      <div className="flex items-center gap-1.5 mb-2">
                        <Loader2 size={11} className="text-accent animate-spin" />
                        <span className="text-[10px] text-accent">Running…</span>
                      </div>
                    )}
                    {status === "completed" && output && (
                      <div className="space-y-1 mb-2">
                        <div className="flex items-center gap-1.5">
                          <CheckCircle2 size={11} className="text-success" />
                          <span className="text-[10px] text-success">Completed</span>
                          {output.duration_seconds != null && (
                            <span className="text-[10px] text-text-ghost ml-auto font-mono">
                              {output.duration_seconds.toFixed(1)}s
                            </span>
                          )}
                        </div>
                        {Object.entries(output.metrics ?? {}).slice(0, 2).map(([k, v]) => (
                          <div key={k} className="flex justify-between text-[10px]">
                            <span className="text-text-ghost truncate">{k}</span>
                            <span className="text-text-secondary font-mono">{fmt(v)}</span>
                          </div>
                        ))}
                      </div>
                    )}
                    {status === "failed" && output && (
                      <div className="flex items-center gap-1.5 mb-2">
                        <AlertTriangle size={11} className="text-destructive" />
                        <span className="text-[10px] text-destructive truncate">
                          {output.error?.slice(0, 40) ?? "Failed"}
                        </span>
                      </div>
                    )}

                    {/* Buttons row */}
                    <div className="flex gap-1.5 mt-1">
                      <button
                        onClick={() => runAgent(agentId)}
                        disabled={!runId || !!runningAgent}
                        className="flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-lg text-[11px] font-semibold bg-accent/[0.08] border border-accent/20 text-accent hover:bg-accent/[0.15] transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                      >
                        {isRun ? (
                          <Loader2 size={10} className="animate-spin" />
                        ) : (
                          <Play size={10} />
                        )}
                        {isRun ? "Running" : "Run"}
                      </button>

                      {output && status !== "running" && (
                        <button
                          onClick={() =>
                            setSelectedAgent(isSel ? null : agentId)
                          }
                          className={`flex items-center justify-center w-8 h-7 rounded-lg text-[11px] border transition-colors ${
                            isSel
                              ? "bg-white/[0.08] border-glass-hover text-text-primary"
                              : "border-glass-border text-text-muted hover:bg-glass-hover"
                          }`}
                        >
                          <ChevronRight
                            size={12}
                            className={`transition-transform ${isSel ? "rotate-90" : ""}`}
                          />
                        </button>
                      )}
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </div>

          {/* ── Output panel ──────────────────────────────────────────────── */}
          <AnimatePresence>
            {selectedAgent && selectedOutput && selectedOutput.status !== "running" && (
              <motion.div
                key={selectedAgent}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.25, ease: [0.25, 0.4, 0, 1] }}
                className="glass p-6 mb-5"
              >
                {/* Panel header */}
                <div className="flex items-center justify-between mb-5">
                  <div className="flex items-center gap-3">
                    <div
                      className="w-9 h-9 rounded-xl flex items-center justify-center shrink-0"
                      style={{ background: `${AGENT_META[selectedAgent].color}18` }}
                    >
                      <Cpu size={16} style={{ color: AGENT_META[selectedAgent].color }} strokeWidth={1.5} />
                    </div>
                    <div>
                      <div className="text-[15px] font-semibold text-text-primary">
                        {AGENT_META[selectedAgent].label}
                      </div>
                      <div className="flex items-center gap-3 mt-0.5 text-[11px] text-text-muted">
                        {selectedOutput.duration_seconds != null && (
                          <span className="flex items-center gap-1">
                            <Clock size={10} /> {selectedOutput.duration_seconds.toFixed(2)}s
                          </span>
                        )}
                        {selectedOutput.memory_mb != null && selectedOutput.memory_mb > 0 && (
                          <span className="flex items-center gap-1">
                            <MemoryStick size={10} /> {selectedOutput.memory_mb.toFixed(1)} MB
                          </span>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <span
                      className={`badge ${
                        selectedOutput.status === "completed" ? "badge-success" :
                        selectedOutput.status === "failed"    ? "badge-error" : "badge-accent"
                      }`}
                    >
                      {selectedOutput.status === "completed" && <CheckCircle2 size={9} />}
                      {selectedOutput.status === "failed" && <AlertTriangle size={9} />}
                      {selectedOutput.status}
                    </span>
                    <button
                      onClick={() => setSelectedAgent(null)}
                      className="text-[11px] text-text-ghost hover:text-text-muted transition-colors ml-1"
                    >
                      ✕
                    </button>
                  </div>
                </div>

                {/* Error message */}
                {selectedOutput.status === "failed" && selectedOutput.error && (
                  <div className="flex items-start gap-2 p-3 rounded-xl border border-destructive/20 bg-destructive/[0.04] mb-4 text-[12px] text-destructive">
                    <AlertTriangle size={13} className="shrink-0 mt-0.5" />
                    {selectedOutput.error}
                  </div>
                )}

                {/* Two-column: Summary + Metrics */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  {Object.keys(selectedOutput.summary ?? {}).length > 0 && (
                    <KVTable
                      title="Summary"
                      data={selectedOutput.summary as Record<string, unknown>}
                    />
                  )}
                  {Object.keys(selectedOutput.metrics ?? {}).length > 0 && (
                    <div>
                      <KVTable
                        title="Metrics"
                        data={Object.fromEntries(
                          Object.entries(selectedOutput.metrics ?? {}).map(([k, v]) => [
                            k, v,
                          ])
                        )}
                      />
                      {/* Metric mini-bar chart */}
                      <div className="mt-3 space-y-2">
                        {Object.entries(selectedOutput.metrics ?? {})
                          .filter(([, v]) => typeof v === "number" && v <= 1 && v >= 0)
                          .slice(0, 4)
                          .map(([k, v]) => (
                            <div key={k}>
                              <div className="flex justify-between text-[10px] text-text-muted mb-1">
                                <span>{k}</span>
                                <span className="font-mono text-text-primary">{(v as number).toFixed(4)}</span>
                              </div>
                              <div className="h-1.5 bg-glass-border rounded-full overflow-hidden">
                                <div
                                  className="h-full rounded-full bg-accent transition-all duration-700"
                                  style={{ width: `${(v as number) * 100}%` }}
                                />
                              </div>
                            </div>
                          ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Logs */}
                <LogsPanel logs={selectedOutput.logs ?? []} />

                {/* Artifacts */}
                {Object.keys(selectedOutput.artifacts ?? {}).length > 0 && (
                  <div className="mt-4">
                    <div className="text-[10px] font-semibold uppercase tracking-[1.5px] text-text-muted mb-2">
                      Artifacts
                    </div>
                    <div className="space-y-1">
                      {Object.entries(selectedOutput.artifacts ?? {}).map(([k, v]) => (
                        <div key={k} className="flex items-center gap-2 text-[11px]">
                          <span className="text-text-muted shrink-0">{k}:</span>
                          <span className="font-mono text-accent break-all">{v}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Visualizations */}
                {selectedOutput.visualizations && selectedOutput.visualizations.length > 0 && (
                  <div className="mt-6 border-t border-glass-border/40 pt-4">
                    <div className="text-[10px] font-semibold uppercase tracking-[1.5px] text-text-muted mb-3 flex items-center gap-1.5">
                      <BarChart3 size={11} className="text-accent" />
                      Visualizations
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {selectedOutput.visualizations.map((viz) => (
                        <div key={viz.type} className="viz-card overflow-hidden flex flex-col justify-between border border-glass-border rounded-xl bg-white/[0.01]">
                          <img
                            src={`data:image/png;base64,${viz.base64_png}`}
                            alt={viz.name}
                            className="w-full object-contain"
                          />
                          <div className="p-3 border-t border-glass-border bg-white/[0.01]">
                            <div className="text-[12px] font-semibold text-text-primary">{viz.name}</div>
                            <div className="text-[10px] text-text-muted mt-0.5">{viz.description}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>

          {/* ── Export bar ────────────────────────────────────────────────── */}
          {runId && completedCount > 0 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex items-center justify-between glass-xs p-4"
            >
              <div className="flex items-center gap-2 text-[12px] text-text-secondary">
                <BarChart3 size={14} className="text-accent" strokeWidth={1.5} />
                <span>
                  <span className="text-text-primary font-semibold">{completedCount}</span> agent
                  {completedCount !== 1 ? "s" : ""} completed - export results to Excel
                </span>
              </div>
              <button
                onClick={handleExport}
                disabled={exporting}
                className="flex items-center gap-2 px-4 py-2 rounded-xl text-[12px] font-semibold bg-success/[0.08] border border-success/20 text-success hover:bg-success/[0.15] transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {exporting ? (
                  <Loader2 size={13} className="animate-spin" />
                ) : (
                  <FileSpreadsheet size={13} />
                )}
                {exporting ? "Generating…" : "Export All to Excel"}
              </button>
            </motion.div>
          )}
        </>
      )}
    </div>
  );
}
