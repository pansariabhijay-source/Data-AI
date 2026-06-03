"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, Play, ChevronDown, Database, Rows3, Columns, Cpu, CheckCircle2, Circle, ArrowRight, Crown, Loader2, XCircle, FileText, BarChart3 } from "lucide-react";
import { uploadDataset, runWorkflow, UploadResponse, getStatus, getReport, StatusResponse, getVisualizations, VizResult } from "@/lib/api";
import { useAppStore } from "@/store/useAppStore";
import { AGENT_ORDER, AGENT_META } from "@/lib/types";
import type { AgentId } from "@/lib/types";
import { fadeUp, stagger } from "@/lib/animations";

export default function WorkflowBuilderPage() {
  const router = useRouter();
  const { setActiveRunId, dataset, setDataset } = useAppStore();
  const [uploadData, setUploadData] = useState<UploadResponse | null>(null);
  const [target, setTarget] = useState("");
  const [selectedAgents, setSelectedAgents] = useState<Set<AgentId>>(new Set(AGENT_ORDER));
  const [uploading, setUploading] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Execution state
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [report, setReport] = useState<string | null>(null);
  const [vizs, setVizs] = useState<VizResult[]>([]);
  const [elapsed, setElapsed] = useState(0);
  const logRef = useRef<HTMLDivElement>(null);

  const [currentRunId, setCurrentRunId] = useState<string | null>(null);

  useEffect(() => {
    if (!currentRunId || !running) return;
    const poll = setInterval(async () => {
      try {
        const s = await getStatus(currentRunId);
        setStatus(s);
        if (s.status === "completed" || s.status === "failed") {
          clearInterval(poll);
          if (s.status === "completed") {
            const [rep, vList] = await Promise.all([
              getReport(currentRunId).catch(() => "No report available."),
              getVisualizations(currentRunId).catch(() => []),
            ]);
            setReport(rep);
            setVizs(vList);
          }
        }
      } catch { /* ignore */ }
    }, 2000);
    return () => clearInterval(poll);
  }, [currentRunId, running]);

  useEffect(() => {
    const t = setInterval(() => {
      if (running && status?.status !== "completed" && status?.status !== "failed") {
        setElapsed(e => e + 1);
      }
    }, 1000);
    return () => clearInterval(t);
  }, [running, status?.status]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: "smooth" });
  }, [status?.logs]);

  const toggleAgent = (id: AgentId) => {
    const next = new Set(selectedAgents);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setSelectedAgents(next);
  };

  const selectAll = () => setSelectedAgents(new Set(AGENT_ORDER));
  const selectNone = () => setSelectedAgents(new Set());

  const handleFile = useCallback(async (file: File) => {
    setUploading(true);
    setError(null);
    try {
      const data = await uploadDataset(file);
      setUploadData(data);
      setDataset({ filename: data.filename, path: data.path, columns: data.columns, dtypes: data.dtypes, n_rows: data.n_rows, preview: data.preview });
    } catch (e: unknown) { setError(e instanceof Error ? e.message : "Upload failed"); }
    finally { setUploading(false); }
  }, [setDataset]);

  const handleRun = async () => {
    if (!uploadData || selectedAgents.size === 0) return;
    setRunning(true);
    setError(null);
    try {
      const agents = AGENT_ORDER.filter((a) => selectedAgents.has(a));
      const res = await runWorkflow(agents, uploadData.path, target || undefined);
      setActiveRunId(res.run_id);
      setCurrentRunId(res.run_id);
    } catch (e: unknown) { setError(e instanceof Error ? e.message : "Workflow failed"); setRunning(false); }
  };

  const fmtElapsed = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, "0")}`;

  return (
    <div className="p-8 max-w-[1200px] mx-auto relative z-10">
      <motion.div variants={stagger} initial="hidden" animate="visible" className="mb-8">
        <motion.div variants={fadeUp} className="flex items-center gap-2 mb-3">
          <div className="badge badge-pro"><Crown size={10} /> Workflow Builder</div>
          <div className="badge badge-pro-gold text-[9px]">PRO ONLY</div>
        </motion.div>
        <motion.h1 variants={fadeUp} className="text-3xl font-bold tracking-[-0.03em] gradient-text-pro-hero">Custom Pipeline</motion.h1>
        <motion.p variants={fadeUp} className="text-text-secondary mt-2 font-light">Select agents, configure, and execute your custom ML workflow.</motion.p>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Upload + Config */}
        <div className="lg:col-span-1 space-y-4">
          {/* Upload */}
          <div
            onClick={() => { const i = document.createElement("input"); i.type = "file"; i.accept = ".csv,.tsv,.txt"; i.onchange = (e) => { const f = (e.target as HTMLInputElement).files?.[0]; if (f) handleFile(f); }; i.click(); }}
            className="pro-glass p-6 cursor-pointer hover:border-pro/20 transition-all text-center"
          >
            {uploading ? <div className="w-6 h-6 border-2 border-pro/30 border-t-pro rounded-full animate-spin mx-auto" />
            : uploadData ? (
              <div className="flex items-center gap-3 text-left">
                <Database size={20} className="text-pro shrink-0" />
                <div className="min-w-0"><div className="text-[13px] font-semibold truncate">{uploadData.filename}</div>
                  <div className="text-[11px] text-text-muted"><Rows3 size={10} className="inline" /> {uploadData.n_rows.toLocaleString()} × <Columns size={10} className="inline" /> {uploadData.columns.length}</div></div>
              </div>
            ) : <div><Upload size={24} className="text-text-muted mx-auto mb-2" /><p className="text-[12px] text-text-muted">Upload dataset</p></div>}
          </div>

          {/* Target */}
          {uploadData && (
            <div className="pro-glass p-4">
              <label className="text-[10px] font-semibold uppercase tracking-[2px] text-text-muted block mb-2">Target</label>
              <div className="relative">
                <select value={target} onChange={(e) => setTarget(e.target.value)} className="w-full appearance-none bg-void/50 border border-pro-glass-border rounded-lg px-3 py-2 text-[13px] text-text-primary focus:outline-none focus:border-pro/40">
                  <option value="">None (unsupervised)</option>
                  {uploadData.columns.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
                <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted pointer-events-none" />
              </div>
            </div>
          )}

          {/* Summary */}
          <div className="pro-glass-xs p-4">
            <div className="text-[10px] font-semibold uppercase tracking-[2px] text-text-muted mb-2">Pipeline Summary</div>
            <div className="flex items-center justify-between">
              <span className="text-[12px] text-text-secondary">Active Agents</span>
              <span className="text-[14px] font-bold text-pro tabular-nums">{selectedAgents.size} / {AGENT_ORDER.length}</span>
            </div>
            <div className="w-full h-1.5 rounded-full bg-pro-glass mt-2 overflow-hidden">
              <motion.div 
                className="h-full rounded-full bg-gradient-to-r from-pro to-pro-bright"
                animate={{ width: `${(selectedAgents.size / AGENT_ORDER.length) * 100}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>
          </div>

          {/* Run */}
          <button onClick={handleRun} disabled={!uploadData || selectedAgents.size === 0 || running}
            className="btn-pro-primary w-full justify-center py-3 disabled:opacity-40 disabled:cursor-not-allowed">
            {running ? <><div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> Running...</>
            : !uploadData ? <><Database size={14} /> Upload dataset to run</>
            : selectedAgents.size === 0 ? <><Cpu size={14} /> Select agents to run</>
            : <><Play size={14} /> Run Workflow ({selectedAgents.size} agents)</>}
          </button>

          {status && (
            <div className="pro-glass-xs p-3 border-success/20 bg-success/[0.04]">
              <div className="text-[11px] text-success font-semibold mb-1">Status: {status.status}</div>
              <div className="text-[10px] text-text-muted font-mono">Elapsed: {fmtElapsed(elapsed)}</div>
            </div>
          )}

          {error && <div className="pro-glass-xs p-3 border-destructive/20 bg-destructive/[0.04] text-destructive text-[12px]">{error}</div>}
        </div>

        {/* Right: Agent selection or Execution View */}
        <div className="lg:col-span-2">
          {!running && !status ? (
            <>
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-[11px] font-semibold uppercase tracking-[2px] text-pro/60">Agent Pipeline</h2>
                <div className="flex gap-2">
                  <button onClick={selectAll} className="btn-pro text-[10px] py-1 px-2">All</button>
                  <button onClick={selectNone} className="btn-pro text-[10px] py-1 px-2">None</button>
                </div>
              </div>

          <div className="space-y-2">
            {AGENT_ORDER.map((agentId, i) => {
              const meta = AGENT_META[agentId];
              const selected = selectedAgents.has(agentId);
              return (
                <motion.div key={agentId} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.05 }}>
                  <div className="flex items-center gap-2">
                    {/* Connection line */}
                    {i > 0 && (
                      <div className="w-8 flex flex-col items-center">
                        <div className={`w-px h-4 ${selectedAgents.has(AGENT_ORDER[i - 1]) && selected ? "bg-pro/30" : "bg-glass-border"}`} />
                        <ArrowRight size={8} className={`${selectedAgents.has(AGENT_ORDER[i - 1]) && selected ? "text-pro/40" : "text-glass-border"} rotate-90`} />
                      </div>
                    )}
                    {i === 0 && <div className="w-8" />}

                    {/* Agent card with toggle switch */}
                    <button
                      onClick={() => toggleAgent(agentId)}
                      className={`flex-1 flex items-center gap-4 p-4 rounded-xl border transition-all duration-300 text-left ${
                        selected ? "border-pro/20 bg-pro/[0.03] hover:bg-pro/[0.06]" : "border-glass-border bg-glass hover:bg-glass-hover opacity-60"
                      }`}
                    >
                      <div className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 ${selected ? "bg-pro/10" : "bg-glass"}`}>
                        {selected ? <CheckCircle2 size={18} style={{ color: meta.color }} /> : <Circle size={18} className="text-text-ghost" />}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <Cpu size={12} style={{ color: meta.color }} />
                          <span className="text-[13px] font-semibold text-text-primary">{meta.label}</span>
                        </div>
                        <p className="text-[11px] text-text-muted mt-0.5">{meta.description}</p>
                      </div>

                      {/* Toggle switch */}
                      <div className={`relative w-10 h-[22px] rounded-full transition-all duration-300 shrink-0 ${
                        selected
                          ? "bg-gradient-to-r from-pro to-pro-bright shadow-[0_0_8px_rgba(129,140,248,0.2)]"
                          : "bg-glass-active"
                      }`}>
                        <motion.div
                          className="absolute top-[3px] w-4 h-4 rounded-full bg-white shadow-md"
                          animate={{ left: selected ? "calc(100% - 19px)" : "3px" }}
                          transition={{ type: "spring", stiffness: 500, damping: 30 }}
                        />
                      </div>
                    </button>
                  </div>
                </motion.div>
              );
            })}
          </div>
          </>
          ) : (
            <div className="space-y-6">
              {/* Workflow Progress Console */}
              <div className="pro-glass overflow-hidden p-6">
                <div className="flex items-center justify-between mb-4 border-b border-pro-glass-border pb-3">
                  <div>
                    <h3 className="text-[13px] font-semibold text-text-primary">Workflow Progress</h3>
                    <p className="text-[11px] text-text-muted mt-0.5">Executing your custom ML pipeline</p>
                  </div>
                  <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full ${
                    status?.status === "completed" ? "bg-success/10 text-success" :
                    status?.status === "failed" ? "bg-destructive/10 text-destructive" : "bg-pro/10 text-pro"
                  }`}>
                    {status?.status === "completed" ? (
                      <CheckCircle2 size={12} />
                    ) : status?.status === "failed" ? (
                      <XCircle size={12} />
                    ) : (
                      <Loader2 size={12} className="animate-spin" />
                    )}
                    <span className="text-[10px] font-semibold uppercase tracking-wider">{status?.status || "Starting"}</span>
                  </div>
                </div>

                {/* Progress bar */}
                <div className="mb-6">
                  <div className="flex justify-between text-[11px] text-text-muted mb-2">
                    <span>Overall Progress</span>
                    <span className="font-mono">{Math.round(((status?.completed_stages?.length || 0) / selectedAgents.size) * 100) || 0}%</span>
                  </div>
                  <div className="h-2 bg-void/50 rounded-full overflow-hidden border border-pro-glass-border">
                    <motion.div
                      className="h-full rounded-full bg-gradient-to-r from-pro to-pro-bright"
                      animate={{ width: `${((status?.completed_stages?.length || 0) / selectedAgents.size) * 100}%` }}
                      transition={{ duration: 0.3 }}
                    />
                  </div>
                </div>

                {/* Agents Status list */}
                <div className="space-y-2.5">
                  {AGENT_ORDER.filter((id) => selectedAgents.has(id)).map((agentId) => {
                    const meta = AGENT_META[agentId];
                    const isCompleted = status?.completed_stages?.includes(agentId);
                    const isActive = status?.current_stage === agentId && !isCompleted;
                    const isFailed = status?.status === "failed" && isActive;

                    return (
                      <div
                        key={agentId}
                        className={`flex items-center justify-between p-3 rounded-xl border transition-all duration-300 ${
                          isActive
                            ? "border-pro/30 bg-pro/[0.04]"
                            : isCompleted
                            ? "border-success/20 bg-success/[0.02]"
                            : "border-glass-border bg-glass/20 opacity-60"
                        }`}
                      >
                        <div className="flex items-center gap-2.5">
                          <div
                            className={`w-7 h-7 rounded-lg flex items-center justify-center ${
                              isCompleted ? "bg-success/10 text-success" : isActive ? "bg-pro/10 text-pro animate-pulse" : "bg-glass"
                            }`}
                          >
                            <Cpu size={13} style={{ color: isCompleted ? "#34d399" : meta.color }} />
                          </div>
                          <div>
                            <span className="text-[12px] font-medium text-text-primary">{meta.label}</span>
                          </div>
                        </div>

                        <div className="text-[11px] font-semibold">
                          {isCompleted ? (
                            <span className="text-success flex items-center gap-1"><CheckCircle2 size={12} /> Done</span>
                          ) : isFailed ? (
                            <span className="text-destructive flex items-center gap-1"><XCircle size={12} /> Failed</span>
                          ) : isActive ? (
                            <span className="text-pro flex items-center gap-1.5"><Loader2 size={12} className="animate-spin" /> Running</span>
                          ) : (
                            <span className="text-text-ghost">Pending</span>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Final Report */}
              {report && (
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="glass p-8">
                  <h3 className="text-[11px] font-semibold uppercase tracking-[2px] text-pro mb-6 flex items-center gap-2"><FileText size={14} /> Final Report</h3>
                  <div className="prose prose-invert prose-sm max-w-none" dangerouslySetInnerHTML={{
                    __html: report.replace(/^### (.+)$/gm, "<h3>$1</h3>").replace(/^## (.+)$/gm, "<h2>$1</h2>").replace(/^# (.+)$/gm, "<h1>$1</h1>").replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`(.+?)`/g, "<code>$1</code>").replace(/^- (.+)$/gm, "<li>$1</li>").replace(/\n/g, "<br>"),
                  }} />
                </motion.div>
              )}

              {/* Visualizations */}
              {vizs && vizs.length > 0 && (
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="glass p-8 mt-5">
                  <h3 className="text-[11px] font-semibold uppercase tracking-[2px] text-pro mb-6 flex items-center gap-2">
                    <BarChart3 size={14} className="text-pro" /> Workflow Visualizations
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {vizs.map((viz) => (
                      <div key={viz.type} className="viz-card overflow-hidden flex flex-col justify-between border border-glass-border rounded-xl bg-white/[0.01]">
                        <img
                          src={`data:image/png;base64,${viz.base64_png}`}
                          alt={viz.name}
                          className="w-full object-contain"
                        />
                        <div className="p-3.5 border-t border-glass-border bg-white/[0.01]">
                          <div className="text-[12px] font-semibold text-text-primary">{viz.name}</div>
                          <div className="text-[10px] text-text-muted mt-0.5">{viz.description}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
