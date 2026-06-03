"use client";

import { useEffect, useState, useRef, use, useMemo } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  CheckCircle2, Loader2, XCircle, Clock, ArrowRight, Radio,
  Database, Eraser, Wrench, Split, Brain, ShieldAlert,
  TrendingUp, PackageCheck, Activity, Cpu, type LucideIcon,
} from "lucide-react";
import Navbar from "@/components/layout/Navbar";
import { getStatus, StatusResponse } from "@/lib/api";
import { AGENT_ORDER, AGENT_META, type AgentId } from "@/lib/types";

const ICONS: Record<AgentId, LucideIcon> = {
  data_collection: Database,
  preprocessing: Eraser,
  feature_engineering: Wrench,
  data_splitting: Split,
  model_training: Brain,
  error_detection: ShieldAlert,
  improvement: TrendingUp,
  finalization: PackageCheck,
};

type StageState = "done" | "active" | "pending" | "failed";

export default function FreePipelinePage({ params }: { params: Promise<{ runId: string }> }) {
  const { runId } = use(params);
  const router = useRouter();
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const logRef = useRef<HTMLDivElement>(null);
  const [showLogs, setShowLogs] = useState(false);

  // ── Poll backend status (unchanged contract) ──────────────────────────────
  useEffect(() => {
    const poll = setInterval(async () => {
      try {
        const s = await getStatus(runId);
        setStatus(s);
        if (s.status === "completed") {
          clearInterval(poll);
          setTimeout(() => router.push(`/free/results/${runId}`), 2200);
        }
        if (s.status === "failed") clearInterval(poll);
      } catch { /* ignore transient errors */ }
    }, 1500);
    return () => clearInterval(poll);
  }, [runId, router]);

  useEffect(() => {
    const t = setInterval(() => setElapsed((e) => e + 1), 1000);
    if (status?.status === "completed" || status?.status === "failed") return () => clearInterval(t);
    return () => clearInterval(t);
  }, [status?.status]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: "smooth" });
  }, [status?.logs]);

  const completedSet = useMemo(() => new Set(status?.completed_stages || []), [status?.completed_stages]);
  const isCompleted = status?.status === "completed";
  const isFailed = status?.status === "failed";
  const progress = isCompleted ? 100 : (completedSet.size / AGENT_ORDER.length) * 100;
  const fmt = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, "0")}`;

  const stageState = (id: AgentId): StageState => {
    if (completedSet.has(id)) return "done";
    if (status?.current_stage === id) return isFailed ? "failed" : "active";
    return "pending";
  };

  const activeId = AGENT_ORDER.find((id) => stageState(id) === "active") ?? null;
  const activeMeta = activeId ? AGENT_META[activeId] : null;

  const statusLabel = isCompleted ? "Pipeline Complete" : isFailed ? "Pipeline Failed" : "Executing";
  const statusColor = isCompleted ? "#34d399" : isFailed ? "#fb7185" : "#6366f1";

  return (
    <>
      <Navbar />
      <main className="pt-20 pb-16 min-h-screen">
        <div className="max-w-[1400px] mx-auto px-6">

          {/* ── Mission header ─────────────────────────────────────────── */}
          <motion.div
            initial={{ opacity: 0, y: -12 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass px-6 py-4 mb-5 flex flex-wrap items-center gap-x-8 gap-y-3"
          >
            <div className="flex items-center gap-3">
              <div className="relative w-10 h-10 rounded-xl grid place-items-center" style={{ background: `${statusColor}14`, border: `1px solid ${statusColor}40` }}>
                <Cpu size={18} style={{ color: statusColor }} />
                {!isCompleted && !isFailed && (
                  <span className="absolute inset-0 rounded-xl animate-ping" style={{ background: `${statusColor}22` }} />
                )}
              </div>
              <div>
                <div className="text-[10px] uppercase tracking-[2.5px] text-text-muted font-bold">Mission Control</div>
                <div className="text-[15px] font-bold text-text-primary leading-tight">Autonomous ML Pipeline</div>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <span className="px-3 py-1.5 rounded-full text-[11px] font-bold flex items-center gap-1.5" style={{ background: `${statusColor}14`, color: statusColor }}>
                {isCompleted ? <CheckCircle2 size={12} /> : isFailed ? <XCircle size={12} /> : <Radio size={12} className="animate-pulse" />}
                {statusLabel}
              </span>
              <span className="px-3 py-1.5 rounded-full bg-glass border border-glass-border text-[11px] font-mono text-text-secondary flex items-center gap-1.5">
                <Clock size={11} /> {fmt(elapsed)}
              </span>
              <span className="px-3 py-1.5 rounded-full bg-glass border border-glass-border text-[11px] font-mono text-text-ghost">
                #{runId.slice(0, 8)}
              </span>
              <button
                onClick={() => setShowLogs(!showLogs)}
                className={`px-3 py-1.5 rounded-full text-[11px] font-semibold border transition-all duration-300 flex items-center gap-1.5 ${
                  showLogs
                    ? "bg-accent/15 border-accent/30 text-accent"
                    : "bg-white/[0.03] border-white/[0.08] text-text-secondary hover:border-white/[0.18]"
                }`}
              >
                <Radio size={11} className={showLogs ? "animate-pulse" : ""} />
                {showLogs ? "Hide Live Logs" : "Show Live Logs"}
              </button>
            </div>

            {/* Progress bar */}
            <div className="flex-1 min-w-[220px] flex items-center gap-3">
              <div className="flex-1 h-1.5 rounded-full bg-white/5 overflow-hidden">
                <motion.div
                  className="h-full rounded-full"
                  style={{ background: isCompleted ? "#34d399" : isFailed ? "#fb7185" : "linear-gradient(90deg,#6366f1,#818cf8)" }}
                  animate={{ width: `${progress}%` }}
                  transition={{ duration: 0.6, ease: [0.25, 0.4, 0, 1] }}
                />
              </div>
              <span className="text-[12px] font-mono font-bold tabular-nums" style={{ color: statusColor }}>
                {Math.round(progress)}%
              </span>
            </div>
          </motion.div>

          {/* ── Three-column mission deck ──────────────────────────────── */}
          <div className={`grid grid-cols-1 ${showLogs ? "lg:grid-cols-[300px_1fr_340px]" : "lg:grid-cols-[300px_1fr]"} gap-5`}>

            {/* LEFT — Agent timeline */}
            <motion.aside
              initial={{ opacity: 0, x: -16 }}
              animate={{ opacity: 1, x: 0 }}
              className="glass p-4"
            >
              <div className="flex items-center gap-2 px-2 pb-3 mb-1">
                <Activity size={13} className="text-primary" />
                <span className="text-[11px] font-bold uppercase tracking-[2px] text-text-muted">Agent Timeline</span>
              </div>
              <div className="relative">
                {/* spine */}
                <div className="absolute left-[26px] top-3 bottom-3 w-px bg-glass-border" />
                <div className="space-y-1">
                  {AGENT_ORDER.map((id) => {
                    const st = stageState(id);
                    const meta = AGENT_META[id];
                    const Icon = ICONS[id];
                    const ao = status?.agent_outputs?.[id];
                    return (
                      <div key={id} className="relative flex items-center gap-3 px-2 py-2 rounded-xl transition-colors"
                        style={{ background: st === "active" ? "rgba(99,102,241,0.06)" : "transparent" }}>
                        <div className="relative z-10 shrink-0">
                          {st === "active" && (
                            <span className="absolute -inset-1 rounded-full animate-ping" style={{ background: `${meta.color}33` }} />
                          )}
                          <div className="relative w-9 h-9 rounded-full grid place-items-center"
                            style={{
                              background: st === "done" ? "rgba(52,211,153,0.12)" : st === "active" ? `${meta.color}1f` : st === "failed" ? "rgba(251,113,133,0.12)" : "rgba(255,255,255,0.03)",
                              border: `1.5px solid ${st === "done" ? "#34d39955" : st === "active" ? meta.color : st === "failed" ? "#fb718555" : "rgba(255,255,255,0.08)"}`,
                            }}>
                            {st === "done" ? <CheckCircle2 size={15} className="text-success" />
                              : st === "active" ? <Loader2 size={15} className="animate-spin" style={{ color: meta.color }} />
                              : st === "failed" ? <XCircle size={15} className="text-destructive" />
                              : <Icon size={14} className="text-text-ghost" />}
                          </div>
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="text-[12.5px] font-semibold truncate" style={{ color: st === "pending" ? "var(--color-text-ghost)" : st === "done" ? "var(--color-text-secondary)" : meta.color }}>
                            {meta.label}
                          </div>
                          <div className="text-[10px] text-text-ghost truncate">{meta.description}</div>
                        </div>
                        {ao?.duration_seconds != null && (
                          <span className="text-[10px] font-mono text-text-ghost shrink-0">{ao.duration_seconds.toFixed(1)}s</span>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            </motion.aside>

            {/* CENTER — Execution graph */}
            <motion.section
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              className="glass relative overflow-hidden p-6 min-h-[440px] flex flex-col"
            >
              <div className="aurora opacity-40" />
              <div className="relative flex items-center justify-between mb-6">
                <span className="text-[11px] font-bold uppercase tracking-[2px] text-text-muted flex items-center gap-2">
                  <Cpu size={13} className="text-primary" /> Execution Graph
                </span>
                {activeMeta && (
                  <span className="text-[11px] font-mono px-2.5 py-1 rounded-full" style={{ background: `${activeMeta.color}14`, color: activeMeta.color }}>
                    ▶ {activeMeta.label}
                  </span>
                )}
              </div>

              {/* Serpentine circuit: 4 + 4 */}
              <div className="relative flex-1 flex flex-col justify-center gap-10">
                {[AGENT_ORDER.slice(0, 4), AGENT_ORDER.slice(4, 8)].map((row, rowIdx) => (
                  <div key={rowIdx} className={`flex items-center justify-between gap-1 ${rowIdx === 1 ? "flex-row-reverse" : ""}`}>
                    {row.map((id, i) => {
                      const st = stageState(id);
                      const meta = AGENT_META[id];
                      const Icon = ICONS[id];
                      const globalIdx = rowIdx * 4 + i;
                      const showConnector = i < row.length - 1;
                      const nextId = row[i + 1];
                      const connectorDone = nextId ? completedSet.has(nextId) || completedSet.has(id) : false;
                      return (
                        <div key={id} className={`flex items-center ${rowIdx === 1 ? "flex-row-reverse" : ""} flex-1`}>
                          <div className="flex flex-col items-center gap-2 shrink-0">
                            <motion.div
                              animate={st === "active" ? { scale: [1, 1.08, 1] } : { scale: 1 }}
                              transition={{ duration: 1.4, repeat: st === "active" ? Infinity : 0 }}
                              className="relative w-16 h-16 rounded-2xl grid place-items-center"
                              style={{
                                background: st === "done" ? "rgba(52,211,153,0.10)" : st === "active" ? `${meta.color}1c` : "rgba(255,255,255,0.02)",
                                border: `1.5px solid ${st === "done" ? "#34d39955" : st === "active" ? meta.color : st === "failed" ? "#fb7185" : "rgba(255,255,255,0.08)"}`,
                                boxShadow: st === "active" ? `0 0 28px ${meta.color}55` : "none",
                              }}
                            >
                              {st === "active" && (
                                <span className="absolute -inset-1.5 rounded-2xl animate-ping" style={{ background: `${meta.color}1a` }} />
                              )}
                              {st === "done" ? <CheckCircle2 size={24} className="text-success" />
                                : st === "failed" ? <XCircle size={24} className="text-destructive" />
                                : <Icon size={22} style={{ color: st === "active" ? meta.color : "var(--color-text-ghost)" }} />}
                              <span className="absolute -top-1.5 -left-1.5 w-5 h-5 rounded-full grid place-items-center text-[9px] font-bold"
                                style={{ background: "var(--color-surface)", border: `1px solid ${st === "pending" ? "rgba(255,255,255,0.1)" : meta.color}`, color: st === "pending" ? "var(--color-text-ghost)" : meta.color }}>
                                {globalIdx + 1}
                              </span>
                            </motion.div>
                            <span className="text-[10.5px] font-semibold text-center w-[72px] leading-tight"
                              style={{ color: st === "pending" ? "var(--color-text-ghost)" : st === "done" ? "var(--color-text-secondary)" : meta.color }}>
                              {meta.shortLabel}
                            </span>
                          </div>
                          {showConnector && (
                            <div className="flex-1 h-[2px] mx-1 relative rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.07)" }}>
                              <motion.div
                                className="absolute inset-y-0 left-0 rounded-full"
                                style={{ background: connectorDone ? "linear-gradient(90deg,#34d399,#2dd4bf)" : `${meta.color}` }}
                                initial={{ width: "0%" }}
                                animate={{ width: connectorDone ? "100%" : st === "active" ? "60%" : "0%" }}
                                transition={{ duration: 0.6 }}
                              />
                              {(connectorDone || st === "active") && (
                                <motion.span
                                  className="absolute top-1/2 -translate-y-1/2 w-1.5 h-1.5 rounded-full bg-white"
                                  animate={{ left: ["0%", "100%"], opacity: [0, 1, 0] }}
                                  transition={{ duration: 1.4, repeat: Infinity, ease: "linear" }}
                                />
                              )}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>

              {/* Center status caption */}
              <div className="relative mt-6 text-center">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={statusLabel + (activeMeta?.id ?? "")}
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -6 }}
                    className="text-[13px] text-text-secondary"
                  >
                    {isCompleted ? "All agents finished — preparing your results…"
                      : isFailed ? "An agent reported a failure. Review the activity feed."
                      : activeMeta ? <>The <span className="font-semibold" style={{ color: activeMeta.color }}>{activeMeta.label}</span> agent is working…</>
                      : "Initializing autonomous pipeline…"}
                  </motion.div>
                </AnimatePresence>
              </div>
            </motion.section>

            {/* RIGHT — Live activity feed */}
            {showLogs && (
              <motion.aside
                initial={{ opacity: 0, x: 16 }}
                animate={{ opacity: 1, x: 0 }}
                className="glass flex flex-col overflow-hidden max-h-[440px] lg:max-h-none"
              >
                <div className="px-4 py-3 border-b border-glass-border flex items-center justify-between">
                  <span className="text-[11px] font-bold uppercase tracking-[2px] text-text-muted flex items-center gap-2">
                    <Radio size={12} className="text-accent" /> Live Activity
                  </span>
                  <span className="live-dot live-dot-success" />
                </div>
                <div ref={logRef} className="flex-1 overflow-y-auto p-4 space-y-2.5 font-mono text-[11px] min-h-[200px]">
                  <AnimatePresence initial={false}>
                    {(status?.logs || []).map((log, i) => {
                      const ok = log.msg.includes("✓");
                      const bad = log.msg.includes("✗") || log.msg.toLowerCase().includes("error");
                      const dot = bad ? "#fb7185" : ok ? "#34d399" : "#6366f1";
                      return (
                        <motion.div
                          key={i}
                          initial={{ opacity: 0, x: 8 }}
                          animate={{ opacity: 1, x: 0 }}
                          className="flex items-start gap-2.5 leading-relaxed"
                        >
                          <span className="mt-1 w-1.5 h-1.5 rounded-full shrink-0" style={{ background: dot, boxShadow: `0 0 6px ${dot}` }} />
                          <span className={bad ? "text-destructive" : ok ? "text-text-secondary" : "text-text-muted"}>{log.msg}</span>
                        </motion.div>
                      );
                    })}
                  </AnimatePresence>
                  {(!status?.logs || status.logs.length === 0) && (
                    <div className="text-text-ghost flex items-center gap-2"><Loader2 size={12} className="animate-spin" /> Awaiting first event…</div>
                  )}
                </div>
              </motion.aside>
            )}
          </div>

          {/* ── Completion CTA ─────────────────────────────────────────── */}
          <AnimatePresence>
            {isCompleted && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-7 text-center"
              >
                <button
                  onClick={() => router.push(`/free/results/${runId}`)}
                  className="group inline-flex items-center gap-3 px-8 py-4 rounded-2xl bg-gradient-to-r from-success to-emerald-400 text-void font-bold text-[15px] hover:shadow-lg hover:shadow-success/20 transition-all duration-300"
                >
                  View Results <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />
                </button>
              </motion.div>
            )}
            {isFailed && status?.error && (
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-7 glass-sm p-4 border border-destructive/20 flex items-start gap-3 max-w-[700px] mx-auto"
              >
                <XCircle size={16} className="text-destructive shrink-0 mt-0.5" />
                <div>
                  <div className="text-[13px] font-semibold text-destructive">Pipeline failed</div>
                  <div className="text-[12px] text-text-muted mt-0.5 font-mono">{status.error}</div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

        </div>
      </main>
    </>
  );
}
