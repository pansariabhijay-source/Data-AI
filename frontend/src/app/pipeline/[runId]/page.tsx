"use client";

import { useEffect, useState, useRef, use } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { CheckCircle2, Circle, Loader2, XCircle, Clock, ArrowRight } from "lucide-react";
import Navbar from "@/components/layout/Navbar";
import { getStatus, StatusResponse } from "@/lib/api";
import { fadeUp, stagger } from "@/lib/animations";

const STAGES = [
  { key: "data_collection", label: "Ingest", desc: "Loading & profiling" },
  { key: "preprocessing", label: "Clean", desc: "Imputation & outliers" },
  { key: "feature_engineering", label: "Engineer", desc: "Encoding & selection" },
  { key: "data_splitting", label: "Split", desc: "Train / val / test" },
  { key: "model_training", label: "Train", desc: "Multi-model training" },
  { key: "error_detection", label: "Audit", desc: "Quality checks" },
  { key: "improvement", label: "Improve", desc: "Hyperparameter tuning" },
  { key: "finalization", label: "Finalize", desc: "Reports & artifacts" },
];

export default function PipelinePage({ params }: { params: Promise<{ runId: string }> }) {
  const { runId } = use(params);
  const router = useRouter();
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const poll = setInterval(async () => {
      try {
        const s = await getStatus(runId);
        setStatus(s);
        if (s.status === "completed") {
          clearInterval(poll);
          setTimeout(() => router.push(`/report/${runId}`), 2000);
        }
        if (s.status === "failed") clearInterval(poll);
      } catch { /* ignore */ }
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

  const completedSet = new Set(status?.completed_stages || []);
  const isCompleted = status?.status === "completed";
  const isFailed = status?.status === "failed";
  const formatTime = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, "0")}`;

  return (
    <>
      <Navbar />

      <main className="pt-28 pb-20 min-h-screen">
        <div className="max-w-[1100px] mx-auto px-8">

          {/* Header */}
          <motion.div variants={stagger} initial="hidden" animate="visible" className="mb-12">
            <motion.div variants={fadeUp} className="flex items-center gap-3 mb-3">
              <span className="text-[11px] font-semibold uppercase tracking-[3px] text-accent">
                Live Execution
              </span>
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full glass-sm">
                <Clock size={11} className="text-text-muted" />
                <span className="text-[11px] text-text-muted font-mono tabular-nums">{formatTime(elapsed)}</span>
              </div>
            </motion.div>
            <motion.h1 variants={fadeUp} className="font-display text-3xl md:text-4xl font-bold text-text-primary tracking-tight">
              Pipeline Orchestration
            </motion.h1>
            <motion.p variants={fadeUp} className="text-text-secondary mt-2 font-light">
              Run <span className="font-mono text-accent text-[13px]">{runId}</span>
            </motion.p>
          </motion.div>

          {/* Pipeline Graph */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.8, ease: [0.25, 0.4, 0, 1] }}
            className="glass p-8 md:p-10 mb-6"
          >
            <div className="flex items-start justify-between gap-2 overflow-x-auto pb-2">
              {STAGES.map((stage, i) => {
                const completed = completedSet.has(stage.key);
                const active = status?.current_stage === stage.key && !completed;
                const failed = isFailed && active;

                return (
                  <div key={stage.key} className="flex items-start gap-2 shrink-0">
                    <div className="flex flex-col items-center gap-2.5 min-w-[80px]">
                      {/* Node */}
                      <div className="relative">
                        {active && !failed && (
                          <div className="absolute inset-0 rounded-2xl bg-accent/20 animate-ping" style={{ animationDuration: "2s" }} />
                        )}
                        <div
                          className={`relative w-14 h-14 rounded-2xl border-2 flex items-center justify-center transition-all duration-700 ${
                            completed
                              ? "border-success/40 bg-success/[0.08]"
                              : active && !failed
                              ? "border-accent/50 bg-accent/[0.08] shadow-lg shadow-accent-glow"
                              : failed
                              ? "border-destructive/40 bg-destructive/[0.08]"
                              : "border-glass-border bg-glass"
                          }`}
                        >
                          {completed ? (
                            <CheckCircle2 size={20} className="text-success" strokeWidth={1.5} />
                          ) : active && !failed ? (
                            <Loader2 size={20} className="text-accent animate-spin" strokeWidth={1.5} />
                          ) : failed ? (
                            <XCircle size={20} className="text-destructive" strokeWidth={1.5} />
                          ) : (
                            <Circle size={20} className="text-text-ghost" strokeWidth={1.5} />
                          )}
                        </div>
                      </div>

                      {/* Label */}
                      <div className="text-center">
                        <div className={`text-[11px] font-semibold uppercase tracking-wider transition-colors duration-500 ${
                          completed ? "text-success" : active ? "text-accent" : "text-text-ghost"
                        }`}>
                          {stage.label}
                        </div>
                        <div className="text-[10px] text-text-ghost mt-0.5 hidden md:block">{stage.desc}</div>
                      </div>
                    </div>

                    {/* Edge */}
                    {i < STAGES.length - 1 && (
                      <div className="flex items-center pt-7 shrink-0">
                        <div className={`w-6 md:w-10 h-[2px] rounded-full transition-colors duration-700 ${
                          completed ? "bg-success/30" : "bg-glass-border"
                        }`} />
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </motion.div>

          {/* Live Logs */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, duration: 0.8, ease: [0.25, 0.4, 0, 1] }}
            className="glass overflow-hidden"
          >
            <div className="px-6 py-3 border-b border-glass-border flex items-center justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-[2px] text-text-muted">
                System Log
              </span>
              <div className={`w-2 h-2 rounded-full ${
                isCompleted ? "bg-success" : isFailed ? "bg-destructive" : "bg-accent animate-pulse"
              }`} />
            </div>

            <div ref={logRef} className="max-h-[300px] overflow-y-auto p-5 font-mono text-[12px] leading-[2]">
              <AnimatePresence>
                {(status?.logs || []).map((log, i) => {
                  const isSuccess = log.msg.includes("✓");
                  const isError = log.msg.includes("✗") || log.msg.includes("failed");
                  return (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.3 }}
                      className={`${isSuccess ? "text-success" : isError ? "text-destructive" : "text-text-muted"}`}
                    >
                      <span className="text-text-ghost mr-3">
                        {log.time ? new Date(log.time).toLocaleTimeString() : "--:--:--"}
                      </span>
                      {log.msg}
                    </motion.div>
                  );
                })}
              </AnimatePresence>
              {!status?.logs?.length && (
                <div className="text-text-ghost">Waiting for pipeline to start...</div>
              )}
            </div>
          </motion.div>

          {/* Completion CTA */}
          {isCompleted && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="mt-6 text-center"
            >
              <button
                onClick={() => router.push(`/report/${runId}`)}
                className="group inline-flex items-center gap-3 px-8 py-4 rounded-2xl bg-white text-void font-semibold text-[15px] hover:bg-white/90 transition-all duration-300"
              >
                View Results
                <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform duration-300" />
              </button>
            </motion.div>
          )}
        </div>
      </main>
    </>
  );
}
