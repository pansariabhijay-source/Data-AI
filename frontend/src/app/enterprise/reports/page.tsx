"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { FileText, Download, Loader2, Inbox, ArrowUpRight, Trophy, Database, Clock, Crosshair } from "lucide-react";
import { listExperiments, downloadReportPdf } from "@/lib/api";

type RunSummary = Awaited<ReturnType<typeof listExperiments>>["runs"][number];

function fmtDateTime(iso?: string | null): { date: string; time: string } {
  if (!iso) return { date: "—", time: "" };
  const d = new Date(iso);
  if (isNaN(d.getTime())) return { date: "—", time: "" };
  return {
    date: d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" }),
    time: d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" }),
  };
}

export default function ReportsPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listExperiments()
      .then((d) => { if (!cancelled) { setRuns(d.runs || []); setLoading(false); } })
      .catch(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, []);

  const reports = useMemo(() => runs.filter((r) => r.status === "completed"), [runs]);

  const handleDownload = async (runId: string) => {
    setDownloading(runId);
    try { await downloadReportPdf(runId); } catch (e) { console.error(e); } finally { setDownloading(null); }
  };

  return (
    <div className="min-h-screen px-6 md:px-10 py-10">
      <motion.header
        initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}
        className="mb-7"
      >
        <h1 className="font-display text-3xl md:text-4xl font-bold text-text-primary tracking-tight flex items-center gap-3">
          <FileText size={28} className="text-primary" strokeWidth={1.75} />
          Reports
        </h1>
        <p className="text-sm text-text-secondary mt-1">Generated outputs from every completed pipeline.</p>
      </motion.header>

      {loading ? (
        <div className="glass-panel rounded-2xl p-16 flex items-center justify-center"><Loader2 className="text-primary animate-spin" size={22} /></div>
      ) : reports.length === 0 ? (
        <div className="glass-panel rounded-2xl p-16 text-center">
          <Inbox size={34} className="text-text-ghost mx-auto mb-3" strokeWidth={1.5} />
          <p className="text-sm text-text-secondary">No reports yet — they appear once a pipeline completes.</p>
          <Link href="/enterprise" className="inline-flex items-center gap-1.5 text-primary text-[13px] font-medium mt-3 hover:gap-2.5 transition-all">
            Run a pipeline <ArrowUpRight size={14} />
          </Link>
        </div>
      ) : (
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {reports.map((r, i) => {
            const href = r.mode === "enterprise" ? `/report/${r.run_id}` : `/free/results/${r.run_id}`;
            const isPro = r.mode === "enterprise";
            const { date, time } = fmtDateTime(r.completed_at || r.started_at);
            const datasetName = r.dataset || "Untitled dataset";
            return (
              <motion.div
                key={r.run_id}
                initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: Math.min(i * 0.04, 0.4) }}
                className="group relative rounded-2xl border border-glass-border bg-gradient-to-b from-white/[0.035] to-white/[0.01] p-5 flex flex-col overflow-hidden hover:border-primary/30 hover:shadow-lg hover:shadow-primary/[0.06] transition-all duration-300"
              >
                <div className="absolute top-0 right-0 h-24 w-24 bg-gradient-to-bl from-primary/[0.07] to-transparent rounded-bl-[60px] pointer-events-none" />

                {/* header: dataset icon + mode pill */}
                <div className="flex items-start justify-between mb-3.5 relative z-10">
                  <div className="h-10 w-10 rounded-xl bg-primary/12 border border-primary/25 flex items-center justify-center shrink-0">
                    <Database size={17} className="text-primary" strokeWidth={1.75} />
                  </div>
                  <span className={`text-[9px] font-bold uppercase tracking-[1.5px] px-2.5 py-1 rounded-full border ${
                    isPro ? "bg-pro/10 text-pro-bright border-pro/25" : "bg-success/10 text-success border-success/25"
                  }`}>
                    {isPro ? "Pro" : "Free"}
                  </span>
                </div>

                {/* dataset name — the headline the user identifies by */}
                <h3 className="text-[15px] font-semibold text-text-primary leading-snug break-words line-clamp-2" title={datasetName}>
                  {datasetName}
                </h3>
                {r.target && (
                  <p className="mt-1 flex items-center gap-1.5 text-[11px] text-text-muted">
                    <Crosshair size={11} className="text-text-ghost" /> target: <span className="text-text-secondary font-medium">{r.target}</span>
                  </p>
                )}

                {/* champion */}
                {r.best_model && (
                  <div className="mt-3 flex items-center gap-1.5 text-[12px] text-text-secondary">
                    <Trophy size={12} className="text-pro-gold shrink-0" />
                    <span className="truncate">{r.best_model}</span>
                    {typeof r.best_metric_value === "number" && (
                      <span className="ml-auto font-mono font-semibold text-accent shrink-0">{r.best_metric_value.toFixed(3)}</span>
                    )}
                  </div>
                )}

                {/* date + time + run id */}
                <div className="mt-3.5 pt-3.5 border-t border-glass-border/50 space-y-1.5">
                  <div className="flex items-center gap-1.5 text-[11px] text-text-muted">
                    <Clock size={11} className="text-text-ghost" />
                    <span className="text-text-secondary">{date}</span>
                    {time && <span className="text-text-ghost">· {time}</span>}
                  </div>
                  <div className="text-[9.5px] font-mono text-text-ghost truncate" title={r.run_id}>{r.run_id}</div>
                </div>

                {/* actions */}
                <div className="mt-4 flex items-center gap-2 relative z-10">
                  <Link href={href} className="flex-1 inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg bg-white/[0.04] border border-white/[0.07] text-[12px] font-medium text-text-secondary hover:text-text-primary hover:bg-white/[0.08] transition-all">
                    View report <ArrowUpRight size={13} />
                  </Link>
                  <button
                    onClick={() => handleDownload(r.run_id)}
                    disabled={downloading === r.run_id}
                    title="Download PDF"
                    className="inline-flex items-center justify-center px-3 py-2 rounded-lg bg-primary/12 border border-primary/25 text-primary hover:bg-primary/20 transition-all disabled:opacity-50"
                  >
                    {downloading === r.run_id ? <Loader2 size={14} className="animate-spin" /> : <Download size={14} />}
                  </button>
                </div>
              </motion.div>
            );
          })}
        </div>
      )}
    </div>
  );
}
