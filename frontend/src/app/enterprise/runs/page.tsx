"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  History, CheckCircle2, XCircle, Loader2, Search, Trophy,
  ArrowUpRight, Inbox, Cpu, Clock,
} from "lucide-react";
import { listExperiments } from "@/lib/api";

type RunSummary = Awaited<ReturnType<typeof listExperiments>>["runs"][number];
type Filter = "all" | "completed" | "running" | "failed";

function fmtDuration(s: number | null): string {
  if (s == null || !Number.isFinite(s) || s < 0) return "—";
  if (s < 60) return `${s.toFixed(1)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m`;
}

function statusConfig(status: string) {
  if (status === "completed") return { Icon: CheckCircle2, cls: "text-success", bg: "bg-success/10", label: "Completed" };
  if (status === "failed") return { Icon: XCircle, cls: "text-destructive", bg: "bg-destructive/10", label: "Failed" };
  return { Icon: Loader2, cls: "text-primary", bg: "bg-primary/10", label: "Running" };
}

export default function RunsPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<Filter>("all");
  const [query, setQuery] = useState("");

  useEffect(() => {
    let cancelled = false;
    const fetchRuns = () =>
      listExperiments()
        .then((d) => { if (!cancelled) { setRuns(d.runs || []); setLoading(false); } })
        .catch(() => { if (!cancelled) setLoading(false); });
    fetchRuns();
    const t = setInterval(fetchRuns, 5000);
    return () => { cancelled = true; clearInterval(t); };
  }, []);

  const counts = useMemo(() => ({
    all: runs.length,
    completed: runs.filter((r) => r.status === "completed").length,
    running: runs.filter((r) => r.status === "running" || r.status === "starting").length,
    failed: runs.filter((r) => r.status === "failed").length,
  }), [runs]);

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    return runs.filter((r) => {
      const matchFilter =
        filter === "all" ? true :
        filter === "running" ? (r.status === "running" || r.status === "starting") :
        r.status === filter;
      const matchQuery = !q || r.run_id.toLowerCase().includes(q) || (r.best_model ?? "").toLowerCase().includes(q);
      return matchFilter && matchQuery;
    });
  }, [runs, filter, query]);

  return (
    <div className="min-h-screen px-6 md:px-10 py-10">
      <motion.header
        initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}
        className="flex flex-wrap items-end justify-between gap-4 mb-7"
      >
        <div>
          <h1 className="font-display text-3xl md:text-4xl font-bold text-text-primary tracking-tight flex items-center gap-3">
            <History size={28} className="text-primary" strokeWidth={1.75} />
            Runs
          </h1>
          <p className="text-sm text-text-secondary mt-1">Every pipeline you&apos;ve launched, persisted across sessions.</p>
        </div>
        <div className="relative">
          <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-ghost" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search run id or model…"
            className="w-64 pl-9 pr-3 py-2 rounded-xl bg-white/[0.03] border border-white/[0.08] text-[13px] text-text-primary placeholder:text-text-ghost focus:outline-none focus:border-primary/40 transition-all"
          />
        </div>
      </motion.header>

      {/* Filter tabs */}
      <div className="flex items-center gap-1.5 mb-5">
        {(["all", "completed", "running", "failed"] as Filter[]).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-3.5 py-1.5 rounded-lg text-[12px] font-medium capitalize transition-all ${
              filter === f ? "bg-primary/15 text-primary border border-primary/25" : "text-text-muted hover:text-text-secondary border border-transparent"
            }`}
          >
            {f} <span className="text-text-ghost ml-1">{counts[f]}</span>
          </button>
        ))}
      </div>

      {/* Table */}
      <motion.div
        initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.1 }}
        className="glass-panel rounded-2xl overflow-hidden"
      >
        {loading ? (
          <div className="p-16 flex items-center justify-center"><Loader2 className="text-primary animate-spin" size={22} /></div>
        ) : visible.length === 0 ? (
          <div className="p-16 text-center">
            <Inbox size={34} className="text-text-ghost mx-auto mb-3" strokeWidth={1.5} />
            <p className="text-sm text-text-secondary">{runs.length === 0 ? "No runs yet." : "No runs match this filter."}</p>
            {runs.length === 0 && (
              <Link href="/enterprise" className="inline-flex items-center gap-1.5 text-primary text-[13px] font-medium mt-3 hover:gap-2.5 transition-all">
                Start your first run <ArrowUpRight size={14} />
              </Link>
            )}
          </div>
        ) : (
          <>
            {/* header row */}
            <div className="hidden md:grid grid-cols-[1.4fr_1fr_0.8fr_0.7fr_auto] gap-4 px-5 py-3 border-b border-white/[0.06] text-[10px] font-semibold uppercase tracking-wider text-text-ghost">
              <span>Run</span><span>Best model</span><span>Stages</span><span>Duration</span><span>Status</span>
            </div>
            <div className="divide-y divide-white/[0.04]">
              {visible.map((r, i) => {
                const c = statusConfig(r.status);
                const href = r.mode === "enterprise" ? `/report/${r.run_id}` : `/free/results/${r.run_id}`;
                return (
                  <motion.div
                    key={r.run_id}
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: Math.min(i * 0.03, 0.4) }}
                  >
                    <Link href={href} className="grid grid-cols-[1fr_auto] md:grid-cols-[1.4fr_1fr_0.8fr_0.7fr_auto] gap-4 px-5 py-4 items-center hover:bg-white/[0.025] transition-colors group">
                      <div className="min-w-0">
                        <p className="text-[13px] font-mono font-medium text-text-primary truncate group-hover:text-primary transition-colors">{r.run_id}</p>
                        <p className="text-[11px] text-text-muted mt-0.5">{r.started_at ? new Date(r.started_at).toLocaleString() : "—"}</p>
                      </div>
                      <div className="hidden md:flex items-center gap-1.5 min-w-0">
                        {r.best_model ? (
                          <>
                            <Trophy size={13} className="text-accent shrink-0" />
                            <span className="text-[12px] text-text-secondary truncate">{r.best_model}</span>
                            {typeof r.best_metric_value === "number" && (
                              <span className="text-[11px] font-mono text-accent shrink-0">{r.best_metric_value.toFixed(3)}</span>
                            )}
                          </>
                        ) : <span className="text-[12px] text-text-ghost">—</span>}
                      </div>
                      <div className="hidden md:flex items-center gap-1.5 text-[12px] text-text-muted">
                        <Cpu size={12} /> {(r.completed_stages?.length ?? 0)}/8
                      </div>
                      <div className="hidden md:flex items-center gap-1.5 text-[12px] text-text-muted">
                        <Clock size={12} /> {fmtDuration(r.duration_seconds)}
                      </div>
                      <span className={`shrink-0 inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full ${c.bg} ${c.cls} text-[10px] font-bold uppercase tracking-wider`}>
                        <c.Icon size={11} className={r.status === "running" || r.status === "starting" ? "animate-spin" : ""} style={{ animationDuration: "2.5s" }} />
                        {c.label}
                      </span>
                    </Link>
                  </motion.div>
                );
              })}
            </div>
          </>
        )}
      </motion.div>
    </div>
  );
}
