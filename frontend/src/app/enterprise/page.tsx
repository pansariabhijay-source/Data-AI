"use client";

import { useState, useCallback, useEffect, useMemo } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import {
  Upload, ArrowRight, ArrowUpRight, CheckCircle2, XCircle, Loader2,
  History, FileText, Cpu, Workflow, Trophy, Sparkles, Play, Clock,
} from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import { useAuthStore } from "@/store/useAuthStore";
import { uploadDataset, listExperiments } from "@/lib/api";

type RunSummary = Awaited<ReturnType<typeof listExperiments>>["runs"][number];

function greeting() {
  const h = new Date().getHours();
  if (h < 12) return "Good morning";
  if (h < 18) return "Good afternoon";
  return "Good evening";
}

function fmtDuration(s: number | null): string {
  if (s == null || !Number.isFinite(s) || s < 0) return "—";
  if (s < 60) return `${s.toFixed(1)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  return `${Math.floor(s / 3600)}h`;
}

function statusConfig(status: string) {
  if (status === "completed") return { Icon: CheckCircle2, cls: "text-success", bg: "bg-success/10" };
  if (status === "failed") return { Icon: XCircle, cls: "text-destructive", bg: "bg-destructive/10" };
  return { Icon: Loader2, cls: "text-primary", bg: "bg-primary/10" };
}

export default function WorkspaceHome() {
  const router = useRouter();
  const { dataset, setDataset } = useAppStore();
  const user = useAuthStore((s) => s.user);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);

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

  const inFlight = useMemo(() => runs.find((r) => r.status === "running" || r.status === "starting"), [runs]);
  const lastCompleted = useMemo(() => runs.find((r) => r.status === "completed"), [runs]);
  const recent = useMemo(() => runs.slice(0, 5), [runs]);
  const reportCount = useMemo(() => runs.filter((r) => r.status === "completed").length, [runs]);

  const handleFile = useCallback(async (file: File) => {
    setUploading(true);
    setUploadError(null);
    try {
      const res = await uploadDataset(file);
      setDataset({
        filename: res.filename, path: res.path, columns: res.columns,
        dtypes: res.dtypes, n_rows: res.n_rows, preview: res.preview,
      });
      router.push("/enterprise/workflow");
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }, [setDataset, router]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, [handleFile]);

  return (
    <div className="min-h-screen px-6 md:px-10 py-10 max-w-[1180px]">
      {/* Greeting */}
      <motion.header
        initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}
        className="mb-8"
      >
        <p className="text-[13px] text-text-muted">{greeting()},</p>
        <h1 className="font-display text-3xl md:text-[2.6rem] font-bold text-text-primary tracking-tight mt-0.5">
          {user?.username ?? "there"}
        </h1>
      </motion.header>

      {/* Continue / primary action band */}
      <motion.section
        initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.05 }}
        className="grid lg:grid-cols-[1.5fr_1fr] gap-4 mb-4"
      >
        {/* Continue previous work OR start fresh */}
        {inFlight ? (
          <Link
            href={`/free/pipeline/${inFlight.run_id}`}
            className="glass-panel rounded-2xl p-6 relative overflow-hidden group hover:border-primary/30 transition-colors"
          >
            <div className="absolute top-0 right-0 w-48 h-48 bg-primary/8 blur-3xl -mr-20 -mt-20 pointer-events-none" />
            <div className="relative z-10 flex items-center justify-between">
              <div>
                <span className="inline-flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider text-primary">
                  <Loader2 size={12} className="animate-spin" style={{ animationDuration: "2.5s" }} /> Pipeline live
                </span>
                <h3 className="text-[18px] font-semibold text-text-primary mt-2">Continue where you left off</h3>
                <p className="text-[13px] text-text-muted mt-1 font-mono">{inFlight.run_id}</p>
              </div>
              <ArrowRight size={20} className="text-text-ghost group-hover:text-primary group-hover:translate-x-1 transition-all" />
            </div>
          </Link>
        ) : lastCompleted ? (
          <Link
            href={lastCompleted.mode === "enterprise" ? `/report/${lastCompleted.run_id}` : `/free/results/${lastCompleted.run_id}`}
            className="glass-panel rounded-2xl p-6 relative overflow-hidden group hover:border-accent/30 transition-colors"
          >
            <div className="absolute top-0 right-0 w-48 h-48 bg-accent/8 blur-3xl -mr-20 -mt-20 pointer-events-none" />
            <div className="relative z-10 flex items-center justify-between">
              <div>
                <span className="inline-flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider text-accent">
                  <Trophy size={12} /> Latest result
                </span>
                <h3 className="text-[18px] font-semibold text-text-primary mt-2">{lastCompleted.best_model ?? "View report"}</h3>
                <p className="text-[13px] text-text-muted mt-1">
                  {typeof lastCompleted.best_metric_value === "number" && (
                    <span className="font-mono text-accent">{lastCompleted.best_metric_value.toFixed(4)}</span>
                  )}{" "}<span className="font-mono">· {lastCompleted.run_id}</span>
                </p>
              </div>
              <ArrowRight size={20} className="text-text-ghost group-hover:text-accent group-hover:translate-x-1 transition-all" />
            </div>
          </Link>
        ) : (
          <div className="glass-panel rounded-2xl p-6 flex items-center gap-4">
            <div className="h-12 w-12 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center shrink-0">
              <Sparkles size={20} className="text-primary" />
            </div>
            <div>
              <h3 className="text-[16px] font-semibold text-text-primary">Welcome to your workspace</h3>
              <p className="text-[13px] text-text-muted mt-0.5">Upload a dataset to launch your first autonomous pipeline.</p>
            </div>
          </div>
        )}

        {/* Upload */}
        <label
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={`glass-panel rounded-2xl p-6 flex flex-col items-center justify-center text-center cursor-pointer relative overflow-hidden transition-all ${
            dragging ? "border-primary/50 bg-primary/[0.04]" : "hover:border-primary/25"
          }`}
        >
          <input type="file" accept=".csv,.tsv,.txt" className="hidden"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />
          <div className="h-12 w-12 rounded-full bg-primary/15 flex items-center justify-center mb-3">
            {uploading ? <Loader2 size={22} className="text-primary animate-spin" /> : <Upload size={22} className="text-primary" />}
          </div>
          <p className="text-[14px] font-semibold text-text-primary">{uploading ? "Uploading…" : "Upload dataset"}</p>
          <p className="text-[11px] text-text-muted mt-1">{dataset ? `Replace ${dataset.filename}` : "Drop a CSV · TSV · TXT · up to 1 GB"}</p>
        </label>

        {uploadError && (
          <div className="mt-3 flex items-start gap-2 rounded-xl border border-destructive/25 bg-destructive/[0.05] px-4 py-3">
            <span className="text-destructive text-[13px]">{uploadError}</span>
          </div>
        )}
      </motion.section>

      {/* Suggested actions */}
      <motion.section
        initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.1 }}
        className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-8"
      >
        {[
          { href: "/enterprise/workflow", label: "Workflow Builder", desc: "Design a custom pipeline", Icon: Workflow },
          { href: "/enterprise/agents", label: "Agent Console", desc: "Run agents step by step", Icon: Cpu },
          { href: "/enterprise/runs", label: "Run History", desc: `${runs.length} total`, Icon: History },
          { href: "/enterprise/reports", label: "Reports", desc: `${reportCount} available`, Icon: FileText },
        ].map((a) => (
          <Link key={a.href} href={a.href}
            className="glass-panel rounded-xl p-4 group hover:border-primary/25 transition-colors">
            <a.Icon size={18} className="text-primary mb-3" strokeWidth={1.75} />
            <p className="text-[13px] font-semibold text-text-primary flex items-center gap-1">
              {a.label}
              <ArrowUpRight size={13} className="text-text-ghost group-hover:text-primary transition-colors" />
            </p>
            <p className="text-[11px] text-text-muted mt-0.5">{a.desc}</p>
          </Link>
        ))}
      </motion.section>

      {/* Recent runs */}
      <motion.section
        initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.15 }}
        className="glass-panel rounded-2xl overflow-hidden"
      >
        <div className="px-5 py-4 border-b border-white/[0.06] flex items-center justify-between">
          <h2 className="font-display text-[20px] text-text-primary flex items-center gap-2">
            <History size={16} className="text-text-muted" /> Recent runs
          </h2>
          <Link href="/enterprise/runs" className="text-[12px] font-medium text-primary hover:underline">View all</Link>
        </div>
        {loading ? (
          <div className="p-12 flex items-center justify-center"><Loader2 className="text-primary animate-spin" size={20} /></div>
        ) : recent.length === 0 ? (
          <div className="p-12 text-center">
            <Play size={28} className="text-text-ghost mx-auto mb-2" strokeWidth={1.5} />
            <p className="text-sm text-text-secondary">No runs yet. Upload a dataset to begin.</p>
          </div>
        ) : (
          <div className="divide-y divide-white/[0.04]">
            {recent.map((r) => {
              const c = statusConfig(r.status);
              const href = r.mode === "enterprise" ? `/report/${r.run_id}` : `/free/results/${r.run_id}`;
              return (
                <Link key={r.run_id} href={href} className="flex items-center justify-between px-5 py-3.5 hover:bg-white/[0.025] transition-colors group">
                  <div className="flex items-center gap-3 min-w-0">
                    <span className={`h-9 w-9 rounded-lg ${c.bg} ${c.cls} flex items-center justify-center shrink-0`}>
                      <c.Icon size={17} className={r.status === "running" || r.status === "starting" ? "animate-spin" : ""} style={{ animationDuration: "2.5s" }} />
                    </span>
                    <div className="min-w-0">
                      <p className="text-[13px] font-mono font-medium text-text-primary truncate group-hover:text-primary transition-colors">{r.run_id}</p>
                      <p className="text-[11px] text-text-muted truncate">
                        {r.best_model ? `Best: ${r.best_model}` : `Stage ${(r.completed_stages?.length ?? 0)}/8`}
                        {" · "}{r.started_at ? new Date(r.started_at).toLocaleDateString() : "—"}
                      </p>
                    </div>
                  </div>
                  <span className="hidden sm:flex items-center gap-1.5 text-[11px] text-text-muted shrink-0">
                    <Clock size={11} /> {fmtDuration(r.duration_seconds)}
                  </span>
                </Link>
              );
            })}
          </div>
        )}
      </motion.section>
    </div>
  );
}
