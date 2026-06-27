"use client";

import { useState, useCallback, useEffect, useMemo } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import {
  Upload, ArrowRight, ArrowUpRight, CheckCircle2, XCircle, Loader2,
  History, FileText, Cpu, Workflow, Trophy, Sparkles, Play, Clock,
  Activity, Database, Radio,
} from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import { useAuthStore } from "@/store/useAuthStore";
import { uploadDataset, listExperiments } from "@/lib/api";
import AnimatedCounter from "@/components/ui/AnimatedCounter";
import { fadeUp, stagger } from "@/lib/animations";

const SERIF = "'Instrument Serif', serif";

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

function relTime(iso?: string | null): string {
  if (!iso) return "—";
  const secs = (Date.now() - new Date(iso).getTime()) / 1000;
  if (!Number.isFinite(secs) || secs < 0) return "—";
  if (secs < 60) return "just now";
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
  return `${Math.floor(secs / 86400)}d ago`;
}

function statusConfig(status: string) {
  if (status === "completed") return { Icon: CheckCircle2, cls: "text-success", bg: "bg-success/10", dot: "bg-success" };
  if (status === "failed") return { Icon: XCircle, cls: "text-destructive", bg: "bg-destructive/10", dot: "bg-destructive" };
  return { Icon: Loader2, cls: "text-foreground", bg: "bg-white/[0.06]", dot: "bg-white/70" };
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
  const activeCount = useMemo(() => runs.filter((r) => r.status === "running" || r.status === "starting").length, [runs]);
  const lastExec = useMemo(() => relTime(runs[0]?.started_at), [runs]);

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

  const stats: { label: string; value: React.ReactNode; sub: string; Icon: typeof Activity; live?: boolean }[] = [
    { label: "Active pipelines", value: <AnimatedCounter value={activeCount} />, sub: activeCount > 0 ? "running now" : "standing by", Icon: Activity, live: activeCount > 0 },
    { label: "Reports generated", value: <AnimatedCounter value={reportCount} />, sub: "ready to review", Icon: FileText },
    { label: "Agents online", value: <AnimatedCounter value={8} />, sub: "all operational", Icon: Cpu, live: true },
    { label: "Last execution", value: lastExec, sub: "most recent run", Icon: Clock },
  ];

  return (
    <div className="relative min-h-screen px-6 md:px-10 lg:px-12 py-12 max-w-[1280px] mx-auto">
      {/* ═══════════════ HERO ═══════════════ */}
      <motion.header
        variants={stagger} initial="hidden" animate="visible"
        className="relative mb-12"
      >
        <motion.div variants={fadeUp} className="inline-flex items-center gap-2.5 px-3.5 py-1.5 rounded-full border border-white/10 bg-white/[0.03] backdrop-blur-md mb-7">
          <span className="live-dot live-dot-success" />
          <span className="text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">Mission Control · online</span>
        </motion.div>

        <motion.h1
          variants={fadeUp}
          className="text-5xl md:text-6xl lg:text-7xl leading-[0.95] tracking-[-1.5px] text-foreground"
          style={{ fontFamily: SERIF }}
        >
          {greeting()},{" "}
          <em className="not-italic text-muted-foreground">{user?.username ?? "operator"}.</em>
        </motion.h1>
        <motion.p variants={fadeUp} className="mt-5 max-w-xl text-base md:text-lg leading-relaxed text-muted-foreground">
          Your autonomous data science command center is active. Eight agents stand ready to
          ingest, engineer, train, and ship — on your command.
        </motion.p>
      </motion.header>

      {/* ═══════════════ STAT CARDS ═══════════════ */}
      <motion.section
        variants={stagger} initial="hidden" animate="visible"
        className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6"
      >
        {stats.map((s) => (
          <motion.div
            key={s.label}
            variants={fadeUp}
            className="group glass-panel rounded-2xl p-5 relative overflow-hidden transition-transform duration-500 hover:-translate-y-1"
          >
            <div className="pointer-events-none absolute inset-x-0 -top-px h-px bg-gradient-to-r from-transparent via-white/30 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            <div className="flex items-center justify-between mb-5">
              <span className="grid h-10 w-10 place-items-center rounded-xl bg-white/[0.05] border border-white/10">
                <s.Icon size={17} className="text-foreground/80" strokeWidth={1.75} />
              </span>
              {s.live && <span className="live-dot live-dot-success" />}
            </div>
            <div className="text-[40px] leading-none text-foreground" style={{ fontFamily: SERIF }}>
              {s.value}
            </div>
            <div className="mt-2 text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">{s.label}</div>
            <div className="mt-0.5 text-[11px] text-foreground/40">{s.sub}</div>
          </motion.div>
        ))}
      </motion.section>

      {/* ═══════════════ COMMAND BAND — continue + launch ═══════════════ */}
      <motion.section
        variants={stagger} initial="hidden" animate="visible"
        className="grid lg:grid-cols-[1.5fr_1fr] gap-4 mb-12"
      >
        {/* Continue previous work OR start fresh */}
        {inFlight ? (
          <motion.div variants={fadeUp}>
            <Link
              href={`/free/pipeline/${inFlight.run_id}`}
              className="group glass-panel rounded-2xl p-7 relative overflow-hidden flex h-full items-center justify-between hover:-translate-y-0.5 transition-transform duration-500"
            >
              <div className="pointer-events-none absolute -top-16 -right-16 w-56 h-56 rounded-full" style={{ background: "radial-gradient(circle, rgba(190,205,225,0.08), transparent 70%)" }} />
              <div className="relative z-10">
                <span className="inline-flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-foreground/80">
                  <Loader2 size={12} className="animate-spin" style={{ animationDuration: "2.5s" }} /> Pipeline live
                </span>
                <h3 className="font-display text-[26px] text-foreground mt-2.5">Resume mission</h3>
                <p className="text-[13px] text-muted-foreground mt-1 font-mono">{inFlight.run_id}</p>
              </div>
              <ArrowRight size={22} className="relative z-10 text-foreground/40 group-hover:text-foreground group-hover:translate-x-1 transition-all" />
            </Link>
          </motion.div>
        ) : lastCompleted ? (
          <motion.div variants={fadeUp}>
            <Link
              href={lastCompleted.mode === "enterprise" ? `/report/${lastCompleted.run_id}` : `/free/results/${lastCompleted.run_id}`}
              className="group glass-panel rounded-2xl p-7 relative overflow-hidden flex h-full items-center justify-between hover:-translate-y-0.5 transition-transform duration-500"
            >
              <div className="pointer-events-none absolute -top-16 -right-16 w-56 h-56 rounded-full" style={{ background: "radial-gradient(circle, rgba(190,205,225,0.08), transparent 70%)" }} />
              <div className="relative z-10">
                <span className="inline-flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-foreground/80">
                  <Trophy size={12} /> Latest result
                </span>
                <h3 className="font-display text-[26px] text-foreground mt-2.5">{lastCompleted.best_model ?? "View report"}</h3>
                <p className="text-[13px] text-muted-foreground mt-1">
                  {typeof lastCompleted.best_metric_value === "number" && (
                    <span className="font-mono text-foreground/80">{lastCompleted.best_metric_value.toFixed(4)}</span>
                  )}{" "}<span className="font-mono">· {lastCompleted.run_id}</span>
                </p>
              </div>
              <ArrowRight size={22} className="relative z-10 text-foreground/40 group-hover:text-foreground group-hover:translate-x-1 transition-all" />
            </Link>
          </motion.div>
        ) : (
          <motion.div variants={fadeUp} className="glass-panel rounded-2xl p-7 flex items-center gap-5">
            <div className="h-14 w-14 rounded-2xl bg-white/[0.05] border border-white/10 flex items-center justify-center shrink-0">
              <Sparkles size={22} className="text-foreground/80" />
            </div>
            <div>
              <h3 className="font-display text-[24px] text-foreground">Welcome aboard</h3>
              <p className="text-[13px] text-muted-foreground mt-1">Upload a dataset to launch your first autonomous pipeline.</p>
            </div>
          </motion.div>
        )}

        {/* Upload — primary launch surface (drag + click) */}
        <motion.div variants={fadeUp}>
          <label
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            className={`group glass-panel rounded-2xl p-7 h-full flex flex-col items-center justify-center text-center cursor-pointer relative overflow-hidden transition-all duration-300 ${
              dragging ? "border-white/30 bg-white/[0.05] scale-[1.01]" : "hover:border-white/15"
            }`}
          >
            <input type="file" accept=".csv,.tsv,.txt" className="hidden"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />
            <div className="h-13 w-13 rounded-full bg-white/[0.06] border border-white/10 flex items-center justify-center mb-3.5 p-3.5 group-hover:scale-110 transition-transform duration-500">
              {uploading ? <Loader2 size={22} className="text-foreground animate-spin" /> : <Upload size={22} className="text-foreground/80" />}
            </div>
            <p className="text-[14px] font-semibold text-foreground">{uploading ? "Uploading…" : "Upload dataset"}</p>
            <p className="text-[11px] text-muted-foreground mt-1">{dataset ? `Replace ${dataset.filename}` : "Drop a CSV · TSV · TXT · up to 1 GB"}</p>
          </label>
        </motion.div>

        {uploadError && (
          <div className="lg:col-span-2 flex items-start gap-2 rounded-xl border border-destructive/25 bg-destructive/[0.05] px-4 py-3">
            <span className="text-destructive text-[13px]">{uploadError}</span>
          </div>
        )}
      </motion.section>

      {/* ═══════════════ QUICK ACTIONS ═══════════════ */}
      <motion.section variants={fadeUp} initial="hidden" animate="visible" className="mb-12">
        <h2 className="font-display text-[26px] text-foreground mb-5">Quick actions</h2>
        <motion.div variants={stagger} initial="hidden" animate="visible" className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
          {[
            { kind: "upload" as const, label: "Upload Dataset", desc: "Launch a new run", Icon: Upload },
            { href: "/enterprise/workflow", label: "Workflow Builder", desc: "Design a pipeline", Icon: Workflow },
            { href: "/enterprise/agents", label: "Agent Console", desc: "Run agents stepwise", Icon: Cpu },
            { href: "/enterprise/reports", label: "Reports", desc: `${reportCount} available`, Icon: FileText },
            { href: "/enterprise/runs", label: "Run History", desc: `${runs.length} total`, Icon: History },
          ].map((a) => {
            const inner = (
              <>
                <div className="pointer-events-none absolute -inset-px rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500" style={{ background: "radial-gradient(ellipse 80% 60% at 50% 0%, rgba(190,205,225,0.10), transparent 70%)" }} />
                <div className="pointer-events-none absolute inset-x-0 -top-px h-px bg-gradient-to-r from-transparent via-white/40 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                <div className="relative z-10 flex flex-col gap-3">
                  <span className="grid h-11 w-11 place-items-center rounded-xl bg-white/[0.05] border border-white/10 group-hover:scale-110 transition-transform duration-500">
                    <a.Icon size={18} className="text-foreground/80" strokeWidth={1.75} />
                  </span>
                  <div>
                    <p className="text-[13px] font-semibold text-foreground flex items-center gap-1">
                      {a.label}
                      <ArrowUpRight size={13} className="text-foreground/30 group-hover:text-foreground transition-colors" />
                    </p>
                    <p className="text-[11px] text-muted-foreground mt-0.5">{a.desc}</p>
                  </div>
                </div>
              </>
            );
            const cls = "group glass-panel rounded-2xl p-5 relative overflow-hidden transition-transform duration-500 hover:-translate-y-1 cursor-pointer";
            return (
              <motion.div key={a.label} variants={fadeUp}>
                {a.kind === "upload" ? (
                  <label className={`${cls} block`}>
                    <input type="file" accept=".csv,.tsv,.txt" className="hidden"
                      onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />
                    {inner}
                  </label>
                ) : (
                  <Link href={a.href!} className={`${cls} block`}>{inner}</Link>
                )}
              </motion.div>
            );
          })}
        </motion.div>
      </motion.section>

      {/* ═══════════════ RECENT ACTIVITY — mission log ═══════════════ */}
      <motion.section variants={fadeUp} initial="hidden" animate="visible">
        <div className="flex items-center justify-between mb-5">
          <h2 className="font-display text-[26px] text-foreground flex items-center gap-2.5">
            <Radio size={18} className="text-muted-foreground" /> Recent activity
          </h2>
          <Link href="/enterprise/runs" className="inline-flex items-center gap-1.5 text-[12px] font-medium text-muted-foreground hover:text-foreground transition-colors">
            View all <ArrowRight size={13} />
          </Link>
        </div>

        {loading ? (
          <div className="glass-panel rounded-2xl p-16 flex items-center justify-center"><Loader2 className="text-foreground/70 animate-spin" size={22} /></div>
        ) : recent.length === 0 ? (
          <div className="glass-panel rounded-2xl p-16 text-center">
            <Database size={30} className="text-foreground/30 mx-auto mb-3" strokeWidth={1.5} />
            <p className="text-sm text-muted-foreground">No missions logged yet. Upload a dataset to begin.</p>
          </div>
        ) : (
          <div className="space-y-2.5">
            {recent.map((r, i) => {
              const c = statusConfig(r.status);
              const running = r.status === "running" || r.status === "starting";
              const href = r.mode === "enterprise" ? `/report/${r.run_id}` : `/free/results/${r.run_id}`;
              return (
                <motion.div
                  key={r.run_id}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: Math.min(i * 0.05, 0.3), duration: 0.45, ease: [0.25, 0.4, 0, 1] }}
                >
                  <Link href={href} className="group glass-panel rounded-xl px-5 py-4 flex items-center justify-between hover:-translate-y-0.5 hover:border-white/15 transition-all duration-300">
                    <div className="flex items-center gap-4 min-w-0">
                      <span className={`relative h-10 w-10 rounded-xl ${c.bg} ${c.cls} flex items-center justify-center shrink-0`}>
                        <c.Icon size={18} className={running ? "animate-spin" : ""} style={{ animationDuration: "2.5s" }} />
                      </span>
                      <div className="min-w-0">
                        <p className="text-[13px] font-mono font-medium text-foreground truncate group-hover:text-white transition-colors">{r.run_id}</p>
                        <p className="text-[11px] text-muted-foreground truncate mt-0.5">
                          {r.started_at ? new Date(r.started_at).toLocaleDateString(undefined, { month: "short", day: "numeric" }) : "—"}
                          {" · "}{relTime(r.started_at)}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      {r.best_model ? (
                        <span className="hidden sm:inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-white/[0.05] border border-white/10 text-[11px] text-foreground/80">
                          <Trophy size={11} /> {r.best_model}
                        </span>
                      ) : (
                        <span className="hidden sm:inline-flex items-center px-2.5 py-1 rounded-full bg-white/[0.05] border border-white/10 text-[11px] text-muted-foreground">
                          Stage {(r.completed_stages?.length ?? 0)}/8
                        </span>
                      )}
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-white/[0.04] border border-white/10 text-[11px] text-muted-foreground font-mono">
                        <Clock size={11} /> {fmtDuration(r.duration_seconds)}
                      </span>
                    </div>
                  </Link>
                </motion.div>
              );
            })}
          </div>
        )}
      </motion.section>
    </div>
  );
}
