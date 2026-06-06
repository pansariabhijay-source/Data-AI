"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload, ChevronDown, ArrowRight, Database, Columns, Rows3,
  Sparkles, Eye, BarChart3, Brain, Crown, Lock,
  Workflow, Cpu, FlaskConical, Activity, FolderOpen,
  CheckCircle2, Zap, TrendingUp, X,
} from "lucide-react";
import Navbar from "@/components/layout/Navbar";
import { uploadDataset, startPipeline, UploadResponse } from "@/lib/api";
import { useAppStore } from "@/store/useAppStore";
import { fadeUp, stagger } from "@/lib/animations";

const PRO_FEATURES = [
  {
    label: "Workflow Builder",
    tagline: "Custom agent pipelines",
    desc: "Choose exactly which agents run, in what order. Full control over your ML workflow.",
    icon: Workflow,
    iconColor: "text-pro",
    accent: "rgba(129,140,248,0.12)",
    stat: "12+ templates",
    statColor: "text-pro",
  },
  {
    label: "Agent Console",
    tagline: "Step-by-step execution",
    desc: "Trigger individual agents, inspect live output, and debug each ML stage in isolation.",
    icon: Cpu,
    iconColor: "text-pro-bright",
    accent: "rgba(167,139,250,0.10)",
    stat: "8 specialized agents",
    statColor: "text-pro-bright",
  },
  {
    label: "Experiment Tracker",
    tagline: "Compare every run",
    desc: "Side-by-side metric comparison, full run history, hyperparameter lineage across experiments.",
    icon: FlaskConical,
    iconColor: "text-pro-gold",
    accent: "rgba(245,158,11,0.10)",
    stat: "Unlimited history",
    statColor: "text-pro-gold",
  },
  {
    label: "Advanced Analytics",
    tagline: "Deep visualizations",
    desc: "Correlation matrices, PCA projections, pair plots, and feature importance — all auto-generated.",
    icon: BarChart3,
    iconColor: "text-pro-gold-bright",
    accent: "rgba(251,191,36,0.08)",
    stat: "7 chart types",
    statColor: "text-pro-gold-bright",
  },
  {
    label: "Observability",
    tagline: "Full telemetry",
    desc: "Structured per-agent logs, memory traces, latency breakdowns, and real-time stage monitoring.",
    icon: Activity,
    iconColor: "text-info",
    accent: "rgba(56,189,248,0.08)",
    stat: "Real-time monitoring",
    statColor: "text-info",
  },
  {
    label: "Artifacts Manager",
    tagline: "Model & report hub",
    desc: "Versioned model files, SHAP reports, Excel exports, and pipeline artifacts — all in one place.",
    icon: FolderOpen,
    iconColor: "text-success",
    accent: "rgba(52,211,153,0.08)",
    stat: "Auto-versioned",
    statColor: "text-success",
  },
];

const STEPS = [
  { n: 1, label: "Upload Dataset" },
  { n: 2, label: "Set Target" },
  { n: 3, label: "Launch" },
];

export default function FreeModePage() {
  const router = useRouter();
  const { setDataset, setActiveRunId } = useAppStore();
  const [uploadData, setUploadData] = useState<UploadResponse | null>(null);
  const [target, setTarget] = useState("");
  const [uploading, setUploading] = useState(false);
  const [launching, setLaunching] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showViz, setShowViz] = useState(false);

  const currentStep = !uploadData ? 1 : 2;

  const handleFile = useCallback(async (file: File) => {
    // Fail fast with a friendly message instead of streaming a huge file only
    // for the backend to reject it. Mirror the backend MAX_UPLOAD_MB (1024).
    const MAX_MB = 1024;
    if (file.size > MAX_MB * 1024 * 1024) {
      setError(`File is ${(file.size / 1024 / 1024).toFixed(0)} MB — the limit is ${MAX_MB} MB.`);
      return;
    }
    if (file.size === 0) {
      setError("That file is empty.");
      return;
    }
    setUploading(true);
    setError(null);
    try {
      const data = await uploadDataset(file);
      setUploadData(data);
      setDataset({
        filename: data.filename,
        path: data.path,
        columns: data.columns,
        dtypes: data.dtypes,
        n_rows: data.n_rows,
        preview: data.preview,
      });
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }, [setDataset]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const openFilePicker = () => {
    if (uploading) return;
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".csv,.tsv,.txt";
    input.onchange = (e) => {
      const f = (e.target as HTMLInputElement).files?.[0];
      if (f) handleFile(f);
    };
    input.click();
  };

  const handleLaunch = async () => {
    if (!uploadData) return;
    setLaunching(true);
    try {
      const res = await startPipeline(uploadData.path, target || undefined);
      setActiveRunId(res.run_id);
      router.push(`/free/pipeline/${res.run_id}`);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to start");
      setLaunching(false);
    }
  };

  const vizCount = uploadData?.visualizations?.length || 0;

  return (
    <>
      <Navbar />
      <main className="pt-24 pb-28 min-h-screen">
        <div className="max-w-[680px] mx-auto px-6">

          {/* ── Hero Header ────────────────────────────────────────── */}
          <motion.div
            variants={stagger}
            initial="hidden"
            animate="visible"
            className="mb-10 text-center"
          >
            <motion.div variants={fadeUp} className="flex items-center justify-center gap-2 mb-5">
              <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-success/[0.07] border border-success/20">
                <Sparkles size={10} className="text-success" />
                <span className="text-[10px] font-bold uppercase tracking-[2.5px] text-success">
                  Free Mode
                </span>
              </div>
            </motion.div>

            <motion.h1
              variants={fadeUp}
              className="text-[42px] md:text-[52px] font-bold tracking-[-0.045em] leading-[1.08] mb-4 text-text-primary"
            >
              Drop your data.{" "}
              <span className="gradient-text">Watch AI work.</span>
            </motion.h1>

            <motion.p
              variants={fadeUp}
              className="text-[15px] text-text-secondary font-light leading-relaxed max-w-[440px] mx-auto"
            >
              Upload any CSV and Axiom automatically cleans, engineers features, trains
              models, and delivers insights — zero configuration required.
            </motion.p>
          </motion.div>

          {/* ── Step Indicator ─────────────────────────────────────── */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.25, duration: 0.5 }}
            className="flex items-center justify-center mb-10"
          >
            {STEPS.map((s, i) => (
              <div key={s.n} className="flex items-center">
                <div className="flex items-center gap-2">
                  <div
                    className={`w-7 h-7 rounded-full flex items-center justify-center text-[11px] font-bold transition-all duration-500 ${
                      currentStep > s.n
                        ? "bg-success text-void shadow-sm"
                        : currentStep === s.n
                        ? "bg-accent text-void ring-[3px] ring-accent/25 ring-offset-1 ring-offset-void"
                        : "bg-transparent border border-glass-border text-text-ghost"
                    }`}
                  >
                    {currentStep > s.n ? <CheckCircle2 size={14} /> : s.n}
                  </div>
                  <span
                    className={`text-[12px] font-medium whitespace-nowrap transition-colors duration-300 ${
                      currentStep >= s.n ? "text-text-secondary" : "text-text-ghost"
                    }`}
                  >
                    {s.label}
                  </span>
                </div>
                {i < STEPS.length - 1 && (
                  <div
                    className={`free-step-connector ${currentStep > s.n ? "done" : ""}`}
                  />
                )}
              </div>
            ))}
          </motion.div>

          {/* ── Upload Zone ─────────────────────────────────────────── */}
          <motion.div
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.35, duration: 0.65 }}
          >
            <div
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={!uploadData ? openFilePicker : undefined}
              className={`free-upload-hero ${
                dragOver ? "dragging" : uploadData ? "has-file" : ""
              }`}
            >
              {/* Ambient glow on drag */}
              {dragOver && (
                <div className="absolute inset-0 bg-gradient-to-br from-accent/[0.07] to-transparent pointer-events-none rounded-3xl" />
              )}

              {uploading ? (
                <div className="flex flex-col items-center gap-5 py-16">
                  <div className="relative">
                    <div className="w-16 h-16 border-2 border-accent/20 border-t-accent rounded-full animate-spin" />
                    <div className="absolute inset-0 flex items-center justify-center">
                      <Database size={20} className="text-accent/50" />
                    </div>
                  </div>
                  <div className="text-center">
                    <p className="text-[14px] font-semibold text-text-secondary">Analyzing dataset…</p>
                    <p className="text-[12px] text-text-ghost mt-1">Profiling columns · generating charts</p>
                  </div>
                </div>
              ) : uploadData ? (
                <div className="p-6">
                  <div className="flex items-start gap-4">
                    <div className="w-11 h-11 rounded-xl bg-success/[0.08] border border-success/25 flex items-center justify-center shrink-0">
                      <Database size={20} className="text-success" strokeWidth={1.5} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-[14px] font-semibold text-text-primary truncate">{uploadData.filename}</span>
                        <span className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-success/[0.07] border border-success/20 text-[9px] font-bold text-success uppercase tracking-wider">
                          <CheckCircle2 size={9} /> Ready
                        </span>
                      </div>
                      <div className="flex items-center gap-4 mt-1.5 flex-wrap">
                        <span className="flex items-center gap-1.5 text-[12px] text-text-muted">
                          <Rows3 size={11} /> {uploadData.n_rows.toLocaleString()} rows
                        </span>
                        <span className="flex items-center gap-1.5 text-[12px] text-text-muted">
                          <Columns size={11} /> {uploadData.columns.length} columns
                        </span>
                        {vizCount > 0 && (
                          <span className="flex items-center gap-1.5 text-[12px] text-accent">
                            <BarChart3 size={11} /> {vizCount} charts
                          </span>
                        )}
                      </div>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setUploadData(null);
                        setTarget("");
                        setError(null);
                      }}
                      className="p-1.5 rounded-lg hover:bg-glass-hover transition-colors shrink-0"
                    >
                      <X size={13} className="text-text-muted" />
                    </button>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-6 py-20 px-8 select-none">
                  <div className="relative">
                    <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-accent/[0.07] to-success/[0.03] border border-glass-border flex items-center justify-center">
                      <Upload size={30} className="text-text-muted" strokeWidth={1.5} />
                    </div>
                    <div className="absolute -inset-3 rounded-[32px] border border-accent/[0.07] animate-pulse pointer-events-none" />
                  </div>
                  <div className="text-center">
                    <p className="text-[16px] text-text-secondary font-medium">
                      Drop your CSV here, or{" "}
                      <span className="text-accent cursor-pointer underline underline-offset-4">
                        browse files
                      </span>
                    </p>
                    <p className="text-[12px] text-text-ghost mt-2">
                      CSV · TSV · any tabular format · up to 300 MB
                    </p>
                  </div>
                </div>
              )}
            </div>
          </motion.div>

          {/* ── Post-upload: Config + Launch ────────────────────────── */}
          <AnimatePresence>
            {uploadData && (
              <motion.div
                initial={{ opacity: 0, y: 14 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.45, ease: [0.25, 0.4, 0, 1] }}
                className="mt-4 space-y-3"
              >
                {/* Dataset Intelligence accordion */}
                {vizCount > 0 && (
                  <div className="glass-sm overflow-hidden">
                    <button
                      onClick={() => setShowViz(!showViz)}
                      className="w-full px-5 py-3.5 flex items-center justify-between hover:bg-glass-hover transition-colors duration-200"
                    >
                      <div className="flex items-center gap-2.5">
                        <Eye size={13} className="text-accent" />
                        <span className="text-[13px] font-semibold text-text-primary">
                          Dataset Intelligence
                        </span>
                        <span className="px-2 py-0.5 rounded-full bg-accent/[0.07] text-accent text-[10px] font-bold">
                          {vizCount} charts
                        </span>
                      </div>
                      <ChevronDown
                        size={13}
                        className={`text-text-muted transition-transform duration-300 ${showViz ? "rotate-180" : ""}`}
                      />
                    </button>
                    <AnimatePresence>
                      {showViz && (
                        <motion.div
                          initial={{ height: 0 }}
                          animate={{ height: "auto" }}
                          exit={{ height: 0 }}
                          transition={{ duration: 0.28 }}
                          className="overflow-hidden border-t border-glass-border"
                        >
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 p-4">
                            {uploadData.visualizations?.map((viz, i) => (
                              <motion.div
                                key={viz.name}
                                initial={{ opacity: 0, y: 6 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: i * 0.06 }}
                                className="viz-card"
                              >
                                <img
                                  src={`data:image/png;base64,${viz.base64_png}`}
                                  alt={viz.name}
                                  className="w-full"
                                />
                                <div className="p-3 border-t border-glass-border">
                                  <div className="text-[12px] font-semibold text-text-primary">{viz.name}</div>
                                  <div className="text-[10px] text-text-muted mt-0.5">{viz.description}</div>
                                </div>
                              </motion.div>
                            ))}
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                )}

                {/* Target Variable */}
                <div className="glass-sm p-5">
                  <label className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-[2px] text-text-muted mb-3">
                    <span className="w-4 h-4 rounded-full bg-accent/[0.08] border border-accent/20 flex items-center justify-center text-[9px] text-accent font-bold">2</span>
                    Target Variable
                  </label>
                  <div className="relative">
                    <select
                      value={target}
                      onChange={(e) => setTarget(e.target.value)}
                      className="w-full appearance-none bg-void/50 border border-glass-border rounded-xl px-4 py-3 text-[14px] text-text-primary focus:outline-none focus:border-accent/40 transition-colors cursor-pointer"
                    >
                      <option value="">None — unsupervised clustering</option>
                      {uploadData.columns.map((col) => (
                        <option key={col} value={col}>{col}</option>
                      ))}
                    </select>
                    <ChevronDown
                      size={13}
                      className="absolute right-4 top-1/2 -translate-y-1/2 text-text-muted pointer-events-none"
                    />
                  </div>
                  {target && (
                    <p className="text-[11px] text-text-muted mt-2 flex items-center gap-1.5">
                      <TrendingUp size={10} className="text-accent" />
                      Axiom will auto-detect classification vs. regression from{" "}
                      <span className="text-text-secondary font-semibold">"{target}"</span>
                    </p>
                  )}
                </div>

                {/* Data Preview (compact) */}
                {uploadData.preview && uploadData.preview.length > 0 && (
                  <div className="glass-sm p-5 overflow-x-auto">
                    <label className="text-[10px] font-bold uppercase tracking-[2px] text-text-muted block mb-3">
                      Data Preview — first 3 rows
                    </label>
                    <table className="w-full text-[11px]">
                      <thead>
                        <tr>
                          {uploadData.columns.slice(0, 7).map((col) => (
                            <th
                              key={col}
                              className={`text-left py-1.5 px-2.5 font-semibold border-b border-glass-border whitespace-nowrap ${
                                col === target ? "text-accent" : "text-text-muted"
                              }`}
                            >
                              {col}
                              {col === target && (
                                <span className="ml-1 text-accent text-[8px]">▲</span>
                              )}
                            </th>
                          ))}
                          {uploadData.columns.length > 7 && (
                            <th className="text-left py-1.5 px-2.5 text-text-ghost font-medium border-b border-glass-border">
                              +{uploadData.columns.length - 7}
                            </th>
                          )}
                        </tr>
                      </thead>
                      <tbody>
                        {uploadData.preview.slice(0, 3).map((row, i) => (
                          <tr key={i} className="border-b border-glass-border/40">
                            {uploadData.columns.slice(0, 7).map((col) => (
                              <td
                                key={col}
                                className={`py-1.5 px-2.5 whitespace-nowrap ${
                                  col === target
                                    ? "text-accent/80 font-medium"
                                    : "text-text-secondary"
                                }`}
                              >
                                {String(row[col] ?? "—")}
                              </td>
                            ))}
                            {uploadData.columns.length > 7 && (
                              <td className="py-1.5 px-2.5 text-text-ghost">…</td>
                            )}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}

                {/* Launch Button */}
                <button
                  onClick={handleLaunch}
                  disabled={launching}
                  className="group w-full relative flex items-center justify-center gap-3 py-[15px] rounded-2xl overflow-hidden font-bold text-[15px] text-void disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-300 hover:shadow-xl hover:shadow-accent/20"
                  style={{
                    background: "linear-gradient(135deg, #00e5c8 0%, #00c9b0 45%, #34d399 100%)",
                  }}
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-700 pointer-events-none" />
                  {launching ? (
                    <>
                      <div className="w-4 h-4 border-2 border-void/25 border-t-void rounded-full animate-spin" />
                      Launching Autonomous Pipeline…
                    </>
                  ) : (
                    <>
                      <Brain size={17} />
                      Run Full AI Pipeline
                      <ArrowRight
                        size={15}
                        className="group-hover:translate-x-1 transition-transform duration-300"
                      />
                    </>
                  )}
                </button>

                {error && (
                  <div className="flex items-start gap-3 p-4 rounded-xl border border-destructive/20 bg-destructive/[0.04]">
                    <X size={14} className="text-destructive shrink-0 mt-0.5" />
                    <span className="text-destructive text-[13px]">{error}</span>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>

          {/* ═══════════════════════════════════════════════════════
              PRO UPGRADE SECTION
              ═══════════════════════════════════════════════════════ */}
          <motion.div
            initial={{ opacity: 0, y: 28 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7, duration: 0.8 }}
            className="mt-24"
          >
            {/* Section divider */}
            <div className="flex items-center gap-5 mb-10">
              <div className="flex-1 h-px bg-gradient-to-r from-transparent to-glass-border" />
              <div className="flex items-center gap-2 px-4 py-2 rounded-full border border-pro/20 bg-pro/[0.04]">
                <Crown size={11} className="text-pro-gold" />
                <span className="text-[10px] font-bold uppercase tracking-[2.5px] text-pro-bright">
                  Unlock Pro
                </span>
              </div>
              <div className="flex-1 h-px bg-gradient-to-l from-transparent to-glass-border" />
            </div>

            <div className="text-center mb-8">
              <h2 className="text-[26px] font-bold tracking-[-0.035em] gradient-text-pro mb-2">
                Free runs the pipeline.
                <br />
                <span className="text-text-secondary font-light">Pro lets you command it.</span>
              </h2>
            </div>

            {/* Feature cards — 2-column marketing layout */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-6">
              {PRO_FEATURES.map((feat, i) => (
                <motion.div
                  key={feat.label}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.8 + i * 0.06 }}
                  className="free-pro-card group"
                >
                  {/* Hover gradient fill */}
                  <div
                    className="absolute inset-0 rounded-[18px] opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"
                    style={{ background: `radial-gradient(ellipse 80% 80% at 20% 20%, ${feat.accent}, transparent)` }}
                  />

                  {/* Lock badge */}
                  <div className="absolute top-4 right-4 flex items-center gap-1 px-2 py-0.5 rounded-full bg-glass border border-glass-border">
                    <Lock size={8} className="text-text-ghost" />
                    <span className="text-[9px] text-text-ghost font-bold uppercase tracking-wide">Pro</span>
                  </div>

                  <div className="relative z-10">
                    <feat.icon
                      size={22}
                      className={`${feat.iconColor} mb-3`}
                      strokeWidth={1.5}
                    />
                    <div className="text-[13px] font-bold text-text-primary leading-tight mb-0.5">
                      {feat.label}
                    </div>
                    <div className={`text-[10px] font-semibold uppercase tracking-wide mb-2 ${feat.iconColor} opacity-70`}>
                      {feat.tagline}
                    </div>
                    <p className="text-[12px] text-text-muted leading-relaxed mb-3">
                      {feat.desc}
                    </p>
                    <div className="flex items-center gap-1.5">
                      <Zap size={9} className={feat.statColor} />
                      <span className={`text-[10px] font-semibold ${feat.statColor}`}>
                        {feat.stat}
                      </span>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>

            {/* Upgrade CTA banner */}
            <div className="free-upgrade-banner">
              <div className="absolute inset-0 bg-gradient-to-br from-pro/[0.05] via-transparent to-pro-gold/[0.04] pointer-events-none rounded-3xl" />
              <div className="relative z-10">
                <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-pro to-pro-muted border border-pro/30 shadow-lg shadow-pro-glow flex items-center justify-center mx-auto mb-5">
                  <Crown size={24} className="text-white" />
                </div>
                <h3 className="text-[20px] font-bold text-text-primary tracking-[-0.025em] mb-1.5">
                  Ready to go deeper?
                </h3>
                <p className="text-text-muted text-[13px] mb-6 max-w-[300px] mx-auto leading-relaxed">
                  Switch to Pro and take full control of your ML pipeline — same account, no payment required.
                </p>
                <button
                  onClick={() => router.push("/enterprise")}
                  className="btn-pro-primary mx-auto group"
                >
                  <Crown size={14} className="text-pro-gold-bright" />
                  Switch to Pro Mode
                  <ArrowRight
                    size={13}
                    className="group-hover:translate-x-1 transition-transform"
                  />
                </button>
                <p className="text-text-ghost text-[11px] mt-4">
                  No payment · Same account · Switch anytime
                </p>
              </div>
            </div>
          </motion.div>

        </div>
      </main>
    </>
  );
}
