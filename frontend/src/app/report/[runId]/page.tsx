"use client";

import { useEffect, useState, use, useCallback, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Trophy, Layers, AlertTriangle, FileText, Brain, Database,
  BarChart3, Zap, Clock, Download, Sparkles, Loader2, RefreshCw,
  CheckCircle2, ArrowRight, TrendingUp, ShieldCheck, Columns3,
  Hash, Percent, Eraser, Wrench, Split, PackageCheck, Activity,
  ChevronRight, Eye, X, ZoomIn,
} from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell,
} from "recharts";
import Navbar from "@/components/layout/Navbar";
import {
  getResults, getReport, getShapData, downloadReportPdf,
  ResultsResponse, getVisualizations, generateVisualization, VizResult,
} from "@/lib/api";
import { fadeUp, stagger } from "@/lib/animations";

// ── Reusable UI Components ──────────────────────────────────────────────────

function StatCard({
  label, value, icon: Icon, color = "text-accent", delay = 0, subtitle,
}: {
  label: string; value: string | number; icon: React.ElementType;
  color?: string; delay?: number; subtitle?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.5, ease: [0.25, 0.4, 0, 1] }}
      className="group relative overflow-hidden rounded-2xl border border-glass-border bg-white/[0.02] p-6 hover:border-glass-hover hover:bg-white/[0.035] transition-all duration-500"
    >
      <div className="absolute top-0 right-0 w-24 h-24 bg-gradient-to-bl from-white/[0.02] to-transparent rounded-bl-[60px] pointer-events-none" />
      <Icon
        size={18}
        strokeWidth={1.5}
        className={`${color} mb-4 opacity-60 group-hover:opacity-100 transition-opacity duration-300`}
      />
      <div className="text-2xl font-bold text-text-primary tracking-tight truncate">
        {value}
      </div>
      <div className="text-[10px] text-text-muted uppercase tracking-[1.8px] mt-1.5 font-semibold">
        {label}
      </div>
      {subtitle && (
        <div className="text-[11px] text-text-ghost mt-1">{subtitle}</div>
      )}
    </motion.div>
  );
}

function SectionHeader({ title, subtitle, icon: Icon, badge }: {
  title: string; subtitle?: string; icon: React.ElementType; badge?: string;
}) {
  return (
    <div className="flex items-start gap-4 mb-6">
      <div className="w-10 h-10 rounded-xl bg-accent/[0.08] border border-accent/20 flex items-center justify-center shrink-0 mt-0.5">
        <Icon size={18} className="text-accent" strokeWidth={1.5} />
      </div>
      <div className="min-w-0">
        <div className="flex items-center gap-3">
          <h2 className="text-[17px] font-bold text-text-primary tracking-tight">
            {title}
          </h2>
          {badge && (
            <span className="text-[9px] font-bold uppercase tracking-widest px-2.5 py-0.5 rounded-full bg-accent/10 text-accent">
              {badge}
            </span>
          )}
        </div>
        {subtitle && (
          <p className="text-[13px] text-text-muted mt-0.5 leading-relaxed">{subtitle}</p>
        )}
      </div>
    </div>
  );
}

function DataRow({ label, value, mono }: { label: string; value: string | number | undefined; mono?: boolean }) {
  return (
    <div className="flex justify-between items-center py-2.5 border-b border-glass-border/40 last:border-0 group/row">
      <dt className="text-[13px] text-text-muted group-hover/row:text-text-secondary transition-colors">
        {label}
      </dt>
      <dd className={`text-[13px] text-text-primary font-medium ${mono ? "font-mono" : ""}`}>
        {value ?? "—"}
      </dd>
    </div>
  );
}

function GlassCard({ children, className = "", hover = true }: {
  children: React.ReactNode; className?: string; hover?: boolean;
}) {
  return (
    <div className={`rounded-2xl border border-glass-border bg-white/[0.02] ${hover ? "hover:border-glass-hover hover:bg-white/[0.035] transition-all duration-500" : ""} ${className}`}>
      {children}
    </div>
  );
}

// ── Markdown renderer (for the Report tab narrative) ─────────────────────────

type MdBlock =
  | { type: "h1" | "h2" | "h3" | "paragraph"; text: string }
  | { type: "hr" }
  | { type: "list"; items: string[] }
  | { type: "table"; headers: string[]; rows: string[][] };

function parseMarkdown(md: string): MdBlock[] {
  const lines = md.split("\n");
  const blocks: MdBlock[] = [];
  let i = 0;

  while (i < lines.length) {
    const t = lines[i].trim();
    if (!t) { i++; continue; }

    const h1 = t.match(/^# (.+)$/);
    if (h1) { blocks.push({ type: "h1", text: h1[1] }); i++; continue; }
    const h2 = t.match(/^## (.+)$/);
    if (h2) { blocks.push({ type: "h2", text: h2[1] }); i++; continue; }
    const h3 = t.match(/^### (.+)$/);
    if (h3) { blocks.push({ type: "h3", text: h3[1] }); i++; continue; }
    if (/^-{3,}$/.test(t)) { blocks.push({ type: "hr" }); i++; continue; }

    // Parse markdown tables into structured data
    if (t.startsWith("|")) {
      const tableLines: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        tableLines.push(lines[i]);
        i++;
      }
      const parseRow = (line: string) =>
        line.trim().split("|").slice(1, -1).map((c) => c.trim());
      const isSep = (row: string[]) => row.every((c) => /^:?-+:?$/.test(c));
      const allRows = tableLines.map(parseRow);
      const headers = allRows[0] ?? [];
      const dataRows = allRows.slice(1).filter((row) => !isSep(row));
      blocks.push({ type: "table", headers, rows: dataRows });
      continue;
    }

    if (/^[-*] /.test(t)) {
      const items: string[] = [];
      while (i < lines.length && /^[-*] /.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^[-*] /, ""));
        i++;
      }
      blocks.push({ type: "list", items });
      continue;
    }

    const paraLines: string[] = [];
    while (i < lines.length) {
      const pt = lines[i].trim();
      if (!pt || /^#{1,6} /.test(pt) || pt.startsWith("|") || /^[-*] /.test(pt) || /^-{3,}$/.test(pt)) break;
      paraLines.push(pt);
      i++;
    }
    if (paraLines.length > 0) {
      blocks.push({ type: "paragraph", text: paraLines.join(" ") });
    }
  }
  return blocks;
}

function InlineText({ text }: { text: string }) {
  const tokens = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/);
  return (
    <>
      {tokens.map((tok, i) => {
        if (tok.startsWith("**") && tok.endsWith("**") && tok.length > 4)
          return <strong key={i} className="font-semibold text-text-primary">{tok.slice(2, -2)}</strong>;
        if (tok.startsWith("`") && tok.endsWith("`") && tok.length > 2)
          return <code key={i} className="font-mono text-[11px] text-accent bg-white/[0.06] px-1.5 py-0.5 rounded">{tok.slice(1, -1)}</code>;
        return <span key={i}>{tok}</span>;
      })}
    </>
  );
}

function MarkdownNarrative({ content }: { content: string }) {
  const blocks = parseMarkdown(content);
  return (
    <div className="space-y-2">
      {blocks.map((block, idx) => {
        if (block.type === "h1") return (
          <h1 key={idx} className="text-2xl font-bold tracking-tight text-text-primary mt-2 mb-4 gradient-text"><InlineText text={block.text} /></h1>
        );
        if (block.type === "h2") return (
          <h2 key={idx} className="text-lg font-semibold text-text-primary mt-10 mb-3 pt-2 flex items-center gap-2.5 border-t border-glass-border/30">
            <span className="w-1 h-5 rounded-full bg-accent inline-block" />
            <InlineText text={block.text} />
          </h2>
        );
        if (block.type === "h3") return (
          <h3 key={idx} className="text-[15px] font-semibold text-text-secondary mt-6 mb-2"><InlineText text={block.text} /></h3>
        );
        if (block.type === "hr") return <hr key={idx} className="border-glass-border my-8" />;
        if (block.type === "table") return (
          <div key={idx} className="overflow-x-auto my-6 rounded-xl border border-glass-border">
            <table className="w-full">
              <thead>
                <tr className="border-b border-glass-border bg-white/[0.025]">
                  {block.headers.map((h, ci) => (
                    <th key={ci} className="py-3 px-5 text-left text-[10px] font-semibold uppercase tracking-[1.5px] text-text-muted whitespace-nowrap">
                      <InlineText text={h} />
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {block.rows.map((row, ri) => (
                  <tr key={ri} className="border-b border-glass-border/30 last:border-0 hover:bg-white/[0.02] transition-colors">
                    {row.map((cell, ci) => (
                      <td key={ci} className={`py-3 px-5 text-[13px] ${ci === 0 ? "text-text-primary font-medium" : "text-text-secondary"}`}>
                        <InlineText text={cell} />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
        if (block.type === "list") return (
          <ul key={idx} className="space-y-2 my-4 pl-1">
            {block.items.map((item, ii) => (
              <li key={ii} className="flex items-start gap-3 text-[13px] text-text-secondary leading-relaxed">
                <ChevronRight size={12} className="text-accent shrink-0 mt-1" />
                <InlineText text={item} />
              </li>
            ))}
          </ul>
        );
        if (block.type === "paragraph") return (
          <p key={idx} className="text-[13px] text-text-secondary leading-[1.85] my-2.5"><InlineText text={block.text} /></p>
        );
        return null;
      })}
    </div>
  );
}

// ── Tab Configuration ────────────────────────────────────────────────────────

const TABS = [
  { id: "overview", label: "Overview", icon: Layers },
  { id: "models",   label: "Leaderboard", icon: Brain },
  { id: "viz",      label: "Visualizations", icon: BarChart3 },
  { id: "shap",     label: "Explainability", icon: Sparkles },
  { id: "errors",   label: "Quality Audit", icon: ShieldCheck },
  { id: "report",   label: "Full Narrative", icon: FileText },
];

const TOOLTIP_STYLE = {
  background: "#111114",
  border: "1px solid rgba(255,255,255,0.06)",
  borderRadius: "12px",
  fontSize: "12px",
  color: "#f0f0f0",
};

// ── Page ─────────────────────────────────────────────────────────────────────

export default function ReportPage({
  params,
}: {
  params: Promise<{ runId: string }>;
}) {
  const { runId } = use(params);

  const [results, setResults]     = useState<ResultsResponse | null>(null);
  const [report,  setReport]      = useState<string>("");
  const [tab,     setTab]         = useState("overview");
  const [loading, setLoading]     = useState(true);

  const [shapData,    setShapData]    = useState<Record<string, number> | null>(null);
  const [shapLoading, setShapLoading] = useState(false);
  const [shapFetched, setShapFetched] = useState(false);

  const [pdfBusy,     setPdfBusy]     = useState(false);
  const [pdfError,    setPdfError]    = useState<string | null>(null);

  const [vizs, setVizs]           = useState<VizResult[]>([]);
  const [generatingViz, setGeneratingViz] = useState<string | null>(null);
  const [lightboxViz, setLightboxViz] = useState<VizResult | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const [r, rep, v] = await Promise.all([
          getResults(runId),
          getReport(runId).catch(() => ""),
          getVisualizations(runId).catch(() => []),
        ]);
        setResults(r);
        setReport(rep);
        setVizs(v);
      } catch { /* ignored */ }
      setLoading(false);
    })();
  }, [runId]);

  useEffect(() => {
    if (tab !== "shap" || shapFetched) return;
    setShapFetched(true);
    setShapLoading(true);
    getShapData(runId)
      .then((d) => setShapData(d))
      .catch(() => setShapData({}))
      .finally(() => setShapLoading(false));
  }, [tab, shapFetched, runId]);

  const handleDownloadPdf = useCallback(async () => {
    setPdfBusy(true);
    setPdfError(null);
    try { await downloadReportPdf(runId); }
    catch (e: unknown) { setPdfError(e instanceof Error ? e.message : "PDF download failed"); }
    finally { setPdfBusy(false); }
  }, [runId]);

  const handleDownloadMd = useCallback(() => {
    if (!report) return;
    const blob = new Blob([report], { type: "text/markdown;charset=utf-8" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `axiom-report-${runId}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [report, runId]);

  // ── Derived data ──────────────────────────────────────────────────────────

  const bestModel = results?.models.find((m) => m.is_best);
  const trainedModels = results?.models.filter((m) => m.status === "trained") ?? [];

  const modelChartData = useMemo(() =>
    trainedModels
      .map((m) => ({
        name: m.name.replace(/Classifier|Regressor/g, "").trim(),
        metric: Object.values(m.metrics)[0] ?? 0,
        isBest: m.is_best,
      }))
      .sort((a, b) => b.metric - a.metric),
    [trainedModels],
  );

  const shapChartData = useMemo(() =>
    shapData
      ? Object.entries(shapData).slice(0, 15).map(([feature, value]) => ({ feature, value }))
      : [],
    [shapData],
  );

  // ── Loading / error states ────────────────────────────────────────────────

  if (loading) {
    return (
      <>
        <Navbar />
        <main className="pt-28 pb-20 min-h-screen flex items-center justify-center">
          <div className="flex flex-col items-center gap-4">
            <div className="w-10 h-10 border-2 border-accent/30 border-t-accent rounded-full animate-spin" />
            <span className="text-[13px] text-text-muted">Loading report…</span>
          </div>
        </main>
      </>
    );
  }

  if (!results || results.status !== "completed") {
    return (
      <>
        <Navbar />
        <main className="pt-28 pb-20 min-h-screen flex items-center justify-center">
          <GlassCard className="p-12 text-center max-w-md">
            <Activity size={32} className="text-text-ghost mx-auto mb-4" strokeWidth={1.5} />
            <p className="text-text-secondary text-[15px] font-medium">Results not available yet</p>
            <p className="text-[12px] text-text-muted mt-2">The pipeline may still be running.</p>
          </GlassCard>
        </main>
      </>
    );
  }

  // ── Feature engineering timeline data ─────────────────────────────────────

  const feTimeline = [
    {
      title: "Input Features",
      value: results.features?.before ?? "—",
      icon: Columns3,
      desc: "Raw columns from the dataset",
    },
    {
      title: "Correlation Pruning",
      value: `${(results.features?.before ?? 0) - (results.features?.after ?? 0)} removed`,
      icon: Eraser,
      desc: "Highly correlated features dropped",
    },
    {
      title: "Mutual Information Scoring",
      value: `Top ${results.features?.after ?? "—"} selected`,
      icon: TrendingUp,
      desc: "Features ranked by predictive power",
    },
    {
      title: "Final Feature Set",
      value: results.features?.after ?? "—",
      icon: CheckCircle2,
      desc: `${results.features?.selected?.length ?? 0} features sent to model training`,
    },
  ];

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <>
      <Navbar />

      <main className="pt-28 pb-20 min-h-screen">
        <div className="max-w-[1280px] mx-auto px-6 md:px-10">

          {/* ── Report Header ──────────────────────────────────────────────── */}
          <motion.div
            variants={stagger}
            initial="hidden"
            animate="visible"
            className="mb-10"
          >
            <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-6">
              <div>
                <motion.div variants={fadeUp} className="flex items-center gap-2.5 mb-3">
                  <span className="badge badge-success"><CheckCircle2 size={10} /> Complete</span>
                  <span className="text-[11px] font-mono text-text-ghost">{runId}</span>
                </motion.div>
                <motion.h1
                  variants={fadeUp}
                  className="text-3xl md:text-[40px] font-bold tracking-[-0.03em] gradient-text-hero leading-tight"
                >
                  Execution Report
                </motion.h1>
                <motion.p variants={fadeUp} className="text-[14px] text-text-muted mt-2 max-w-lg">
                  Autonomous end-to-end machine learning — every stage profiled, every model evaluated, every artifact catalogued.
                </motion.p>
              </div>
              <motion.div variants={fadeUp} className="flex items-center gap-3 shrink-0">
                <button
                  onClick={handleDownloadMd}
                  disabled={!report}
                  className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-[12px] font-semibold bg-white/[0.04] border border-glass-border text-text-secondary hover:bg-white/[0.08] transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <FileText size={13} strokeWidth={2} />
                  Markdown
                </button>
                <button
                  onClick={handleDownloadPdf}
                  disabled={pdfBusy}
                  className="group flex items-center gap-2 px-5 py-2.5 rounded-xl text-[13px] font-semibold bg-accent text-void hover:bg-accent-bright transition-all duration-300 shadow-lg shadow-accent/20 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {pdfBusy ? (
                    <div className="w-3.5 h-3.5 border-2 border-void/30 border-t-void rounded-full animate-spin" />
                  ) : (
                    <Download size={14} strokeWidth={2} />
                  )}
                  {pdfBusy ? "Generating…" : "Download PDF"}
                </button>
              </motion.div>
            </div>
          </motion.div>
          {pdfError && (
            <div className="mb-6 flex items-center gap-2 p-3 rounded-xl border border-destructive/20 bg-destructive/[0.04] text-destructive text-[12px]">
              <AlertTriangle size={13} /> {pdfError}
            </div>
          )}

          {/* ── Hero Card: Champion Model ───────────────────────────────────── */}
          {bestModel && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="mb-8 rounded-2xl border border-accent/20 bg-gradient-to-br from-accent/[0.06] via-white/[0.015] to-transparent p-8 relative overflow-hidden"
            >
              <div className="absolute top-0 right-0 w-60 h-60 bg-gradient-to-bl from-accent/[0.06] to-transparent rounded-bl-full pointer-events-none" />
              <div className="flex flex-col md:flex-row md:items-center gap-6 relative z-10">
                <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-warning/20 to-warning/5 border border-warning/20 flex items-center justify-center shrink-0">
                  <Trophy size={28} className="text-warning" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="text-[10px] font-bold uppercase tracking-[2.5px] text-accent mb-1">
                    Champion Model
                  </div>
                  <div className="text-2xl md:text-3xl font-bold text-text-primary tracking-tight">
                    {bestModel.name}
                  </div>
                  <div className="flex items-center gap-4 mt-2 flex-wrap">
                    <span className="text-[14px] font-mono font-semibold text-accent">
                      {results.best_metric_name}: {results.best_metric_value?.toFixed(4)}
                    </span>
                    <span className="text-[12px] text-text-muted">
                      {results.problem_type?.replace("_", " ")} · Target: <strong className="text-text-secondary">{results.target_column}</strong>
                    </span>
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-6 text-center shrink-0">
                  {[
                    { label: "Models", value: trainedModels.length },
                    { label: "Rows", value: results.dataset.rows?.toLocaleString() },
                    { label: "Retries", value: results.retry_count },
                  ].map((kpi) => (
                    <div key={kpi.label}>
                      <div className="text-xl font-bold text-text-primary">{kpi.value}</div>
                      <div className="text-[9px] text-text-muted uppercase tracking-[1.5px] font-semibold">{kpi.label}</div>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}

          {/* ── Tab Bar ────────────────────────────────────────────────────── */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="flex gap-1 mb-10 p-1.5 rounded-2xl border border-glass-border bg-white/[0.015] w-fit overflow-x-auto"
          >
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-[11px] font-semibold uppercase tracking-wider transition-all duration-300 whitespace-nowrap ${
                  tab === t.id
                    ? "bg-white/[0.08] text-text-primary shadow-sm"
                    : "text-text-muted hover:text-text-secondary hover:bg-white/[0.03]"
                }`}
              >
                <t.icon size={13} strokeWidth={1.5} />
                {t.label}
              </button>
            ))}
          </motion.div>

          {/* ── Tab Content ────────────────────────────────────────────────── */}
          <AnimatePresence mode="wait">
            <motion.div
              key={tab}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.35, ease: [0.25, 0.4, 0, 1] }}
            >

              {/* ════════════════ OVERVIEW ════════════════ */}
              {tab === "overview" && (
                <div className="space-y-8">

                  {/* Executive Summary KPI Strip */}
                  <div>
                    <SectionHeader
                      icon={Sparkles}
                      title="Executive Summary"
                      subtitle="Key performance indicators from the completed pipeline run."
                    />
                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                      <StatCard label="Problem Type" value={results.problem_type || "—"} icon={Layers} color="text-primary" delay={0} />
                      <StatCard label="Best Model" value={bestModel?.name || "—"} icon={Trophy} color="text-warning" delay={0.06} />
                      <StatCard label={results.best_metric_name || "Metric"} value={results.best_metric_value?.toFixed(4) || "—"} icon={TrendingUp} color="text-success" delay={0.12} />
                      <StatCard label="Models Trained" value={trainedModels.length} icon={Brain} color="text-info" delay={0.18} />
                    </div>
                  </div>

                  {/* Dataset & Preprocessing */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
                    <GlassCard className="p-7">
                      <div className="flex items-center gap-2.5 mb-5">
                        <Database size={15} className="text-accent" strokeWidth={1.5} />
                        <h3 className="text-[10px] font-bold uppercase tracking-[2px] text-text-muted">
                          Dataset Summary
                        </h3>
                      </div>
                      <dl className="space-y-0.5">
                        <DataRow label="Total Rows" value={results.dataset.rows?.toLocaleString()} mono />
                        <DataRow label="Total Columns" value={results.dataset.columns} mono />
                        <DataRow label="Data Quality Score" value={results.dataset.quality_score?.toFixed(4)} mono />
                        <DataRow label="Target Column" value={results.target_column || "None (clustering)"} />
                        <DataRow label="Problem Type" value={results.problem_type?.replace("_", " ")} />
                      </dl>
                    </GlassCard>

                    <GlassCard className="p-7">
                      <div className="flex items-center gap-2.5 mb-5">
                        <Eraser size={15} className="text-accent" strokeWidth={1.5} />
                        <h3 className="text-[10px] font-bold uppercase tracking-[2px] text-text-muted">
                          Preprocessing Metrics
                        </h3>
                      </div>
                      <dl className="space-y-0.5">
                        <DataRow label="Rows Before" value={results.preprocessing.rows_before?.toLocaleString()} mono />
                        <DataRow label="Rows After" value={results.preprocessing.rows_after?.toLocaleString()} mono />
                        <DataRow label="Duplicates Removed" value={results.preprocessing.duplicates_removed} mono />
                        <DataRow label="Quality After" value={results.preprocessing.quality_score?.toFixed(4)} mono />
                        <DataRow label="Features Before → After" value={`${results.features?.before ?? "—"} → ${results.features?.after ?? "—"}`} />
                      </dl>
                    </GlassCard>
                  </div>

                  {/* Feature Engineering Timeline */}
                  {results.features && (
                    <div>
                      <SectionHeader
                        icon={Wrench}
                        title="Feature Engineering"
                        subtitle="Automated feature selection and dimensionality reduction pipeline."
                      />
                      <GlassCard className="p-7" hover={false}>
                        <div className="relative">
                          {feTimeline.map((step, i) => (
                            <div key={step.title} className="flex items-start gap-5 relative pb-8 last:pb-0">
                              {/* Vertical connector line */}
                              {i < feTimeline.length - 1 && (
                                <div className="absolute left-[19px] top-10 w-px h-[calc(100%-28px)] bg-gradient-to-b from-accent/30 to-accent/5" />
                              )}
                              <div className={`w-10 h-10 rounded-xl shrink-0 flex items-center justify-center border ${
                                i === feTimeline.length - 1
                                  ? "bg-accent/10 border-accent/30"
                                  : "bg-white/[0.03] border-glass-border"
                              }`}>
                                <step.icon size={16} className={i === feTimeline.length - 1 ? "text-accent" : "text-text-muted"} strokeWidth={1.5} />
                              </div>
                              <div className="pt-1">
                                <div className="flex items-center gap-3">
                                  <span className="text-[13px] font-semibold text-text-primary">{step.title}</span>
                                  <span className="text-[12px] font-mono text-accent">{step.value}</span>
                                </div>
                                <div className="text-[12px] text-text-muted mt-0.5">{step.desc}</div>
                              </div>
                            </div>
                          ))}
                        </div>

                        {/* Feature chips */}
                        {results.features.selected?.length > 0 && (
                          <div className="mt-6 pt-6 border-t border-glass-border/40">
                            <div className="text-[10px] font-bold uppercase tracking-[2px] text-text-ghost mb-3">
                              Selected Features
                            </div>
                            <div className="flex flex-wrap gap-1.5">
                              {results.features.selected.slice(0, 16).map((f) => (
                                <span
                                  key={f}
                                  className="font-mono text-[11px] text-accent bg-accent/[0.06] border border-accent/20 px-2.5 py-1 rounded-lg hover:bg-accent/[0.12] transition-colors"
                                >
                                  {f}
                                </span>
                              ))}
                              {results.features.selected.length > 16 && (
                                <span className="text-[11px] text-text-muted px-2 py-1">
                                  +{results.features.selected.length - 16} more
                                </span>
                              )}
                            </div>
                          </div>
                        )}
                      </GlassCard>
                    </div>
                  )}
                </div>
              )}

              {/* ════════════════ MODEL LEADERBOARD ════════════════ */}
              {tab === "models" && (
                <div className="space-y-6">
                  <SectionHeader
                    icon={Brain}
                    title="Model Leaderboard"
                    subtitle={`${trainedModels.length} models trained · champion highlighted below.`}
                    badge={bestModel?.name}
                  />

                  {/* Performance bar chart */}
                  <GlassCard className="p-8" hover={false}>
                    <div className="text-[10px] font-bold uppercase tracking-[2px] text-text-ghost mb-6">
                      Performance Comparison
                    </div>
                    <div className="h-[280px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={modelChartData}
                          layout="vertical"
                          margin={{ left: 40, right: 30, top: 5, bottom: 5 }}
                        >
                          <XAxis type="number" domain={[0, 1]} tick={{ fill: "#505050", fontSize: 11 }} axisLine={false} tickLine={false} />
                          <YAxis type="category" dataKey="name" tick={{ fill: "#8a8a8a", fontSize: 12 }} axisLine={false} tickLine={false} width={130} />
                          <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v) => [typeof v === "number" ? v.toFixed(4) : v, "Score"]} />
                          <Bar dataKey="metric" radius={[0, 6, 6, 0]} barSize={18}>
                            {modelChartData.map((entry, i) => (
                              <Cell key={i} fill={entry.isBest ? "#00e5c8" : "rgba(0,229,200,0.2)"} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </GlassCard>

                  {/* Model leaderboard table */}
                  <GlassCard className="overflow-hidden" hover={false}>
                    <div className="overflow-x-auto">
                      <table className="w-full">
                        <thead>
                          <tr className="border-b border-glass-border bg-white/[0.02]">
                            {["Rank", "Model", "Status", "Metrics", "Time"].map((h) => (
                              <th key={h} className={`py-3.5 px-5 text-[10px] font-bold uppercase tracking-[1.5px] text-text-ghost ${h === "Time" || h === "Rank" ? "text-center" : "text-left"}`}>
                                {h}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {[...results.models]
                            .sort((a, b) => {
                              const va = Object.values(a.metrics)[0] ?? 0;
                              const vb = Object.values(b.metrics)[0] ?? 0;
                              return vb - va;
                            })
                            .map((m, i) => (
                            <tr
                              key={m.name}
                              className={`border-b border-glass-border/40 last:border-0 transition-colors ${
                                m.is_best
                                  ? "bg-accent/[0.04] hover:bg-accent/[0.07]"
                                  : "hover:bg-white/[0.02]"
                              }`}
                            >
                              <td className="py-4 px-5 text-center">
                                <span className={`inline-flex items-center justify-center w-7 h-7 rounded-lg text-[11px] font-bold ${
                                  i === 0 ? "bg-warning/10 text-warning" :
                                  i === 1 ? "bg-text-muted/10 text-text-secondary" :
                                  i === 2 ? "bg-warning/5 text-warning/60" :
                                  "bg-white/[0.03] text-text-ghost"
                                }`}>
                                  {i + 1}
                                </span>
                              </td>
                              <td className="py-4 px-5">
                                <div className="flex items-center gap-2.5">
                                  {m.is_best && <Trophy size={14} className="text-warning shrink-0" />}
                                  <span className={`text-[13px] font-medium ${m.is_best ? "text-text-primary" : "text-text-secondary"}`}>
                                    {m.name}
                                  </span>
                                  {m.is_best && (
                                    <span className="text-[8px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-full bg-accent/10 text-accent">
                                      Winner
                                    </span>
                                  )}
                                </div>
                              </td>
                              <td className="py-4 px-5">
                                <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wider ${
                                  m.status === "trained" ? "bg-success/10 text-success" : "bg-destructive/10 text-destructive"
                                }`}>
                                  {m.status === "trained" ? <Zap size={10} /> : <AlertTriangle size={10} />}
                                  {m.status}
                                </span>
                              </td>
                              <td className="py-4 px-5">
                                <div className="flex items-center gap-3">
                                  {Object.entries(m.metrics ?? {}).map(([k, v]) => (
                                    <div key={k} className="text-[12px]">
                                      <span className="text-text-ghost uppercase text-[9px] tracking-wider">{k}</span>
                                      <span className="ml-1.5 font-mono font-semibold text-text-primary">{v.toFixed(4)}</span>
                                    </div>
                                  ))}
                                </div>
                              </td>
                              <td className="py-4 px-5 text-center">
                                <span className="inline-flex items-center gap-1 text-[12px] text-text-muted font-mono">
                                  <Clock size={11} />
                                  {m.time_s?.toFixed(2) ?? "—"}s
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </GlassCard>
                </div>
              )}

              {/* ════════════════ VISUALIZATIONS ════════════════ */}
              {tab === "viz" && (
                <div className="space-y-6">
                  <SectionHeader
                    icon={BarChart3}
                    title="Visualization Studio"
                    subtitle="Automatic high-resolution charts generated with Matplotlib and Seaborn."
                  />

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                    {[
                      { id: "missing_values", name: "Missing Values Heatmap", desc: "Heatmap of missing values across all columns", category: "basic" },
                      { id: "correlation", name: "Feature Correlation Matrix", desc: "Pearson coefficients showing feature relations", category: "basic" },
                      { id: "distributions", name: "Feature Distributions", desc: "Univariate distribution histograms", category: "basic" },
                      { id: "boxplots", name: "Outlier Box Plots", desc: "Box plots after standardization", category: "basic" },
                      { id: "dtype_distribution", name: "Data Type Distribution", desc: "Features by Pandas data type category", category: "basic" },
                      ...(results?.target_column ? [{ id: "target_distribution", name: "Target Distribution", desc: `Distribution of "${results.target_column}"`, category: "basic" }] : []),
                      { id: "pairplot", name: "Pair Plot", desc: "Pairwise relation grid with joint distributions", category: "advanced" },
                      { id: "pca", name: "PCA 2D Projection", desc: "Dimensionality reduction to 2 principal components", category: "advanced" },
                    ].map((chart) => {
                      const existing = vizs.find((v) => v.type === chart.id || v.name.toLowerCase().includes(chart.name.toLowerCase()));
                      const isGenerating = generatingViz === chart.id;

                      return (
                        <GlassCard key={chart.id} className="flex flex-col justify-between h-full overflow-hidden group">
                          {existing ? (
                            <>
                              <div className="relative aspect-[3/2] w-full bg-void/30 flex items-center justify-center overflow-hidden">
                                <img
                                  src={`data:image/png;base64,${existing.base64_png}`}
                                  alt={existing.name}
                                  className="w-full h-full object-contain group-hover:scale-[1.01] transition-transform duration-300"
                                />
                                <div className="absolute inset-0 bg-void/60 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center gap-3">
                                  <button
                                    onClick={() => setLightboxViz(existing)}
                                    className="p-2.5 rounded-full bg-white/10 hover:bg-white/20 border border-white/20 text-white transition-all shadow-md"
                                  >
                                    <ZoomIn size={16} />
                                  </button>
                                </div>
                              </div>
                              <div className="p-4 border-t border-glass-border mt-auto">
                                <div className="flex items-center justify-between">
                                  <span className="text-[13px] font-semibold text-text-primary">{existing.name}</span>
                                  <span className={`text-[9px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full ${
                                    existing.category === "advanced" ? "bg-pro-gold/10 text-pro-gold-bright" : "bg-accent/10 text-accent"
                                  }`}>
                                    {existing.category}
                                  </span>
                                </div>
                                <div className="text-[11px] text-text-muted mt-1">{existing.description}</div>
                              </div>
                            </>
                          ) : (
                            <div className="p-10 flex flex-col items-center justify-center text-center min-h-[250px] bg-white/[0.005]">
                              <BarChart3 size={32} className="text-text-ghost mb-3" strokeWidth={1.25} />
                              <div className="text-[14px] font-semibold text-text-secondary">{chart.name}</div>
                              <div className="text-[11px] text-text-muted mt-1.5 max-w-[280px]">{chart.desc}</div>
                              <span className={`text-[9px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full mt-2.5 ${
                                chart.category === "advanced" ? "bg-pro-gold/10 text-pro-gold-bright" : "bg-accent/10 text-accent"
                              }`}>
                                {chart.category}
                              </span>
                              <button
                                onClick={async () => {
                                  setGeneratingViz(chart.id);
                                  try {
                                    const newViz = await generateVisualization(runId, chart.id);
                                    setVizs((prev) => [...prev, newViz]);
                                  } catch (e) {
                                    console.error("Viz generation failed", e);
                                  } finally {
                                    setGeneratingViz(null);
                                  }
                                }}
                                disabled={!!generatingViz}
                                className="mt-6 flex items-center gap-1.5 px-4.5 py-2.5 rounded-xl text-[11px] font-semibold bg-accent text-void hover:bg-accent-bright transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
                              >
                                {isGenerating ? (
                                  <><Loader2 size={12} className="animate-spin" /> Rendering…</>
                                ) : (
                                  <><RefreshCw size={12} /> Generate Chart</>
                                )}
                              </button>
                            </div>
                          )}
                        </GlassCard>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* ════════════════ EXPLAINABILITY (SHAP) ════════════════ */}
              {tab === "shap" && (
                <div className="space-y-6">
                  <SectionHeader
                    icon={Sparkles}
                    title="Explainability — SHAP"
                    subtitle="SHapley Additive exPlanations quantify each feature's average contribution to model predictions."
                  />

                  {shapLoading ? (
                    <GlassCard className="p-16 flex items-center justify-center"><Loader2 size={24} className="text-accent animate-spin" /></GlassCard>
                  ) : shapData && Object.keys(shapData).length > 0 ? (
                    <>
                      <GlassCard className="p-8" hover={false}>
                        <div className="text-[10px] font-bold uppercase tracking-[2px] text-text-ghost mb-6">
                          Top {shapChartData.length} Features by Mean |SHAP Value|
                        </div>
                        <div style={{ height: Math.max(260, shapChartData.length * 36 + 40) }}>
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={shapChartData} layout="vertical" margin={{ left: 20, right: 60, top: 5, bottom: 5 }}>
                              <XAxis type="number" tick={{ fill: "#505050", fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={(v: number) => v.toFixed(3)} />
                              <YAxis type="category" dataKey="feature" tick={{ fill: "#8a8a8a", fontSize: 12 }} axisLine={false} tickLine={false} width={160} />
                              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v) => [typeof v === "number" ? v.toFixed(6) : v, "Mean |SHAP|"]} />
                              <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={20}>
                                {shapChartData.map((_, i) => {
                                  const alpha = Math.max(0.25, 1 - i * (0.7 / shapChartData.length));
                                  return <Cell key={i} fill={`rgba(0, 229, 200, ${alpha.toFixed(2)})`} />;
                                })}
                              </Bar>
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      </GlassCard>

                      <GlassCard className="overflow-hidden" hover={false}>
                        <div className="overflow-x-auto">
                          <table className="w-full">
                            <thead>
                              <tr className="border-b border-glass-border bg-white/[0.02]">
                                <th className="py-3.5 px-5 text-center text-[10px] font-bold uppercase tracking-[1.5px] text-text-ghost w-16">Rank</th>
                                <th className="py-3.5 px-5 text-left text-[10px] font-bold uppercase tracking-[1.5px] text-text-ghost">Feature</th>
                                <th className="py-3.5 px-5 text-right text-[10px] font-bold uppercase tracking-[1.5px] text-text-ghost">Mean |SHAP Value|</th>
                              </tr>
                            </thead>
                            <tbody>
                              {shapChartData.map((item, i) => (
                                <tr key={item.feature} className="border-b border-glass-border/40 last:border-0 hover:bg-white/[0.02] transition-colors">
                                  <td className="py-3.5 px-5 text-center">
                                    <span className={`inline-flex items-center justify-center w-6 h-6 rounded-md text-[10px] font-bold ${
                                      i < 3 ? "bg-accent/10 text-accent" : "bg-white/[0.03] text-text-ghost"
                                    }`}>{i + 1}</span>
                                  </td>
                                  <td className="py-3.5 px-5 text-[13px] font-medium text-text-primary">{item.feature}</td>
                                  <td className="py-3.5 px-5 text-right font-mono text-[13px] text-accent font-semibold">{item.value.toFixed(6)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </GlassCard>
                    </>
                  ) : (
                    <GlassCard className="p-16 text-center">
                      <Sparkles size={28} className="text-text-ghost mx-auto mb-4" strokeWidth={1.5} />
                      <p className="text-text-secondary text-[14px] font-medium">SHAP data not available</p>
                      <p className="text-[12px] text-text-muted mt-2 max-w-sm mx-auto">
                        SHAP explanations are generated when a tree or linear model with numeric features is present.
                      </p>
                    </GlassCard>
                  )}
                </div>
              )}

              {/* ════════════════ QUALITY AUDIT ════════════════ */}
              {tab === "errors" && (
                <div className="space-y-6">
                  <SectionHeader
                    icon={ShieldCheck}
                    title="Quality Audit"
                    subtitle="Automated findings from the Error Detection agent — overfitting, leakage, imbalance, and data quality flags."
                  />

                  {results.errors.length === 0 ? (
                    <GlassCard className="p-14 text-center">
                      <div className="w-14 h-14 rounded-2xl bg-success/[0.08] border border-success/20 flex items-center justify-center mx-auto mb-4">
                        <CheckCircle2 size={24} className="text-success" strokeWidth={1.5} />
                      </div>
                      <p className="text-text-primary text-[15px] font-semibold">Clean Run</p>
                      <p className="text-text-muted text-[13px] mt-1">No issues detected — pipeline ran cleanly.</p>
                    </GlassCard>
                  ) : (
                    <div className="space-y-3">
                      {results.errors.map((err, i) => (
                        <motion.div
                          key={i}
                          initial={{ opacity: 0, y: 8 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: i * 0.05 }}
                        >
                          <GlassCard className={`p-6 border-l-2 ${
                            err.severity === "critical" ? "border-l-destructive"
                            : err.severity === "warning" ? "border-l-warning"
                            : "border-l-accent"
                          }`} hover={false}>
                            <div className="flex items-start gap-3">
                              <AlertTriangle
                                size={15}
                                strokeWidth={1.5}
                                className={`mt-0.5 shrink-0 ${
                                  err.severity === "critical" ? "text-destructive"
                                  : err.severity === "warning" ? "text-warning"
                                  : "text-accent"
                                }`}
                              />
                              <div className="min-w-0">
                                <div className="flex items-center gap-2.5 mb-1.5">
                                  <span className="text-[13px] font-semibold text-text-primary capitalize">{err.type}</span>
                                  <span className={`text-[8px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-full ${
                                    err.severity === "critical" ? "bg-destructive/10 text-destructive"
                                    : err.severity === "warning" ? "bg-warning/10 text-warning"
                                    : "bg-accent/10 text-accent"
                                  }`}>{err.severity}</span>
                                </div>
                                <p className="text-[12px] text-text-muted leading-relaxed">{err.cause}</p>
                                {err.fix && (
                                  <p className="text-[12px] text-accent mt-2 flex items-center gap-1.5">
                                    <ArrowRight size={11} /> {err.fix}
                                  </p>
                                )}
                              </div>
                            </div>
                          </GlassCard>
                        </motion.div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* ════════════════ FULL NARRATIVE ════════════════ */}
              {tab === "report" && (
                <div className="space-y-5">
                  <SectionHeader
                    icon={FileText}
                    title="Pipeline Narrative"
                    subtitle="Full auto-generated report compiled by Axiom agents."
                  />

                  {report ? (
                    <GlassCard className="p-8 md:p-12" hover={false}>
                      <MarkdownNarrative content={report} />
                    </GlassCard>
                  ) : (
                    <GlassCard className="p-14 text-center">
                      <FileText size={28} className="text-text-ghost mx-auto mb-4" strokeWidth={1.5} />
                      <p className="text-text-muted">No report narrative available for this run.</p>
                    </GlassCard>
                  )}
                </div>
              )}

            </motion.div>
          </AnimatePresence>

          {/* ── Report Footer ──────────────────────────────────────────────── */}
          <div className="mt-16 pt-8 border-t border-glass-border/40 flex items-center justify-between text-[11px] text-text-ghost">
            <span>Generated by <strong className="text-text-muted">Axiom</strong> — Autonomous Data Scientist</span>
            <span className="font-mono">{runId}</span>
          </div>

        </div>
      </main>

      {/* ── Visualization Lightbox ─────────────────────────────────────────── */}
      <AnimatePresence>
        {lightboxViz && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-void/90 backdrop-blur-md"
            onClick={() => setLightboxViz(null)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="relative max-w-[1000px] w-full rounded-2xl border border-glass-border bg-void overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between px-5 py-4 border-b border-glass-border">
                <div>
                  <h3 className="text-[14px] font-semibold text-text-primary">{lightboxViz.name}</h3>
                  <p className="text-[11px] text-text-muted mt-0.5">{lightboxViz.description}</p>
                </div>
                <button
                  onClick={() => setLightboxViz(null)}
                  className="p-1.5 rounded-lg hover:bg-glass-hover text-text-ghost hover:text-text-secondary transition-colors"
                >
                  <X size={16} />
                </button>
              </div>
              <div className="p-6 bg-white/[0.01] flex items-center justify-center max-h-[70vh]">
                <img
                  src={`data:image/png;base64,${lightboxViz.base64_png}`}
                  alt={lightboxViz.name}
                  className="max-h-[60vh] object-contain rounded-lg"
                />
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
