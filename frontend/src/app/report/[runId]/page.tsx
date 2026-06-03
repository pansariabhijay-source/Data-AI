"use client";

import { useEffect, useState, use, useCallback } from "react";
import { motion } from "framer-motion";
import {
  Trophy, Layers, AlertTriangle, FileText, Brain,
  BarChart3, Zap, Clock, Download, Sparkles, Loader2, RefreshCw,
} from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell,
} from "recharts";
import Navbar from "@/components/layout/Navbar";
import { getResults, getReport, getShapData, downloadReportPdf, ResultsResponse, getVisualizations, generateVisualization, VizResult } from "@/lib/api";
import { fadeUp, stagger } from "@/lib/animations";

// ── Markdown renderer ────────────────────────────────────────────────────────

type MdBlock =
  | { type: "h1" | "h2" | "h3" | "paragraph"; text: string }
  | { type: "hr" }
  | { type: "list"; items: string[] }
  | { type: "table"; lines: string[] };

function parseBlocks(md: string): MdBlock[] {
  const lines = md.split("\n");
  const blocks: MdBlock[] = [];
  let i = 0;

  while (i < lines.length) {
    const raw = lines[i];
    const t = raw.trim();

    if (!t) { i++; continue; }

    const h1 = t.match(/^# (.+)$/);
    if (h1) { blocks.push({ type: "h1", text: h1[1] }); i++; continue; }

    const h2 = t.match(/^## (.+)$/);
    if (h2) { blocks.push({ type: "h2", text: h2[1] }); i++; continue; }

    const h3 = t.match(/^### (.+)$/);
    if (h3) { blocks.push({ type: "h3", text: h3[1] }); i++; continue; }

    if (/^-{3,}$/.test(t)) { blocks.push({ type: "hr" }); i++; continue; }

    // Table: block of lines starting with |
    if (t.startsWith("|")) {
      const tableLines: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        tableLines.push(lines[i]);
        i++;
      }
      blocks.push({ type: "table", lines: tableLines });
      continue;
    }

    // List: block of lines starting with - or *
    if (/^[-*] /.test(t)) {
      const items: string[] = [];
      while (i < lines.length && /^[-*] /.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^[-*] /, ""));
        i++;
      }
      blocks.push({ type: "list", items });
      continue;
    }

    // Paragraph: accumulate until blank line or block-level marker
    const paraLines: string[] = [];
    while (i < lines.length) {
      const pt = lines[i].trim();
      if (
        !pt ||
        /^#{1,6} /.test(pt) ||
        pt.startsWith("|") ||
        /^[-*] /.test(pt) ||
        /^-{3,}$/.test(pt)
      ) break;
      paraLines.push(pt);
      i++;
    }
    if (paraLines.length > 0) {
      blocks.push({ type: "paragraph", text: paraLines.join(" ") });
    }
  }

  return blocks;
}

function Inline({ text }: { text: string }) {
  const tokens = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/);
  return (
    <>
      {tokens.map((tok, i) => {
        if (tok.startsWith("**") && tok.endsWith("**") && tok.length > 4)
          return <strong key={i} className="font-semibold text-text-primary">{tok.slice(2, -2)}</strong>;
        if (tok.startsWith("`") && tok.endsWith("`") && tok.length > 2)
          return (
            <code key={i} className="font-mono text-[11px] text-accent bg-white/[0.06] px-1.5 py-0.5 rounded">
              {tok.slice(1, -1)}
            </code>
          );
        return <span key={i}>{tok}</span>;
      })}
    </>
  );
}

function MdTable({ lines }: { lines: string[] }) {
  const parseRow = (line: string) =>
    line.trim().split("|").slice(1, -1).map((c) => c.trim());

  const isSep = (row: string[]) => row.every((c) => /^:?-+:?$/.test(c));

  const allRows = lines.map(parseRow);
  const headers = allRows[0] ?? [];
  const dataRows = allRows.slice(1).filter((row) => !isSep(row));

  return (
    <div className="overflow-x-auto my-5 rounded-2xl border border-glass-border">
      <table className="w-full">
        <thead>
          <tr className="border-b border-glass-border bg-white/[0.02]">
            {headers.map((h, ci) => (
              <th
                key={ci}
                className="py-2.5 px-4 text-left text-[10px] font-semibold uppercase tracking-[1.5px] text-text-muted whitespace-nowrap"
              >
                <Inline text={h} />
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {dataRows.map((row, ri) => (
            <tr
              key={ri}
              className="border-b border-glass-border/40 last:border-0 hover:bg-white/[0.02] transition-colors"
            >
              {row.map((cell, ci) => (
                <td
                  key={ci}
                  className={`py-2.5 px-4 text-[13px] ${
                    ci === 0
                      ? "text-text-primary font-medium"
                      : "text-text-secondary"
                  }`}
                >
                  <Inline text={cell} />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MdContent({ content }: { content: string }) {
  const blocks = parseBlocks(content);

  return (
    <div className="space-y-1">
      {blocks.map((block, idx) => {
        if (block.type === "h1")
          return (
            <h1 key={idx} className="text-2xl font-bold tracking-tight text-text-primary mt-2 mb-4 gradient-text">
              <Inline text={block.text} />
            </h1>
          );
        if (block.type === "h2")
          return (
            <h2 key={idx} className="text-lg font-semibold text-text-primary mt-8 mb-3 pt-1 flex items-center gap-2">
              <span className="w-1 h-4 rounded-full bg-accent inline-block" />
              <Inline text={block.text} />
            </h2>
          );
        if (block.type === "h3")
          return (
            <h3 key={idx} className="text-[15px] font-semibold text-text-secondary mt-5 mb-2">
              <Inline text={block.text} />
            </h3>
          );
        if (block.type === "hr")
          return <hr key={idx} className="border-glass-border my-6" />;
        if (block.type === "table")
          return <MdTable key={idx} lines={block.lines} />;
        if (block.type === "list")
          return (
            <ul key={idx} className="space-y-1.5 my-3 pl-2">
              {block.items.map((item, ii) => (
                <li key={ii} className="flex items-start gap-2.5 text-[13px] text-text-secondary leading-relaxed">
                  <span className="mt-2 w-1 h-1 rounded-full bg-accent shrink-0" />
                  <Inline text={item} />
                </li>
              ))}
            </ul>
          );
        if (block.type === "paragraph")
          return (
            <p key={idx} className="text-[13px] text-text-secondary leading-relaxed my-2">
              <Inline text={block.text} />
            </p>
          );
        return null;
      })}
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────

const TABS = [
  { id: "overview", label: "Overview", icon: Layers },
  { id: "models",   label: "Models",   icon: Brain },
  { id: "viz",      label: "Visualizations", icon: BarChart3 },
  { id: "shap",     label: "SHAP",     icon: Sparkles },
  { id: "errors",   label: "Issues",   icon: AlertTriangle },
  { id: "report",   label: "Report",   icon: FileText },
];

const SHAP_TOOLTIP_STYLE = {
  background: "#111114",
  border: "1px solid rgba(255,255,255,0.06)",
  borderRadius: "12px",
  fontSize: "12px",
  color: "#f0f0f0",
};

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

  // Lazy-load SHAP when the tab is first opened
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
    try {
      await downloadReportPdf(runId);
    } catch (e: unknown) {
      setPdfError(e instanceof Error ? e.message : "PDF download failed");
    } finally {
      setPdfBusy(false);
    }
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

  // ── Loading / error states ─────────────────────────────────────────────────

  if (loading) {
    return (
      <>
        <Navbar />
        <main className="pt-28 pb-20 min-h-screen flex items-center justify-center">
          <div className="w-8 h-8 border-2 border-accent/30 border-t-accent rounded-full animate-spin" />
        </main>
      </>
    );
  }

  if (!results || results.status !== "completed") {
    return (
      <>
        <Navbar />
        <main className="pt-28 pb-20 min-h-screen flex items-center justify-center">
          <div className="text-text-muted text-center">
            <p className="text-lg">Results not available yet.</p>
            <p className="text-sm mt-2">The pipeline may still be running.</p>
          </div>
        </main>
      </>
    );
  }

  // ── Derived data ───────────────────────────────────────────────────────────

  const bestModel = results.models.find((m) => m.is_best);

  const modelChartData = results.models
    .filter((m) => m.status === "trained")
    .map((m) => ({
      name:   m.name.replace(/Classifier|Regressor/g, "").trim(),
      metric: Object.values(m.metrics)[0] ?? 0,
      isBest: m.is_best,
    }))
    .sort((a, b) => b.metric - a.metric);

  const shapChartData = shapData
    ? Object.entries(shapData)
        .slice(0, 15)
        .map(([feature, value]) => ({ feature, value }))
    : [];

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <>
      <Navbar />

      <main className="pt-28 pb-20 min-h-screen">
        <div className="max-w-[1200px] mx-auto px-8">

          {/* Header */}
          <motion.div
            variants={stagger}
            initial="hidden"
            animate="visible"
            className="mb-10 flex flex-col md:flex-row md:items-end md:justify-between gap-4"
          >
            <div>
              <motion.span
                variants={fadeUp}
                className="text-[11px] font-semibold uppercase tracking-[3px] text-accent"
              >
                Pipeline Complete
              </motion.span>
              <motion.h1
                variants={fadeUp}
                className="text-3xl md:text-4xl font-bold tracking-[-0.03em] mt-3 gradient-text-hero"
              >
                Execution Report
              </motion.h1>
              <motion.p variants={fadeUp} className="text-text-muted mt-2">
                <span className="font-mono text-[13px] text-accent">{runId}</span>
              </motion.p>
            </div>
            <motion.div variants={fadeUp} className="flex items-center gap-3">
              <button
                onClick={handleDownloadPdf}
                disabled={pdfBusy}
                className="group flex items-center gap-2 px-5 py-3 rounded-xl text-[13px] font-semibold bg-accent text-void hover:bg-accent-bright transition-all duration-300 shadow-lg shadow-accent/20 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {pdfBusy ? (
                  <div className="w-3.5 h-3.5 border-2 border-void/30 border-t-void rounded-full animate-spin" />
                ) : (
                  <Download size={14} strokeWidth={2} />
                )}
                {pdfBusy ? "Generating…" : "Download PDF"}
              </button>
            </motion.div>
          </motion.div>
          {pdfError && (
            <div className="mb-6 glass-sm p-3 border-destructive/20 bg-destructive/[0.04] text-destructive text-[12px]">
              {pdfError}
            </div>
          )}

          {/* Tabs */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="flex gap-1 mb-8 glass-sm p-1.5 w-fit"
          >
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl text-[12px] font-semibold uppercase tracking-wider transition-all duration-300 ${
                  tab === t.id
                    ? "bg-white/[0.08] text-text-primary"
                    : "text-text-muted hover:text-text-secondary"
                }`}
              >
                <t.icon size={13} strokeWidth={1.5} />
                {t.label}
              </button>
            ))}
          </motion.div>

          {/* Tab content */}
          <motion.div
            key={tab}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35, ease: [0.25, 0.4, 0, 1] }}
          >
            {/* ── Overview ────────────────────────────────────────────────── */}
            {tab === "overview" && (
              <div className="space-y-5">
                {/* KPI cards */}
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                  {[
                    { label: "Problem Type",   value: results.problem_type || "-",                                    icon: Layers,   color: "text-accent" },
                    { label: "Best Model",     value: bestModel?.name || "-",                                         icon: Trophy,   color: "text-warning" },
                    { label: results.best_metric_name || "Metric", value: results.best_metric_value?.toFixed(4) || "-", icon: BarChart3, color: "text-success" },
                    { label: "Models Trained", value: String(results.models.filter((m) => m.status === "trained").length), icon: Brain, color: "text-info" },
                  ].map((kpi, i) => (
                    <motion.div
                      key={kpi.label}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.08 * i }}
                      className="glass p-6 hover:border-glass-hover transition-all duration-500"
                    >
                      <kpi.icon size={16} className={`${kpi.color} mb-3 opacity-60`} strokeWidth={1.5} />
                      <div className="text-2xl font-bold text-text-primary tracking-tight truncate">
                        {kpi.value}
                      </div>
                      <div className="text-[10px] text-text-muted uppercase tracking-[1.5px] mt-1 font-medium">
                        {kpi.label}
                      </div>
                    </motion.div>
                  ))}
                </div>

                {/* Dataset + Preprocessing cards */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  {/* Dataset */}
                  <div className="glass p-6">
                    <h3 className="text-[10px] font-semibold uppercase tracking-[2px] text-text-muted mb-4">
                      Dataset
                    </h3>
                    <dl className="space-y-2.5">
                      {[
                        { label: "Rows",          value: results.dataset.rows?.toLocaleString() },
                        { label: "Columns",       value: results.dataset.columns },
                        { label: "Quality Score", value: results.dataset.quality_score?.toFixed(3) },
                        { label: "Target",        value: results.target_column || "None (clustering)" },
                      ].map((item) => (
                        <div
                          key={item.label}
                          className="flex justify-between items-center py-1.5 border-b border-glass-border/50 last:border-0"
                        >
                          <dt className="text-[13px] text-text-muted">{item.label}</dt>
                          <dd className="text-[13px] text-text-primary font-medium">{item.value ?? "-"}</dd>
                        </div>
                      ))}
                    </dl>
                  </div>

                  {/* Preprocessing */}
                  <div className="glass p-6">
                    <h3 className="text-[10px] font-semibold uppercase tracking-[2px] text-text-muted mb-4">
                      Preprocessing
                    </h3>
                    <dl className="space-y-2.5">
                      {[
                        { label: "Rows Before",        value: results.preprocessing.rows_before?.toLocaleString() },
                        { label: "Rows After",         value: results.preprocessing.rows_after?.toLocaleString() },
                        { label: "Duplicates Removed", value: results.preprocessing.duplicates_removed },
                        { label: "Quality Score",      value: results.preprocessing.quality_score?.toFixed(3) },
                      ].map((item) => (
                        <div
                          key={item.label}
                          className="flex justify-between items-center py-1.5 border-b border-glass-border/50 last:border-0"
                        >
                          <dt className="text-[13px] text-text-muted">{item.label}</dt>
                          <dd className="text-[13px] text-text-primary font-medium">{item.value ?? "-"}</dd>
                        </div>
                      ))}
                    </dl>
                  </div>
                </div>

                {/* Features card */}
                {results.features && (
                  <div className="glass p-6">
                    <h3 className="text-[10px] font-semibold uppercase tracking-[2px] text-text-muted mb-4">
                      Feature Engineering
                    </h3>
                    <div className="flex gap-8 mb-4">
                      {[
                        { label: "Features Before", value: results.features.before },
                        { label: "Features After",  value: results.features.after },
                      ].map((item) => (
                        <div key={item.label}>
                          <div className="text-xl font-bold text-text-primary">{item.value ?? "-"}</div>
                          <div className="text-[10px] text-text-muted uppercase tracking-[1.2px] mt-0.5">{item.label}</div>
                        </div>
                      ))}
                    </div>
                    {results.features.selected?.length > 0 && (
                      <div className="flex flex-wrap gap-1.5">
                        {results.features.selected.slice(0, 12).map((f) => (
                          <span
                            key={f}
                            className="font-mono text-[11px] text-accent bg-accent/[0.06] border border-accent/20 px-2 py-0.5 rounded-lg"
                          >
                            {f}
                          </span>
                        ))}
                        {results.features.selected.length > 12 && (
                          <span className="text-[11px] text-text-muted px-2 py-0.5">
                            +{results.features.selected.length - 12} more
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* ── Models ──────────────────────────────────────────────────── */}
            {tab === "models" && (
              <div className="space-y-5">
                {/* Bar chart */}
                <div className="glass p-8">
                  <h3 className="text-[10px] font-semibold uppercase tracking-[2px] text-text-muted mb-6">
                    Model Performance Comparison
                  </h3>
                  <div className="h-[280px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={modelChartData}
                        layout="vertical"
                        margin={{ left: 40, right: 30, top: 5, bottom: 5 }}
                      >
                        <XAxis
                          type="number"
                          domain={[0, 1]}
                          tick={{ fill: "#505050", fontSize: 11 }}
                          axisLine={false}
                          tickLine={false}
                        />
                        <YAxis
                          type="category"
                          dataKey="name"
                          tick={{ fill: "#8a8a8a", fontSize: 12 }}
                          axisLine={false}
                          tickLine={false}
                          width={130}
                        />
                        <Tooltip
                          contentStyle={SHAP_TOOLTIP_STYLE}
                          formatter={(v) => [typeof v === "number" ? v.toFixed(4) : v, "Score"]}
                        />
                        <Bar dataKey="metric" radius={[0, 6, 6, 0]} barSize={18}>
                          {modelChartData.map((entry, i) => (
                            <Cell
                              key={i}
                              fill={
                                entry.isBest
                                  ? "#00e5c8"
                                  : "rgba(0,229,200,0.2)"
                              }
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Model table */}
                <div className="glass overflow-hidden">
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b border-glass-border bg-white/[0.015]">
                          {["Model", "Status", "Metrics", "Time"].map((h) => (
                            <th
                              key={h}
                              className={`py-3 px-5 text-[10px] font-semibold uppercase tracking-[1.5px] text-text-ghost ${
                                h === "Time" ? "text-right" : "text-left"
                              }`}
                            >
                              {h}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {results.models.map((m) => (
                          <tr
                            key={m.name}
                            className="border-b border-glass-border/50 last:border-0 hover:bg-white/[0.02] transition-colors"
                          >
                            <td className="py-3.5 px-5">
                              <div className="flex items-center gap-2.5">
                                {m.is_best && <Trophy size={13} className="text-warning shrink-0" />}
                                <span className="text-[13px] font-medium text-text-primary">{m.name}</span>
                              </div>
                            </td>
                            <td className="py-3.5 px-5">
                              <span
                                className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wider ${
                                  m.status === "trained"
                                    ? "bg-success/10 text-success"
                                    : "bg-destructive/10 text-destructive"
                                }`}
                              >
                                {m.status === "trained" ? <Zap size={10} /> : <AlertTriangle size={10} />}
                                {m.status}
                              </span>
                            </td>
                            <td className="py-3.5 px-5 text-[12px] text-text-secondary font-mono">
                              {Object.entries(m.metrics ?? {})
                                .map(([k, v]) => `${k}: ${v.toFixed(4)}`)
                                .join("  ·  ")}
                            </td>
                            <td className="py-3.5 px-5 text-right">
                              <span className="inline-flex items-center gap-1 text-[12px] text-text-muted">
                                <Clock size={11} />
                                {m.time_s?.toFixed(2) ?? "-"}s
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            {/* ── SHAP ────────────────────────────────────────────────────── */}
            {tab === "shap" && (
              <div className="space-y-5">
                {shapLoading ? (
                  <div className="glass p-16 flex items-center justify-center">
                    <div className="w-7 h-7 border-2 border-accent/30 border-t-accent rounded-full animate-spin" />
                  </div>
                ) : shapData && Object.keys(shapData).length > 0 ? (
                  <>
                    {/* Explanation header */}
                    <div className="glass p-6">
                      <div className="flex items-center gap-2.5 mb-2">
                        <Sparkles size={15} className="text-accent" strokeWidth={1.5} />
                        <h3 className="text-[10px] font-semibold uppercase tracking-[2px] text-text-muted">
                          SHAP Feature Importance
                        </h3>
                      </div>
                      <p className="text-[13px] text-text-secondary leading-relaxed">
                        SHAP (SHapley Additive exPlanations) quantifies each feature&apos;s
                        average contribution to model predictions. Higher mean |SHAP| values
                        indicate greater influence on the output.
                      </p>
                    </div>

                    {/* Bar chart */}
                    <div className="glass p-8">
                      <h3 className="text-[10px] font-semibold uppercase tracking-[2px] text-text-muted mb-6">
                        Top {shapChartData.length} Features by Mean |SHAP Value|
                      </h3>
                      <div style={{ height: Math.max(260, shapChartData.length * 36 + 40) }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={shapChartData}
                            layout="vertical"
                            margin={{ left: 20, right: 60, top: 5, bottom: 5 }}
                          >
                            <XAxis
                              type="number"
                              tick={{ fill: "#505050", fontSize: 11 }}
                              axisLine={false}
                              tickLine={false}
                              tickFormatter={(v: number) => v.toFixed(3)}
                            />
                            <YAxis
                              type="category"
                              dataKey="feature"
                              tick={{ fill: "#8a8a8a", fontSize: 12 }}
                              axisLine={false}
                              tickLine={false}
                              width={160}
                            />
                            <Tooltip
                              contentStyle={SHAP_TOOLTIP_STYLE}
                              formatter={(v) => [typeof v === "number" ? v.toFixed(6) : v, "Mean |SHAP|"]}
                            />
                            <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={20}>
                              {shapChartData.map((_, i) => {
                                const alpha = Math.max(0.25, 1 - i * (0.7 / shapChartData.length));
                                return (
                                  <Cell
                                    key={i}
                                    fill={`rgba(0, 229, 200, ${alpha.toFixed(2)})`}
                                  />
                                );
                              })}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    {/* SHAP table */}
                    <div className="glass overflow-hidden">
                      <div className="overflow-x-auto">
                        <table className="w-full">
                          <thead>
                            <tr className="border-b border-glass-border bg-white/[0.015]">
                              <th className="py-3 px-5 text-left text-[10px] font-semibold uppercase tracking-[1.5px] text-text-ghost">
                                Rank
                              </th>
                              <th className="py-3 px-5 text-left text-[10px] font-semibold uppercase tracking-[1.5px] text-text-ghost">
                                Feature
                              </th>
                              <th className="py-3 px-5 text-right text-[10px] font-semibold uppercase tracking-[1.5px] text-text-ghost">
                                Mean |SHAP Value|
                              </th>
                            </tr>
                          </thead>
                          <tbody>
                            {shapChartData.map((item, i) => (
                              <tr
                                key={item.feature}
                                className="border-b border-glass-border/50 last:border-0 hover:bg-white/[0.02] transition-colors"
                              >
                                <td className="py-3 px-5 text-[12px] font-mono text-text-muted">
                                  #{i + 1}
                                </td>
                                <td className="py-3 px-5 text-[13px] font-medium text-text-primary">
                                  {item.feature}
                                </td>
                                <td className="py-3 px-5 text-right font-mono text-[13px] text-accent">
                                  {item.value.toFixed(6)}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="glass p-16 text-center">
                    <div className="w-12 h-12 rounded-2xl bg-accent/[0.06] border border-accent/20 flex items-center justify-center mx-auto mb-4">
                      <Sparkles size={20} className="text-accent" strokeWidth={1.5} />
                    </div>
                    <p className="text-text-secondary text-[14px]">SHAP data not available for this run.</p>
                    <p className="text-[12px] text-text-muted mt-2">
                      SHAP explanations are generated when a trained tree or linear model with
                      numeric features is present.
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* ── Issues ──────────────────────────────────────────────────── */}
            {tab === "errors" && (
              <div className="space-y-3">
                {results.errors.length === 0 ? (
                  <div className="glass p-14 text-center">
                    <div className="w-12 h-12 rounded-2xl bg-success/[0.08] border border-success/20 flex items-center justify-center mx-auto mb-4">
                      <Zap size={20} className="text-success" strokeWidth={1.5} />
                    </div>
                    <p className="text-text-secondary">No issues detected - pipeline ran cleanly.</p>
                  </div>
                ) : (
                  results.errors.map((err, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.05 }}
                      className={`glass p-6 border-l-2 ${
                        err.severity === "critical"
                          ? "border-l-destructive"
                          : err.severity === "warning"
                          ? "border-l-warning"
                          : "border-l-accent"
                      }`}
                    >
                      <div className="flex items-start gap-3">
                        <AlertTriangle
                          size={15}
                          strokeWidth={1.5}
                          className={
                            err.severity === "critical"
                              ? "text-destructive mt-0.5 shrink-0"
                              : err.severity === "warning"
                              ? "text-warning mt-0.5 shrink-0"
                              : "text-accent mt-0.5 shrink-0"
                          }
                        />
                        <div className="min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-[13px] font-semibold text-text-primary capitalize">
                              {err.type}
                            </span>
                            <span
                              className={`text-[9px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-full ${
                                err.severity === "critical"
                                  ? "bg-destructive/10 text-destructive"
                                  : err.severity === "warning"
                                  ? "bg-warning/10 text-warning"
                                  : "bg-accent/10 text-accent"
                              }`}
                            >
                              {err.severity}
                            </span>
                          </div>
                          <p className="text-[12px] text-text-muted leading-relaxed">{err.cause}</p>
                          {err.fix && (
                            <p className="text-[12px] text-accent mt-2">→ {err.fix}</p>
                          )}
                        </div>
                      </div>
                    </motion.div>
                  ))
                )}
              </div>
            )}

            {/* ── Visualizations ────────────────────────────────────────────── */}
            {tab === "viz" && (
              <div className="space-y-6">
                <div className="glass p-6">
                  <div className="flex items-center gap-2.5 mb-2">
                    <BarChart3 size={15} className="text-accent" strokeWidth={1.5} />
                    <h3 className="text-[10px] font-semibold uppercase tracking-[2px] text-text-muted">
                      Axiom Visualization Studio
                    </h3>
                  </div>
                  <p className="text-[13px] text-text-secondary leading-relaxed">
                    Explore automatic high-resolution charts generated using Matplotlib and Seaborn under the Axiom Dark Theme. If a chart has not been computed yet, click <strong>Generate Chart</strong> to render it on the fly.
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {[
                    { id: "missing_values", name: "Missing Values Heatmap", desc: "Heatmap analysis of missing values across all columns", category: "basic" },
                    { id: "correlation", name: "Feature Correlation Matrix", desc: "Pearson correlation coefficients showing feature-to-feature relations", category: "basic" },
                    { id: "distributions", name: "Feature Distributions", desc: "Univariate distribution histograms for numerical features", category: "basic" },
                    { id: "boxplots", name: "Outlier Box Plots", desc: "Box plots showing outlier distributions after standardization", category: "basic" },
                    { id: "dtype_distribution", name: "Data Type Distribution", desc: "Breakdown of features by Pandas data type category", category: "basic" },
                    ...(results?.target_column ? [{ id: "target_distribution", name: "Target Distribution", desc: `Distribution of target variable: "${results.target_column}"`, category: "basic" }] : []),
                    { id: "pairplot", name: "Pair Plot", desc: "Seaborn pairwise relation grid showing joint and marginal distributions", category: "advanced" },
                    { id: "pca", name: "PCA 2D Projection", desc: "Dimensionality reduction projection mapping features to 2 principal components", category: "advanced" },
                  ].map((chart) => {
                    const existing = vizs.find((v) => v.type === chart.id || v.name.toLowerCase().includes(chart.name.toLowerCase()));
                    const isGenerating = generatingViz === chart.id;

                    return (
                      <div key={chart.id} className="viz-card flex flex-col justify-between h-full">
                        {existing ? (
                          <>
                            <img
                              src={`data:image/png;base64,${existing.base64_png}`}
                              alt={existing.name}
                              className="w-full object-contain"
                            />
                            <div className="p-4 border-t border-glass-border mt-auto">
                              <div className="flex items-center justify-between">
                                <span className="text-[13px] font-semibold text-text-primary">{existing.name}</span>
                                <span className={`text-[9px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full ${existing.category === "advanced" ? "bg-pro-gold/10 text-pro-gold-bright" : "bg-accent/10 text-accent"}`}>
                                  {existing.category}
                                </span>
                              </div>
                              <div className="text-[11px] text-text-muted mt-1">{existing.description}</div>
                            </div>
                          </>
                        ) : (
                          <div className="p-8 flex flex-col items-center justify-center text-center min-h-[220px] bg-white/[0.01]">
                            <BarChart3 size={32} className="text-text-ghost mb-3" strokeWidth={1.25} />
                            <div className="text-[14px] font-semibold text-text-secondary">{chart.name}</div>
                            <div className="text-[11px] text-text-muted mt-1 max-w-[280px]">{chart.desc}</div>
                            <span className={`text-[9px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full mt-2 ${chart.category === "advanced" ? "bg-pro-gold/10 text-pro-gold-bright" : "bg-accent/10 text-accent"}`}>
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
                              className="mt-5 flex items-center gap-1.5 px-4 py-2 rounded-xl text-[11px] font-semibold bg-accent text-void hover:bg-accent-bright transition-colors disabled:opacity-40 cursor-pointer"
                            >
                              {isGenerating ? (
                                <>
                                  <Loader2 size={11} className="animate-spin" />
                                  Rendering...
                                </>
                              ) : (
                                <>
                                  <RefreshCw size={11} />
                                  Generate Chart
                                </>
                              )}
                            </button>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* ── Report ──────────────────────────────────────────────────── */}
            {tab === "report" && (
              <div className="space-y-4">
                {/* Action bar */}
                <div className="flex items-center justify-between">
                  <p className="text-[12px] text-text-muted">
                    Full pipeline report generated by Axiom.
                  </p>
                  <button
                    onClick={handleDownloadMd}
                    disabled={!report}
                    className="flex items-center gap-2 px-4 py-2 rounded-xl text-[12px] font-semibold bg-accent/[0.08] border border-accent/20 text-accent hover:bg-accent/[0.14] transition-colors duration-200 disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    <Download size={13} strokeWidth={2} />
                    Download .md
                  </button>
                </div>

                {/* Rendered markdown */}
                {report ? (
                  <div className="glass p-8 md:p-10">
                    <MdContent content={report} />
                  </div>
                ) : (
                  <div className="glass p-14 text-center">
                    <p className="text-text-muted">No report available for this run.</p>
                  </div>
                )}
              </div>
            )}
          </motion.div>
        </div>
      </main>
    </>
  );
}
