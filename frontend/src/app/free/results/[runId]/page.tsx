"use client";

import { useEffect, useState, useCallback, use } from "react";
import { motion } from "framer-motion";
import { Trophy, Layers, Brain, BarChart3, Zap, Clock, AlertTriangle, FileText, Sparkles, Download } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import Navbar from "@/components/layout/Navbar";
import { getResults, getReport, getVisualizations, downloadReportPdf, ResultsResponse, VizResult } from "@/lib/api";
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

export default function FreeResultsPage({ params }: { params: Promise<{ runId: string }> }) {
  const { runId } = use(params);
  const [results, setResults] = useState<ResultsResponse | null>(null);
  const [report, setReport] = useState("");
  const [vizs, setVizs] = useState<VizResult[]>([]);
  const [tab, setTab] = useState("overview");
  const [loading, setLoading] = useState(true);
  const [pdfBusy, setPdfBusy] = useState(false);
  const [pdfError, setPdfError] = useState<string | null>(null);

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

  if (loading) return (<><Navbar /><main className="pt-28 min-h-screen flex items-center justify-center"><div className="w-8 h-8 border-2 border-accent/30 border-t-accent rounded-full animate-spin" /></main></>);
  if (!results || results.status !== "completed") return (<><Navbar /><main className="pt-28 min-h-screen flex items-center justify-center"><p className="text-text-muted">Results not available yet.</p></main></>);

  const bestModel = results.models.find((m) => m.is_best);
  const trainedModels = results.models.filter((m) => m.status === "trained");
  const chartData = trainedModels.map((m) => ({ name: m.name.replace(/Classifier|Regressor/g, "").trim(), metric: Object.values(m.metrics)[0] || 0, isBest: m.is_best })).sort((a, b) => b.metric - a.metric);

  const TABS = [
    { id: "overview", label: "Overview", icon: Layers },
    { id: "models", label: "Models", icon: Brain },
    { id: "viz", label: "Visualizations", icon: BarChart3 },
    { id: "issues", label: "Issues", icon: AlertTriangle },
    { id: "report", label: "Report", icon: FileText },
  ];

  return (
    <>
      <Navbar />
      <main className="pt-28 pb-20 min-h-screen">
        <div className="max-w-[1200px] mx-auto px-8">

          {/* Header */}
          <motion.div variants={stagger} initial="hidden" animate="visible" className="mb-10 flex flex-col md:flex-row md:items-end md:justify-between gap-4">
            <div>
              <motion.div variants={fadeUp} className="flex items-center gap-2 mb-3">
                <div className="badge badge-success"><Sparkles size={10} /> Complete</div>
              </motion.div>
              <motion.h1 variants={fadeUp} className="text-3xl md:text-4xl font-bold tracking-[-0.03em] gradient-text-hero">Your AI Results</motion.h1>
              <motion.p variants={fadeUp} className="text-[14px] text-text-secondary mt-2">Your data scientist finished the job. Here's everything it found.</motion.p>
            </div>
            <motion.div variants={fadeUp}>
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

          {/* Hero metric */}
          {bestModel && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="glass p-8 mb-8 flex items-center gap-6">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-warning/20 to-warning/5 flex items-center justify-center">
                <Trophy size={28} className="text-warning" />
              </div>
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-[2px] text-text-muted mb-1">Best Model</div>
                <div className="text-2xl font-bold text-text-primary">{bestModel.name}</div>
                <div className="text-[14px] text-accent font-mono mt-1">{results.best_metric_name}: {results.best_metric_value?.toFixed(4)}</div>
              </div>
              <div className="ml-auto grid grid-cols-3 gap-6 text-center">
                <div><div className="text-xl font-bold text-text-primary">{trainedModels.length}</div><div className="text-[10px] text-text-muted uppercase tracking-wider">Models</div></div>
                <div><div className="text-xl font-bold text-text-primary">{results.dataset.rows?.toLocaleString()}</div><div className="text-[10px] text-text-muted uppercase tracking-wider">Rows</div></div>
                <div><div className="text-xl font-bold text-text-primary">{results.retry_count}</div><div className="text-[10px] text-text-muted uppercase tracking-wider">Retries</div></div>
              </div>
            </motion.div>
          )}

          {/* Tabs */}
          <div className="flex gap-1 mb-8 glass-sm p-1.5 w-fit">
            {TABS.map((t) => (
              <button key={t.id} onClick={() => setTab(t.id)} className={`flex items-center gap-2 px-4 py-2 rounded-xl text-[11px] font-semibold uppercase tracking-wider transition-all duration-300 ${tab === t.id ? "bg-white/[0.08] text-text-primary" : "text-text-muted hover:text-text-secondary"}`}>
                <t.icon size={13} strokeWidth={1.5} /> {t.label}
              </button>
            ))}
          </div>

          {/* Content */}
          <motion.div key={tab} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>

            {tab === "overview" && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div className="glass glass-card-hover p-6"><h3 className="text-[11px] font-semibold uppercase tracking-[2px] text-text-muted mb-4">Dataset</h3><div className="space-y-3">
                  {[{ l: "Rows", v: results.dataset.rows?.toLocaleString() }, { l: "Columns", v: results.dataset.columns }, { l: "Quality", v: results.dataset.quality_score?.toFixed(3) }, { l: "Target", v: results.target_column || "None" }, { l: "Type", v: results.problem_type }].map((x) => (
                    <div key={x.l} className="flex justify-between py-1.5 border-b border-glass-border/50 last:border-0"><span className="text-[13px] text-text-muted">{x.l}</span><span className="text-[13px] text-text-primary font-medium">{x.v ?? "-"}</span></div>))}
                </div></div>
                <div className="glass glass-card-hover p-6"><h3 className="text-[11px] font-semibold uppercase tracking-[2px] text-text-muted mb-4">Preprocessing</h3><div className="space-y-3">
                  {[{ l: "Rows Before", v: results.preprocessing.rows_before?.toLocaleString() }, { l: "Rows After", v: results.preprocessing.rows_after?.toLocaleString() }, { l: "Duplicates Removed", v: results.preprocessing.duplicates_removed }, { l: "Quality Score", v: results.preprocessing.quality_score?.toFixed(3) }].map((x) => (
                    <div key={x.l} className="flex justify-between py-1.5 border-b border-glass-border/50 last:border-0"><span className="text-[13px] text-text-muted">{x.l}</span><span className="text-[13px] text-text-primary font-medium">{x.v ?? "-"}</span></div>))}
                </div></div>
              </div>
            )}

            {tab === "models" && (
              <div className="space-y-6">
                <div className="glass p-8">
                  <h3 className="text-[11px] font-semibold uppercase tracking-[2px] text-text-muted mb-6">Performance</h3>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData} layout="vertical" margin={{ left: 40, right: 20 }}>
                        <XAxis type="number" domain={[0, 1]} tick={{ fill: "#52525b", fontSize: 11 }} axisLine={false} tickLine={false} />
                        <YAxis type="category" dataKey="name" tick={{ fill: "#a1a1aa", fontSize: 12 }} axisLine={false} tickLine={false} width={120} />
                        <Tooltip contentStyle={{ background: "#18181c", border: "1px solid rgba(255,255,255,0.06)", borderRadius: "12px", fontSize: "12px", color: "#fafafa" }} />
                        <Bar dataKey="metric" radius={[0, 6, 6, 0]} barSize={20}>
                          {chartData.map((e, i) => (<Cell key={i} fill={e.isBest ? "#818cf8" : "rgba(129,140,248,0.25)"} />))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                <div className="glass overflow-hidden"><table className="w-full"><thead><tr className="border-b border-glass-border">
                  <th className="text-left py-3 px-5 text-[10px] font-semibold uppercase tracking-[1.5px] text-text-ghost">Model</th>
                  <th className="text-left py-3 px-5 text-[10px] font-semibold uppercase tracking-[1.5px] text-text-ghost">Status</th>
                  <th className="text-left py-3 px-5 text-[10px] font-semibold uppercase tracking-[1.5px] text-text-ghost">Metrics</th>
                  <th className="text-right py-3 px-5 text-[10px] font-semibold uppercase tracking-[1.5px] text-text-ghost">Time</th>
                </tr></thead><tbody>
                  {results.models.map((m) => (
                    <tr key={m.name} className="border-b border-glass-border/50 hover:bg-glass-hover transition-colors">
                      <td className="py-3 px-5"><div className="flex items-center gap-2">{m.is_best && <Trophy size={14} className="text-warning" />}<span className="text-[13px] font-medium">{m.name}</span></div></td>
                      <td className="py-3 px-5"><span className={`badge ${m.status === "trained" ? "badge-success" : "badge-error"}`}>{m.status === "trained" ? <Zap size={10} /> : <AlertTriangle size={10} />}{m.status}</span></td>
                      <td className="py-3 px-5 text-[12px] text-text-secondary font-mono">{Object.entries(m.metrics || {}).map(([k, v]) => `${k}: ${v.toFixed(4)}`).join(" · ")}</td>
                      <td className="py-3 px-5 text-right text-[12px] text-text-muted"><Clock size={11} className="inline mr-1" />{m.time_s?.toFixed(2)}s</td>
                    </tr>))}
                </tbody></table></div>
              </div>
            )}

            {tab === "viz" && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {vizs.length === 0 ? (
                  <div className="col-span-2 glass p-12 text-center"><p className="text-text-muted">No visualizations available.</p></div>
                ) : vizs.map((viz) => (
                  <div key={viz.name} className="viz-card">
                    <img src={`data:image/png;base64,${viz.base64_png}`} alt={viz.name} />
                    <div className="p-4 border-t border-glass-border">
                      <div className="text-[13px] font-semibold text-text-primary">{viz.name}</div>
                      <div className="text-[11px] text-text-muted mt-1">{viz.description}</div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {tab === "issues" && (
              <div className="space-y-3">
                {results.errors.length === 0 ? (
                  <div className="glass p-12 text-center"><div className="badge badge-success mx-auto mb-3"><Zap size={10} /> Clean</div><p className="text-text-secondary">No issues detected.</p></div>
                ) : results.errors.map((err, i) => (
                  <div key={i} className={`glass p-5 border-l-2 ${err.severity === "critical" ? "border-l-destructive" : err.severity === "warning" ? "border-l-warning" : "border-l-accent"}`}>
                    <div className="text-[13px] font-semibold capitalize">{err.type}</div>
                    <div className="text-[12px] text-text-muted mt-1">{err.cause}</div>
                    {err.fix && <div className="text-[12px] text-accent mt-2">→ {err.fix}</div>}
                  </div>
                ))}
              </div>
            )}

            {tab === "report" && (
              <div className="glass p-8">
                {report ? (
                  <MdContent content={report} />
                ) : (
                  <p className="text-text-muted text-center">No report available.</p>
                )}
              </div>
            )}
          </motion.div>
        </div>
      </main>
    </>
  );
}
