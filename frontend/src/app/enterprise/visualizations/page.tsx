"use client";

import { useEffect, useState, useMemo, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  BarChart3, Loader2, Sparkles, AlertTriangle, ChevronDown,
  CheckCircle2, Play, RefreshCw, X, Download, ZoomIn, Info,
  Database, Layers, Eye, RefreshCw as LoopIcon
} from "lucide-react";
import { listExperiments, getVisualizations, generateVisualization, VizResult } from "@/lib/api";
import { fadeUp, stagger } from "@/lib/animations";

type RunSummary = Awaited<ReturnType<typeof listExperiments>>["runs"][number];
type Tab = "all" | "basic" | "advanced" | "model";

const CHART_TEMPLATES = [
  { id: "missing_values", name: "Missing Values Heatmap", desc: "Heatmap analysis of missing values across all columns", category: "basic" },
  { id: "correlation", name: "Feature Correlation Matrix", desc: "Pearson correlation coefficients showing feature-to-feature relations", category: "basic" },
  { id: "distributions", name: "Feature Distributions", desc: "Univariate distribution histograms for numerical features", category: "basic" },
  { id: "boxplots", name: "Outlier Box Plots", desc: "Box plots showing outlier distributions after standardization", category: "basic" },
  { id: "dtype_distribution", name: "Data Type Distribution", desc: "Breakdown of features by Pandas data type category", category: "basic" },
  { id: "target_distribution", name: "Target Distribution", desc: "Distribution of target variable if one is selected", category: "basic" },
  { id: "pairplot", name: "Pair Plot", desc: "Seaborn pairwise relation grid showing joint and marginal distributions", category: "advanced" },
  { id: "pca", name: "PCA 2D Projection", desc: "Dimensionality reduction projection mapping features to 2 principal components", category: "advanced" },
  { id: "model_comparison", name: "Model Performance Comparison", desc: "Performance comparison across all trained models", category: "model" },
  { id: "feature_importance", name: "Feature Importance", desc: "Features ranked by their predictive contribution to the best model", category: "model" },
];

export default function VisualizationsStudio() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string>("");
  const [loadingRuns, setLoadingRuns] = useState(true);
  const [vizs, setVizs] = useState<VizResult[]>([]);
  const [loadingVizs, setLoadingVizs] = useState(false);
  const [generatingViz, setGeneratingViz] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("all");
  const [activeModalViz, setActiveModalViz] = useState<VizResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Fetch runs list on mount
  useEffect(() => {
    listExperiments()
      .then((data) => {
        setRuns(data.runs || []);
        if (data.runs && data.runs.length > 0) {
          setSelectedRunId(data.runs[0].run_id);
        }
      })
      .catch((e) => {
        setError(e instanceof Error ? e.message : "Failed to load runs list");
      })
      .finally(() => {
        setLoadingRuns(false);
      });
  }, []);

  // Fetch visualizations for selected run
  useEffect(() => {
    if (!selectedRunId) {
      setVizs([]);
      return;
    }
    setLoadingVizs(true);
    getVisualizations(selectedRunId)
      .then((data) => {
        setVizs(data || []);
      })
      .catch(() => {
        setVizs([]);
      })
      .finally(() => {
        setLoadingVizs(false);
      });
  }, [selectedRunId]);

  const selectedRun = useMemo(() => {
    return runs.find((r) => r.run_id === selectedRunId) || null;
  }, [runs, selectedRunId]);

  const handleGenerate = async (type: string) => {
    if (!selectedRunId || generatingViz) return;
    setGeneratingViz(type);
    setError(null);
    try {
      const newViz = await generateVisualization(selectedRunId, type);
      setVizs((prev) => {
        const filtered = prev.filter((v) => v.type !== type && !v.name.toLowerCase().includes(type.toLowerCase()));
        return [...filtered, newViz];
      });
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to generate visualization chart.");
    } finally {
      setGeneratingViz(null);
    }
  };

  const handleDownload = (viz: VizResult) => {
    const link = document.createElement("a");
    link.href = `data:image/png;base64,${viz.base64_png}`;
    link.download = `${viz.type || "chart"}-${selectedRunId.slice(0, 8)}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const filteredTemplates = useMemo(() => {
    return CHART_TEMPLATES.filter((template) => {
      // If template is target_distribution, check if target_column exists in run metadata
      if (template.id === "target_distribution" && selectedRun && !selectedRun.best_model && !selectedRun.best_metric_value) {
        // We allow generating it since target might be specified, but if it has no metrics let's keep it best effort
      }
      
      if (activeTab === "all") return true;
      return template.category === activeTab;
    });
  }, [activeTab, selectedRun]);

  return (
    <div className="min-h-screen px-6 md:px-10 py-10 max-w-[1240px]">
      <motion.header
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="mb-8 flex flex-col md:flex-row md:items-center md:justify-between gap-6"
      >
        <div>
          <div className="badge badge-accent mb-3">
            <BarChart3 size={10} /> Visualizations Studio
          </div>
          <h1 className="font-display text-3xl md:text-4xl font-bold text-text-primary tracking-tight">
            Data Visualization
          </h1>
          <p className="text-sm text-text-secondary mt-1">
            Generate and explore deep analytical charts for dataset profiling, preprocessing, and training stages.
          </p>
        </div>

        {/* Run Selector */}
        {!loadingRuns && runs.length > 0 && (
          <div className="flex flex-col gap-1.5 min-w-[280px]">
            <label className="text-[10px] uppercase tracking-[1.5px] font-bold text-text-ghost">
              Select ML Execution Run
            </label>
            <div className="relative">
              <select
                value={selectedRunId}
                onChange={(e) => setSelectedRunId(e.target.value)}
                className="w-full appearance-none bg-void/50 border border-glass-border hover:border-glass-hover rounded-xl px-4 py-2.5 text-[13px] text-text-primary font-mono focus:outline-none focus:border-accent/40 transition-colors pr-9 cursor-pointer"
              >
                {runs.map((r) => (
                  <option key={r.run_id} value={r.run_id}>
                    {r.run_id.slice(0, 12)}... ({r.best_model || r.status})
                  </option>
                ))}
              </select>
              <ChevronDown
                size={14}
                className="absolute right-3.5 top-1/2 -translate-y-1/2 text-text-muted pointer-events-none"
              />
            </div>
          </div>
        )}
      </motion.header>

      {error && (
        <div className="flex items-start gap-3 p-4 rounded-xl border border-destructive/20 bg-destructive/[0.04] mb-6">
          <AlertTriangle size={15} className="text-destructive shrink-0 mt-0.5" />
          <span className="text-destructive text-[12.5px]">{error}</span>
        </div>
      )}

      {loadingRuns ? (
        <div className="p-20 flex items-center justify-center">
          <Loader2 size={24} className="text-accent animate-spin" />
        </div>
      ) : runs.length === 0 ? (
        <div className="glass p-16 text-center rounded-2xl border border-glass-border">
          <Database size={36} className="text-text-ghost mx-auto mb-3" strokeWidth={1.5} />
          <h3 className="text-md font-semibold text-text-secondary">No runs detected</h3>
          <p className="text-xs text-text-muted mt-1 max-w-sm mx-auto">
            Please run the pipeline or custom workflow on a dataset to generate run logs and visualizations.
          </p>
        </div>
      ) : (
        <>
          {/* Filters + Metadata info bar */}
          <div className="flex flex-wrap items-center justify-between gap-4 mb-6 border-b border-glass-border pb-4">
            <div className="flex items-center gap-1.5">
              {(["all", "basic", "advanced", "model"] as Tab[]).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-3.5 py-1.5 rounded-lg text-[12px] font-semibold uppercase tracking-wider transition-all ${
                    activeTab === tab
                      ? "bg-white/[0.08] text-text-primary"
                      : "text-text-muted hover:text-text-secondary"
                  }`}
                >
                  {tab}
                </button>
              ))}
            </div>

            {selectedRun && (
              <div className="flex items-center gap-4 text-[11px] text-text-muted">
                <span>
                  Status: <strong className="text-text-secondary capitalize">{selectedRun.status}</strong>
                </span>
                <span>•</span>
                <span>
                  Stages: <strong className="text-text-secondary">{selectedRun.completed_stages?.length || 0}/8</strong>
                </span>
              </div>
            )}
          </div>

          {/* Visualizations Grid */}
          {loadingVizs ? (
            <div className="p-20 flex items-center justify-center">
              <Loader2 size={24} className="text-accent animate-spin" />
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              {filteredTemplates.map((template) => {
                const existing = vizs.find(
                  (v) =>
                    v.type === template.id ||
                    v.name.toLowerCase().includes(template.name.toLowerCase())
                );
                const isGenerating = generatingViz === template.id;

                return (
                  <div
                    key={template.id}
                    className="viz-card flex flex-col justify-between h-full border border-glass-border rounded-2xl bg-white/[0.01] overflow-hidden group hover:border-glass-hover transition-all duration-300"
                  >
                    {existing ? (
                      <div className="flex flex-col h-full justify-between">
                        {/* Image panel */}
                        <div className="relative aspect-[3/2] w-full bg-void/30 flex items-center justify-center overflow-hidden">
                          <img
                            src={`data:image/png;base64,${existing.base64_png}`}
                            alt={existing.name}
                            className="w-full h-full object-contain group-hover:scale-[1.01] transition-transform duration-300"
                          />
                          <div className="absolute inset-0 bg-void/60 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center gap-3">
                            <button
                              onClick={() => setActiveModalViz(existing)}
                              className="p-2.5 rounded-full bg-white/10 hover:bg-white/20 border border-white/20 text-white transition-all shadow-md"
                              title="Expand view"
                            >
                              <ZoomIn size={16} />
                            </button>
                            <button
                              onClick={() => handleDownload(existing)}
                              className="p-2.5 rounded-full bg-white/10 hover:bg-white/20 border border-white/20 text-white transition-all shadow-md"
                              title="Download chart"
                            >
                              <Download size={16} />
                            </button>
                          </div>
                        </div>

                        {/* Title and details panel */}
                        <div className="p-4 border-t border-glass-border mt-auto flex justify-between items-start gap-4">
                          <div>
                            <div className="flex items-center gap-2">
                              <span className="text-[13px] font-semibold text-text-primary">
                                {existing.name}
                              </span>
                              <span
                                className={`text-[8.5px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full ${
                                  existing.category === "advanced"
                                    ? "bg-pro-gold/10 text-pro-gold-bright"
                                    : "bg-accent/10 text-accent"
                                }`}
                              >
                                {existing.category}
                              </span>
                            </div>
                            <div className="text-[11px] text-text-muted mt-1">
                              {existing.description}
                            </div>
                          </div>
                        </div>
                      </div>
                    ) : (
                      /* Placeholder view with Generate CTA */
                      <div className="p-10 flex flex-col items-center justify-center text-center min-h-[250px] bg-white/[0.005]">
                        <BarChart3 size={32} className="text-text-ghost mb-3" strokeWidth={1.25} />
                        <div className="text-[14px] font-semibold text-text-secondary">
                          {template.name}
                        </div>
                        <div className="text-[11px] text-text-muted mt-1.5 max-w-[280px]">
                          {template.desc}
                        </div>
                        <span
                          className={`text-[8.5px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full mt-2.5 ${
                            template.category === "advanced"
                              ? "bg-pro-gold/10 text-pro-gold-bright"
                              : "bg-accent/10 text-accent"
                          }`}
                        >
                          {template.category}
                        </span>
                        <button
                          onClick={() => handleGenerate(template.id)}
                          disabled={!!generatingViz}
                          className="mt-6 flex items-center gap-1.5 px-4.5 py-2.5 rounded-xl text-[11px] font-semibold bg-accent text-void hover:bg-accent-bright transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
                        >
                          {isGenerating ? (
                            <>
                              <Loader2 size={12} className="animate-spin" />
                              Rendering Plot...
                            </>
                          ) : (
                            <>
                              <RefreshCw size={12} />
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
          )}
        </>
      )}

      {/* Lightbox / Expansion modal */}
      <AnimatePresence>
        {activeModalViz && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-void/90 backdrop-blur-md"
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="relative max-w-[1000px] w-full bg-void border border-glass-border rounded-2xl overflow-hidden flex flex-col"
            >
              {/* Header */}
              <div className="flex items-center justify-between px-5 py-4 border-b border-glass-border">
                <div>
                  <h3 className="text-[14px] font-semibold text-text-primary">
                    {activeModalViz.name}
                  </h3>
                  <p className="text-[11px] text-text-muted mt-0.5">
                    {activeModalViz.description}
                  </p>
                </div>
                <button
                  onClick={() => setActiveModalViz(null)}
                  className="p-1.5 rounded-lg hover:bg-glass-hover text-text-ghost hover:text-text-secondary transition-colors"
                >
                  <X size={16} />
                </button>
              </div>

              {/* Body */}
              <div className="p-6 bg-white/[0.01] flex items-center justify-center max-h-[70vh]">
                <img
                  src={`data:image/png;base64,${activeModalViz.base64_png}`}
                  alt={activeModalViz.name}
                  className="max-h-[60vh] object-contain rounded-lg"
                />
              </div>

              {/* Footer */}
              <div className="flex items-center justify-between px-5 py-4 border-t border-glass-border bg-white/[0.015]">
                <span
                  className={`text-[9px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full ${
                    activeModalViz.category === "advanced"
                      ? "bg-pro-gold/10 text-pro-gold-bright"
                      : "bg-accent/10 text-accent"
                  }`}
                >
                  {activeModalViz.category} category
                </span>

                <button
                  onClick={() => handleDownload(activeModalViz)}
                  className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-[12px] font-semibold bg-accent text-void hover:bg-accent-bright transition-colors"
                >
                  <Download size={13} />
                  Download Image
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
