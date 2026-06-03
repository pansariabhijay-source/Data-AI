"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import {
  ArrowRight, Cpu, Layers, Brain, Zap, Shield, BarChart3,
  Workflow, Activity, GitBranch, Eye, RefreshCw, Database,
  Terminal, Boxes, ChevronRight, ChevronDown, Sparkles, Upload, Rocket, CheckCircle2,
  Crown, Star, Infinity as InfinityIcon, Lock,
} from "lucide-react";
import Navbar from "@/components/layout/Navbar";
import AgentNetwork from "@/components/ui/AgentNetwork";
import { fadeUp, stagger } from "@/lib/animations";
import { useAuthStore } from "@/store/useAuthStore";

const PIPELINE_STAGES = [
  { n: "01", icon: Database, label: "Data Collection", desc: "Ingest raw data, auto-detect schema and problem type.", color: "#00e5c8" },
  { n: "02", icon: Layers, label: "Preprocessing", desc: "Null handling, outlier removal, quality scoring.", color: "#22d3ee" },
  { n: "03", icon: Brain, label: "Feature Engineering", desc: "Automated feature creation and dimensionality reduction.", color: "#38bdf8" },
  { n: "04", icon: GitBranch, label: "Data Splitting", desc: "Stratified train/val/test splits with cross-validation.", color: "#818cf8" },
  { n: "05", icon: Cpu, label: "Model Training", desc: "13+ models trained in parallel — best auto-selected.", color: "#a78bfa" },
  { n: "06", icon: Eye, label: "Error Detection", desc: "Overfitting, bias, and data quality issues flagged.", color: "#c084fc" },
  { n: "07", icon: RefreshCw, label: "Improvement", desc: "Bayesian hyperparameter search. Ensembles applied.", color: "#34d399" },
  { n: "08", icon: Zap, label: "Finalization", desc: "Artifacts saved. Performance report generated.", color: "#5efce8" },
];

const STEPS = [
  {
    n: "01", icon: Upload, label: "Upload Your Dataset",
    desc: "Drop any CSV. Axiom reads the schema, detects your target variable, and determines the problem type — classification, regression, or clustering — automatically.",
    color: "#00e5c8",
  },
  {
    n: "02", icon: Cpu, label: "Agents Take Over",
    desc: "Eight specialized AI agents orchestrate the entire pipeline without any configuration. Watch each stage complete in real-time.",
    color: "#818cf8",
  },
  {
    n: "03", icon: Rocket, label: "Production-Ready Models",
    desc: "Receive trained models, performance reports, and visualizations. Download artifacts or access results via the REST API.",
    color: "#34d399",
  },
];

const TECH = [
  "CrewAI", "FastAPI", "scikit-learn", "LightGBM", "XGBoost", "CatBoost",
  "Optuna", "pandas", "Next.js 15", "React 19", "Pydantic v2", "Zustand",
  "Framer Motion", "Python 3.11", "NumPy", "Tailwind CSS",
];

export default function LandingPage() {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const authOr = (path: string) => (isAuthenticated ? path : "/auth");

  return (
    <>
      <Navbar />

      {/* ═══════════════ HERO ═══════════════ */}
      <section className="relative min-h-screen flex items-center overflow-hidden">
        <div className="hero-mesh" />
        <div className="aurora" />
        <div className="absolute top-1/4 -left-20 w-[700px] h-[700px] bg-primary/[0.04] rounded-full blur-[200px] pointer-events-none" />
        <div className="absolute bottom-1/4 -right-20 w-[600px] h-[600px] bg-pro/[0.04] rounded-full blur-[170px] pointer-events-none" />

        <div className="relative z-10 w-full max-w-[1340px] mx-auto px-8 pt-28 pb-16">
          <div className="grid lg:grid-cols-[1.05fr_0.95fr] gap-10 items-center">

            {/* ── Left: copy ── */}
            <motion.div variants={stagger} initial="hidden" animate="visible" className="space-y-7 text-center lg:text-left">
              {/* Announcement pill */}
              <motion.div variants={fadeUp} className="flex justify-center lg:justify-start">
                <div className="inline-flex items-center gap-2.5 px-4 py-2 rounded-full border border-primary/[0.2] bg-primary/[0.05] text-primary text-[12px] font-semibold tracking-wide">
                  <span className="live-dot live-dot-primary" />
                  8 Autonomous Agents · Zero Configuration
                  <ChevronRight size={12} className="opacity-60" />
                </div>
              </motion.div>

              {/* Headline */}
              <motion.h1 variants={fadeUp} className="text-[clamp(2.8rem,6vw,5.2rem)] font-black leading-[0.92] tracking-[-0.04em]">
                <span className="gradient-text-hero">Your Autonomous</span>
                <br />
                <span className="text-text-primary">Data Scientist</span>
              </motion.h1>

              {/* Subtext */}
              <motion.p variants={fadeUp} className="text-[17px] md:text-[18px] text-text-secondary max-w-[560px] mx-auto lg:mx-0 leading-[1.75] font-light">
                Upload data. Get{" "}
                <span className="text-text-primary font-medium">insights, models, explainability, and reports</span>.
                No manual ML required.
              </motion.p>

              {/* CTAs */}
              <motion.div variants={fadeUp} className="flex items-center justify-center lg:justify-start gap-4 pt-1 flex-wrap">
                <Link
                  href={authOr("/free")}
                  className="group inline-flex items-center gap-2.5 px-7 py-3.5 rounded-2xl btn-primary text-[15px]"
                >
                  <Upload size={16} />
                  Start Free
                  <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform duration-300" />
                </Link>
                <Link
                  href={authOr("/enterprise")}
                  className="group inline-flex items-center gap-2.5 px-6 py-3.5 rounded-2xl border border-glass-border bg-glass text-text-secondary hover:text-text-primary hover:bg-glass-hover font-medium text-[14px] transition-all duration-300"
                >
                  <Terminal size={15} />
                  Enterprise Console
                </Link>
              </motion.div>

              {/* Honest, meaningful capability line */}
              <motion.div variants={fadeUp} className="flex items-center justify-center lg:justify-start gap-5 pt-1 text-[12px] text-text-muted flex-wrap">
                {["Auto-detects problem type", "Models & reports you can download", "No ML config required"].map((t) => (
                  <span key={t} className="flex items-center gap-1.5">
                    <CheckCircle2 size={11} className="text-success" />
                    {t}
                  </span>
                ))}
              </motion.div>

              {/* Real capability tiles (no vanity metrics) */}
              <motion.div variants={fadeUp} className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-6 max-w-[560px] mx-auto lg:mx-0">
                {[
                  { Icon: Cpu, label: "8 specialized agents", color: "text-primary" },
                  { Icon: Brain, label: "Auto model selection", color: "text-tertiary" },
                  { Icon: Eye, label: "SHAP explainability", color: "text-accent" },
                  { Icon: BarChart3, label: "One-click reports", color: "text-secondary" },
                ].map((s) => (
                  <div key={s.label} className="glass-sm p-3 flex flex-col gap-2">
                    <s.Icon size={16} className={s.color} strokeWidth={1.75} />
                    <div className="text-[11px] text-text-secondary font-medium leading-tight">{s.label}</div>
                  </div>
                ))}
              </motion.div>
            </motion.div>

            {/* ── Right: animated agent network ── */}
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3, duration: 1.1, ease: [0.25, 0.4, 0, 1] }}
              className="relative flex items-center justify-center"
            >
              <div className="absolute inset-0 bg-primary/[0.04] rounded-full blur-[120px] pointer-events-none" />
              <AgentNetwork size={580} className="relative" />
            </motion.div>
          </div>
        </div>

        {/* Scroll cue */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.4 }}
          className="absolute bottom-7 left-1/2 -translate-x-1/2 text-text-ghost"
        >
          <motion.div animate={{ y: [0, 6, 0] }} transition={{ duration: 1.8, repeat: Infinity }}>
            <ChevronDown size={18} />
          </motion.div>
        </motion.div>
      </section>

      {/* ═══════════════ PIPELINE ARCHITECTURE — SPLIT ═══════════════ */}
      <section className="relative py-36 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-accent/[0.012] to-transparent pointer-events-none" />
        <div className="max-w-[1280px] mx-auto px-8 relative z-10">

          <motion.div variants={stagger} initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-80px" }} className="text-center mb-20">
            <motion.span variants={fadeUp} className="text-[10px] font-semibold uppercase tracking-[4px] text-accent">Architecture</motion.span>
            <motion.h2 variants={fadeUp} className="text-4xl md:text-[3.5rem] font-bold tracking-[-0.03em] mt-4 gradient-text-hero leading-tight">
              Eight Agents. One Pipeline.
            </motion.h2>
            <motion.p variants={fadeUp} className="text-text-secondary text-lg mt-4 max-w-[500px] mx-auto font-light">
              Every stage is a specialized agent with its own toolset, context window, and decision logic.
            </motion.p>
          </motion.div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left: vertical agent stepper */}
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
              className="glass p-8 space-y-0.5"
            >
              {PIPELINE_STAGES.map((s, i) => (
                <div key={s.label} className="pro-agent-step py-3">
                  <div
                    className="pro-agent-dot text-void font-black text-[9px]"
                    style={{ background: s.color, boxShadow: `0 0 12px ${s.color}50` }}
                  >
                    {i + 1}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-0.5">
                      <s.icon size={12} style={{ color: s.color }} strokeWidth={2} />
                      <span className="text-[13px] font-semibold text-text-primary">{s.label}</span>
                    </div>
                    <p className="text-[12px] text-text-muted leading-snug">{s.desc}</p>
                  </div>
                </div>
              ))}
            </motion.div>

            {/* Right: live terminal */}
            <motion.div
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8, delay: 0.1 }}
              className="glass glow-accent border-glow flex flex-col"
            >
              <div className="terminal flex-1 rounded-[18px]">
                <div className="terminal-header">
                  <div className="terminal-dot bg-destructive/70" />
                  <div className="terminal-dot bg-warning/70" />
                  <div className="terminal-dot bg-success/70" />
                  <span className="text-[10px] text-text-muted ml-2 font-mono">axiom — run #a4e2f1</span>
                  <span className="ml-auto flex items-center gap-1.5 text-[10px] text-success">
                    <span className="w-1.5 h-1.5 rounded-full bg-success inline-block animate-pulse" />
                    live
                  </span>
                </div>
                <div className="p-6 space-y-1.5 text-[12px] font-mono leading-[1.85]">
                  <div className="text-text-ghost">$ axiom run --dataset churn_data.csv --target churn</div>
                  <div className="text-text-ghost">Initializing pipeline... ██████████ 100%</div>
                  <div className="h-px bg-glass-border my-3" />
                  <div className="text-success">✓ <span className="text-text-secondary">data_collection</span><span className="text-text-muted pl-2">2,847 rows · classification · 0.04s</span></div>
                  <div className="text-success">✓ <span className="text-text-secondary">preprocessing</span><span className="text-text-muted pl-2">quality 0.9891 · 0 nulls · 3 outliers · 0.09s</span></div>
                  <div className="text-success">✓ <span className="text-text-secondary">feature_engineering</span><span className="text-text-muted pl-2">18 → 47 features · PCA applied · 3.1s</span></div>
                  <div className="text-success">✓ <span className="text-text-secondary">data_splitting</span><span className="text-text-muted pl-2">train=2,277 val=285 test=285 · 0.01s</span></div>
                  <div className="text-success">✓ <span className="text-text-secondary">model_training</span><span className="text-text-muted pl-2">7 models · best: LGBMClassifier 0.931 · 5.8s</span></div>
                  <div className="text-success">✓ <span className="text-text-secondary">error_detection</span><span className="text-text-muted pl-2">2 warnings · 0 critical · 0.02s</span></div>
                  <div className="text-accent animate-pulse">● <span className="text-text-secondary">improvement</span><span className="text-text-muted pl-2">hyperparam search (iter 22/50) · Δ+0.8%...</span></div>
                  <div className="text-text-ghost">○ <span className="text-text-muted opacity-50">finalization</span><span className="text-text-ghost pl-2">waiting...</span></div>

                  <div className="mt-5 pt-4 border-t border-glass-border">
                    <div className="text-text-ghost text-[10px] mb-2 font-bold uppercase tracking-[1.5px]">Model Leaderboard</div>
                    <div className="space-y-2">
                      {[
                        { name: "LGBMClassifier", score: "0.931", pct: 93 },
                        { name: "RandomForest", score: "0.918", pct: 91 },
                        { name: "XGBoost", score: "0.904", pct: 90 },
                      ].map((m, i) => (
                        <div key={m.name} className="flex items-center gap-3">
                          {i === 0 && <span className="text-[9px] text-accent font-bold w-3">★</span>}
                          {i > 0 && <span className="text-[9px] text-text-ghost w-3">{i + 1}</span>}
                          <span className="text-[11px] text-text-secondary w-28 truncate">{m.name}</span>
                          <div className="flex-1 h-1 bg-glass-border rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full"
                              style={{ width: `${m.pct}%`, background: i === 0 ? "#00e5c8" : "rgba(255,255,255,0.15)" }}
                            />
                          </div>
                          <span className="text-[11px] font-mono w-10 text-right" style={{ color: i === 0 ? "#00e5c8" : "#505050" }}>{m.score}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* ═══════════════ FEATURES — BENTO GRID ═══════════════ */}
      <section className="relative py-36">
        <div className="max-w-[1280px] mx-auto px-8">
          <motion.div variants={stagger} initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-80px" }} className="text-center mb-20">
            <motion.span variants={fadeUp} className="text-[10px] font-semibold uppercase tracking-[4px] text-accent">Capabilities</motion.span>
            <motion.h2 variants={fadeUp} className="text-4xl md:text-[3.5rem] font-bold tracking-[-0.03em] mt-4 gradient-text-hero">
              Infrastructure That Thinks
            </motion.h2>
            <motion.p variants={fadeUp} className="text-text-secondary text-lg mt-4 max-w-[480px] mx-auto font-light">
              Every component is designed for autonomy, observability, and production-grade reliability.
            </motion.p>
          </motion.div>

          <motion.div
            variants={stagger}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-40px" }}
            className="grid grid-cols-1 md:grid-cols-3 gap-4"
          >
            {/* Large hero card — 2 columns */}
            <motion.div
              variants={fadeUp}
              className="md:col-span-2 glass p-8 group hover:border-accent/[0.14] hover:bg-accent/[0.015] transition-all duration-500 cursor-default"
            >
              <div className="flex items-start justify-between mb-6">
                <div className="w-12 h-12 rounded-2xl bg-accent/[0.07] border border-accent/[0.1] flex items-center justify-center group-hover:bg-accent/[0.13] transition-colors">
                  <Workflow size={22} className="text-accent" strokeWidth={1.5} />
                </div>
                <span className="text-[8px] font-bold uppercase tracking-[2.5px] text-text-ghost">AGENTS</span>
              </div>
              <h3 className="text-[20px] font-bold text-text-primary mb-3 tracking-tight">Agentic Orchestration</h3>
              <p className="text-[14px] text-text-muted leading-relaxed mb-6 max-w-[520px]">
                CrewAI-powered multi-agent system where eight specialized agents collaborate, share a unified pipeline state, make autonomous decisions, and execute the full ML lifecycle without human intervention.
              </p>
              {/* Mini agent flow */}
              <div className="flex items-center gap-2 overflow-x-auto pb-1 scrollbar-hide">
                {PIPELINE_STAGES.slice(0, 6).map((s, i) => (
                  <div key={s.label} className="flex items-center gap-2 shrink-0">
                    <div
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-glass-border bg-glass text-[10px] font-semibold whitespace-nowrap"
                      style={{ color: s.color }}
                    >
                      <s.icon size={10} strokeWidth={2} />
                      {s.label.split(" ")[0]}
                    </div>
                    {i < 5 && <div className="w-3 h-px bg-accent/20 shrink-0" />}
                  </div>
                ))}
                <span className="text-text-ghost text-[11px] pl-1 shrink-0">+2 more</span>
              </div>
            </motion.div>

            {/* Medium card */}
            <motion.div
              variants={fadeUp}
              className="glass p-7 group hover:border-info/[0.14] hover:bg-info/[0.01] transition-all duration-500 cursor-default"
            >
              <div className="flex items-center justify-between mb-5">
                <div className="w-10 h-10 rounded-xl bg-info/[0.07] border border-info/[0.1] flex items-center justify-center group-hover:bg-info/[0.13] transition-colors">
                  <Activity size={18} className="text-info" strokeWidth={1.5} />
                </div>
                <span className="text-[8px] font-bold uppercase tracking-[2.5px] text-text-ghost">TELEMETRY</span>
              </div>
              <h3 className="text-[15px] font-semibold text-text-primary mb-2 tracking-tight">Real-Time Observability</h3>
              <p className="text-[13px] text-text-muted leading-relaxed">
                Structured logs and live status updates across every agent execution with sub-second polling and full audit trail.
              </p>
            </motion.div>

            {/* Three small cards */}
            {[
              { icon: BarChart3, title: "Experiment Tracking", desc: "Every run logged with hyperparameters, metrics, and artifacts. Compare experiments visually.", tag: "TRACK", iconClass: "text-[#a78bfa]", bgClass: "bg-[rgba(167,139,250,0.07)]", borderClass: "border-[rgba(167,139,250,0.1)]", hoverClass: "hover:border-[rgba(167,139,250,0.2)] hover:bg-[rgba(167,139,250,0.01)]" },
              { icon: Shield, title: "Self-Healing Workflows", desc: "Automatic error detection, recovery, and retry logic with configurable fault tolerance per stage.", tag: "RESILIENCE", iconClass: "text-success", bgClass: "bg-success/[0.07]", borderClass: "border-success/[0.1]", hoverClass: "hover:border-success/[0.2] hover:bg-success/[0.01]" },
              { icon: Boxes, title: "Scalable Compute", desc: "Elastic compute with automatic scheduling and resource optimization across the full pipeline.", tag: "COMPUTE", iconClass: "text-warning", bgClass: "bg-warning/[0.07]", borderClass: "border-warning/[0.1]", hoverClass: "hover:border-warning/[0.2] hover:bg-warning/[0.01]" },
            ].map((f) => (
              <motion.div key={f.title} variants={fadeUp} className={`glass p-7 group ${f.hoverClass} transition-all duration-500 cursor-default`}>
                <div className="flex items-center justify-between mb-5">
                  <div className={`w-10 h-10 rounded-xl ${f.bgClass} border ${f.borderClass} flex items-center justify-center`}>
                    <f.icon size={18} className={f.iconClass} strokeWidth={1.5} />
                  </div>
                  <span className="text-[8px] font-bold uppercase tracking-[2.5px] text-text-ghost">{f.tag}</span>
                </div>
                <h3 className="text-[15px] font-semibold text-text-primary mb-2 tracking-tight">{f.title}</h3>
                <p className="text-[13px] text-text-muted leading-relaxed">{f.desc}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* ═══════════════ HOW IT WORKS ═══════════════ */}
      <section className="relative py-36 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-accent/[0.01] to-transparent pointer-events-none" />
        <div className="max-w-[1200px] mx-auto px-8 relative z-10">
          <motion.div variants={stagger} initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-80px" }} className="text-center mb-20">
            <motion.span variants={fadeUp} className="text-[10px] font-semibold uppercase tracking-[4px] text-accent">Workflow</motion.span>
            <motion.h2 variants={fadeUp} className="text-4xl md:text-[3.5rem] font-bold tracking-[-0.03em] mt-4 gradient-text-hero">
              Three Steps to Production
            </motion.h2>
          </motion.div>

          <motion.div variants={stagger} initial="hidden" whileInView="visible" viewport={{ once: true }} className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {STEPS.map((step, i) => (
              <motion.div key={step.n} variants={fadeUp} className="relative glass p-8 group hover:scale-[1.02] transition-all duration-500 overflow-hidden cursor-default">
                {/* Connector for non-last items */}
                {i < 2 && (
                  <div className="hidden md:block absolute top-1/2 -right-3 z-20 w-6 h-px bg-gradient-to-r from-glass-hover to-transparent" />
                )}
                {/* Big ghost number */}
                <div
                  className="absolute -top-4 -right-2 text-[110px] font-black leading-none tracking-[-0.06em] select-none pointer-events-none"
                  style={{ color: step.color, opacity: 0.06 }}
                >
                  {step.n}
                </div>
                {/* Icon */}
                <div
                  className="w-12 h-12 rounded-2xl flex items-center justify-center mb-6 relative z-10 transition-all duration-300 group-hover:scale-110"
                  style={{ background: `${step.color}10`, border: `1px solid ${step.color}25` }}
                >
                  <step.icon size={22} style={{ color: step.color }} strokeWidth={1.5} />
                </div>
                {/* Step number badge */}
                <div className="text-[10px] font-bold uppercase tracking-[3px] mb-3 relative z-10" style={{ color: step.color }}>
                  Step {step.n}
                </div>
                <h3 className="text-[18px] font-bold text-text-primary mb-3 tracking-tight relative z-10">{step.label}</h3>
                <p className="text-[13px] text-text-muted leading-relaxed relative z-10">{step.desc}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* ═══════════════ TECH MARQUEE ═══════════════ */}
      <section className="relative py-16 overflow-hidden border-y border-glass-border">
        <div className="absolute left-0 top-0 bottom-0 w-32 bg-gradient-to-r from-void to-transparent z-10 pointer-events-none" />
        <div className="absolute right-0 top-0 bottom-0 w-32 bg-gradient-to-l from-void to-transparent z-10 pointer-events-none" />
        <p className="text-center text-[10px] font-semibold uppercase tracking-[4px] text-text-muted mb-8">Built On</p>
        <div className="overflow-hidden">
          <motion.div
            animate={{ x: ["0%", "-50%"] }}
            transition={{ duration: 35, ease: "linear", repeat: Infinity }}
            className="flex gap-5 w-max"
          >
            {[...TECH, ...TECH].map((t, i) => (
              <div
                key={`${t}-${i}`}
                className="px-5 py-2.5 rounded-xl border border-glass-border bg-glass text-[12px] font-mono text-text-muted whitespace-nowrap shrink-0 hover:text-accent hover:border-accent/20 transition-all duration-300 cursor-default"
              >
                {t}
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* ═══════════════ CHOOSE YOUR PATH — FREE vs ENTERPRISE ═══════════════ */}
      <section className="relative py-36 overflow-hidden">
        {/* Split ambient backdrop — teal pooling on the left, indigo on the right */}
        <div className="absolute top-1/2 left-0 -translate-y-1/2 w-[640px] h-[640px] bg-accent/[0.04] rounded-full blur-[190px] pointer-events-none" />
        <div className="absolute top-1/2 right-0 -translate-y-1/2 w-[640px] h-[640px] bg-pro/[0.06] rounded-full blur-[190px] pointer-events-none" />

        <div className="max-w-[1180px] mx-auto px-8 relative z-10">
          <motion.div variants={stagger} initial="hidden" whileInView="visible" viewport={{ once: true }} className="text-center mb-16">
            <motion.span variants={fadeUp} className="text-[10px] font-semibold uppercase tracking-[4px] text-text-muted">Pricing</motion.span>
            <motion.h2 variants={fadeUp} className="text-4xl md:text-[3.5rem] font-bold tracking-[-0.04em] mt-3 gradient-text-hero">
              Two Ways to Run Axiom
            </motion.h2>
            <motion.p variants={fadeUp} className="text-text-secondary text-lg mt-4 font-light max-w-[480px] mx-auto">
              Start free in your browser — no account required. Or command the full enterprise platform.
            </motion.p>
          </motion.div>

          <motion.div variants={stagger} initial="hidden" whileInView="visible" viewport={{ once: true }} className="grid grid-cols-1 lg:grid-cols-[1fr_1.18fr] gap-6 items-stretch">

            {/* ───────────── FREE — flat, teal, minimal ───────────── */}
            <motion.div variants={fadeUp} className="glass p-9 flex flex-col relative hover:border-accent/[0.25] transition-colors duration-500">
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-accent/[0.1] border border-accent/[0.22] text-accent text-[10px] font-bold uppercase tracking-[2px] w-fit">
                <Sparkles size={11} />
                Free Forever
              </div>

              <div className="mt-7 flex items-end gap-2">
                <span className="text-[56px] font-black leading-none text-text-primary tracking-tighter">$0</span>
                <span className="text-text-muted text-sm mb-2.5">/ forever</span>
              </div>

              <h3 className="text-[22px] font-bold text-text-primary mt-5 tracking-[-0.02em]">Browser Quick-Start</h3>
              <p className="text-[14px] text-text-muted leading-relaxed mt-2 mb-7">
                Drop a CSV, watch eight agents run live, and download a trained model. No signup, no credit card.
              </p>

              <ul className="space-y-3 mb-9 flex-1">
                {[
                  "Full 8-agent autonomous pipeline",
                  "Classification, regression & clustering",
                  "Up to 10,000 rows per dataset",
                  "Downloadable model artifacts",
                ].map((f) => (
                  <li key={f} className="flex items-center gap-3 text-[13px] text-text-secondary">
                    <CheckCircle2 size={15} className="text-accent shrink-0" />
                    {f}
                  </li>
                ))}
              </ul>

              <Link
                href={authOr("/free")}
                className="group/btn inline-flex items-center justify-center gap-3 w-full px-7 py-3.5 rounded-xl bg-accent text-void font-bold text-[14px] hover:bg-accent-bright hover:shadow-lg hover:shadow-accent-glow transition-all duration-300"
              >
                <Upload size={16} />
                Start Free Now
                <ArrowRight size={15} className="group-hover/btn:translate-x-1 transition-transform" />
              </Link>
            </motion.div>

            {/* ───────────── ENTERPRISE — premium, indigo + gold, rich ───────────── */}
            <motion.div variants={fadeUp} className="pro-glass border-glow-pro p-9 flex flex-col relative overflow-hidden lg:scale-[1.025] shadow-2xl shadow-pro/10">
              {/* "Most Powerful" gold ribbon */}
              <div
                className="absolute top-7 -right-12 rotate-45 px-12 py-1.5 text-[10px] font-black uppercase tracking-[2px] text-void shadow-lg"
                style={{ background: "linear-gradient(135deg, #fbbf24, #f59e0b)" }}
              >
                Most Powerful
              </div>
              {/* Gold + indigo ambient wash */}
              <div className="absolute -top-24 -right-24 w-[340px] h-[340px] rounded-full pointer-events-none" style={{ background: "radial-gradient(circle, rgba(245,158,11,0.07) 0%, transparent 70%)" }} />
              <div className="absolute -bottom-20 -left-20 w-[300px] h-[300px] rounded-full pointer-events-none" style={{ background: "radial-gradient(circle, rgba(129,140,248,0.10) 0%, transparent 70%)" }} />

              <div className="relative z-10 flex flex-col flex-1">
                <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-pro/[0.14] border border-pro/[0.32] text-pro-bright text-[10px] font-bold uppercase tracking-[2px] w-fit">
                  <Crown size={11} />
                  Enterprise
                </div>

                <div className="mt-7 flex items-end gap-2">
                  <span className="text-[56px] font-black leading-none tracking-tighter gradient-text-hero">Custom</span>
                  <span className="text-text-muted text-sm mb-2.5">/ talk to sales</span>
                </div>

                <h3 className="text-[22px] font-bold text-text-primary mt-5 tracking-[-0.02em]">Axiom Pro Console</h3>
                <p className="text-[14px] text-text-muted leading-relaxed mt-2 mb-6">
                  The full command center — visual workflow builder, experiment tracking, multi-run comparison, governance, and deep analytics.
                </p>

                {/* Premium stat strip */}
                <div className="grid grid-cols-3 gap-2 mb-7">
                  {[
                    { icon: InfinityIcon, label: "Unlimited rows" },
                    { icon: Zap, label: "Priority compute" },
                    { icon: Lock, label: "SSO + RBAC" },
                  ].map((s) => (
                    <div key={s.label} className="rounded-xl border border-pro/[0.18] bg-pro/[0.05] px-3 py-3 text-center">
                      <s.icon size={16} className="text-pro-bright mx-auto mb-1.5" strokeWidth={2} />
                      <span className="text-[10px] font-semibold text-text-secondary leading-tight block">{s.label}</span>
                    </div>
                  ))}
                </div>

                <ul className="space-y-3 mb-9 flex-1">
                  {[
                    { f: "Visual workflow builder (drag & drop)", gold: false },
                    { f: "Parallel experiment management", gold: false },
                    { f: "Advanced model comparison & analytics", gold: true },
                    { f: "REST API access + artifact export", gold: true },
                  ].map(({ f, gold }) => (
                    <li key={f} className="flex items-center gap-3 text-[13px] text-text-secondary">
                      {gold ? (
                        <Star size={15} className="text-pro-gold-bright shrink-0" fill="currentColor" />
                      ) : (
                        <CheckCircle2 size={15} className="text-pro-bright shrink-0" />
                      )}
                      {f}
                    </li>
                  ))}
                </ul>

                <div className="flex flex-col sm:flex-row gap-3">
                  <Link
                    href={authOr("/enterprise")}
                    className="group/btn inline-flex items-center justify-center gap-3 flex-1 px-7 py-3.5 rounded-xl font-bold text-[14px] text-void hover:shadow-lg hover:shadow-pro-glow transition-all duration-300"
                    style={{ background: "linear-gradient(135deg, #a78bfa, #818cf8)" }}
                  >
                    <Terminal size={16} />
                    Open Console
                    <ArrowRight size={15} className="group-hover/btn:translate-x-1 transition-transform" />
                  </Link>
                  <Link
                    href={authOr("/enterprise")}
                    className="inline-flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl font-semibold text-[13px] border border-pro/[0.3] text-pro-bright hover:bg-pro/[0.1] transition-all duration-300"
                  >
                    Sign In
                  </Link>
                </div>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* ═══════════════ FOOTER ═══════════════ */}
      <footer className="border-t border-glass-border py-12">
        <div className="max-w-[1200px] mx-auto px-8">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center gap-3">
              <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-accent to-accent-muted flex items-center justify-center">
                <span className="text-void font-black text-[10px]">A</span>
              </div>
              <span className="text-[13px] text-text-muted font-semibold tracking-wide">Axiom</span>
            </div>
            <span className="text-[11px] text-text-ghost font-mono">Autonomous ML Infrastructure</span>
            <div className="flex items-center gap-6 text-[11px] text-text-muted">
              <Link href={authOr("/free")} className="hover:text-accent transition-colors">Free</Link>
              <Link href={authOr("/enterprise")} className="hover:text-tertiary transition-colors">Enterprise</Link>
            </div>
          </div>
        </div>
      </footer>
    </>
  );
}
