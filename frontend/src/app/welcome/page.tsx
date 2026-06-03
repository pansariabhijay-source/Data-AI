"use client";

import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import {
  Sparkles, Command, ArrowRight, Check, GraduationCap,
  LineChart, FlaskConical, Users, Building2, SlidersHorizontal,
} from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import { useAuthStore } from "@/store/useAuthStore";
import { setWorkspaceMode } from "@/lib/api";
import type { Mode } from "@/lib/types";

const MODES: {
  id: Mode;
  name: string;
  tagline: string;
  accent: string;
  glow: string;
  Icon: typeof Sparkles;
  audience: { Icon: typeof Sparkles; label: string }[];
  features: string[];
}[] = [
  {
    id: "free",
    name: "Free",
    tagline: "Simple autonomous pipeline",
    accent: "#2dd4bf",
    glow: "rgba(45,212,191,0.18)",
    Icon: Sparkles,
    audience: [
      { Icon: GraduationCap, label: "Students" },
      { Icon: LineChart, label: "Analysts" },
      { Icon: FlaskConical, label: "Quick experiments" },
    ],
    features: ["Upload → run → results in one flow", "Eight agents, zero configuration", "Downloadable models & reports"],
  },
  {
    id: "enterprise",
    name: "Enterprise",
    tagline: "Advanced control",
    accent: "#818cf8",
    glow: "rgba(129,140,248,0.2)",
    Icon: Command,
    audience: [
      { Icon: Users, label: "Teams" },
      { Icon: FlaskConical, label: "Researchers" },
      { Icon: Building2, label: "Enterprises" },
    ],
    features: ["Agent console & live execution", "Visual workflow builder", "Full run history & reports"],
  },
];

export default function WelcomePage() {
  const router = useRouter();
  const selectMode = useAppStore((s) => s.selectMode);
  const user = useAuthStore((s) => s.user);

  const choose = (mode: Mode) => {
    selectMode(mode);
    // Persist the choice durably; never block navigation on it.
    setWorkspaceMode(mode).catch(() => {});
    router.push(mode === "free" ? "/free" : "/enterprise");
  };

  return (
    <main className="min-h-screen flex flex-col items-center justify-center px-6 py-16 relative overflow-hidden">
      <div className="aurora" />

      <motion.div
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: [0.25, 0.4, 0, 1] }}
        className="text-center mb-12 relative z-10"
      >
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-glass-border bg-glass text-text-muted text-[11px] font-medium tracking-wide mb-5">
          <span className="live-dot live-dot-primary" />
          Workspace ready
        </div>
        <h1 className="text-[clamp(2rem,4.5vw,3.2rem)] font-bold tracking-[-0.035em] text-text-primary">
          Welcome{user?.username ? <>, <span className="gradient-text">{user.username}</span></> : ""}.
        </h1>
        <p className="text-text-secondary text-[15px] mt-3 max-w-[440px] mx-auto leading-relaxed">
          Choose how you want to work with Axiom. You can switch anytime from Settings.
        </p>
      </motion.div>

      <div className="grid md:grid-cols-2 gap-5 w-full max-w-[860px] relative z-10">
        {MODES.map((m, i) => (
          <motion.button
            key={m.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 + i * 0.1, duration: 0.6, ease: [0.25, 0.4, 0, 1] }}
            whileHover={{ y: -4 }}
            onClick={() => choose(m.id)}
            className="group relative text-left rounded-3xl p-7 overflow-hidden transition-colors duration-300"
            style={{ background: "rgba(255,255,255,0.02)", border: "1px solid var(--color-glass-border)" }}
          >
            {/* hover wash */}
            <div
              className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"
              style={{ background: `radial-gradient(ellipse 90% 70% at 30% 0%, ${m.glow}, transparent 70%)` }}
            />
            <div
              className="absolute inset-x-0 -top-px h-px opacity-60"
              style={{ background: `linear-gradient(90deg, transparent, ${m.accent}, transparent)` }}
            />

            <div className="relative z-10">
              <div className="flex items-center justify-between mb-6">
                <div
                  className="w-12 h-12 rounded-2xl grid place-items-center"
                  style={{ background: `${m.accent}18`, border: `1px solid ${m.accent}40` }}
                >
                  <m.Icon size={22} style={{ color: m.accent }} strokeWidth={1.75} />
                </div>
                <span
                  className="w-9 h-9 rounded-full grid place-items-center border border-glass-border text-text-muted group-hover:text-text-primary transition-all duration-300 group-hover:translate-x-0.5"
                  style={{ background: "rgba(255,255,255,0.02)" }}
                >
                  <ArrowRight size={16} />
                </span>
              </div>

              <h2 className="text-[22px] font-bold text-text-primary tracking-[-0.02em]">{m.name}</h2>
              <p className="text-[13px] font-medium mt-0.5" style={{ color: m.accent }}>{m.tagline}</p>

              {/* audience */}
              <div className="flex flex-wrap gap-2 mt-5">
                {m.audience.map((a) => (
                  <span key={a.label} className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-glass border border-glass-border text-[11px] text-text-secondary">
                    <a.Icon size={11} style={{ color: m.accent }} />
                    {a.label}
                  </span>
                ))}
              </div>

              {/* features */}
              <ul className="mt-6 space-y-2.5">
                {m.features.map((f) => (
                  <li key={f} className="flex items-center gap-2.5 text-[13px] text-text-secondary">
                    <Check size={14} style={{ color: m.accent }} className="shrink-0" />
                    {f}
                  </li>
                ))}
              </ul>
            </div>
          </motion.button>
        ))}
      </div>

      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="text-text-ghost text-[12px] mt-10 flex items-center gap-2 relative z-10"
      >
        <SlidersHorizontal size={12} />
        Same account · switch modes whenever you need
      </motion.p>
    </main>
  );
}
