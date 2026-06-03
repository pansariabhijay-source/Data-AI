"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import {
  User as UserIcon, Mail, Calendar, Sparkles, Command, Check,
  LogOut, ArrowRight, Shield, Bell, Moon,
} from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import { useAuthStore } from "@/store/useAuthStore";
import { setWorkspaceMode } from "@/lib/api";
import type { Mode } from "@/lib/types";

function Section({ title, desc, children }: { title: string; desc?: string; children: React.ReactNode }) {
  return (
    <section className="glass-panel rounded-2xl p-6 md:p-7">
      <div className="mb-5">
        <h2 className="text-[16px] font-semibold text-text-primary tracking-tight">{title}</h2>
        {desc && <p className="text-[13px] text-text-muted mt-0.5">{desc}</p>}
      </div>
      {children}
    </section>
  );
}

const MODES: { id: Mode; name: string; tagline: string; Icon: typeof Sparkles; accent: string }[] = [
  { id: "free", name: "Free", tagline: "Focused upload → run → results", Icon: Sparkles, accent: "#2dd4bf" },
  { id: "enterprise", name: "Enterprise", tagline: "Agents, workflows & run history", Icon: Command, accent: "#818cf8" },
];

export default function SettingsPage() {
  const router = useRouter();
  const { mode, selectMode } = useAppStore();
  const { user, logout } = useAuthStore();
  const [switching, setSwitching] = useState<Mode | null>(null);

  const choose = async (next: Mode) => {
    if (next === mode || switching) return;
    setSwitching(next);
    selectMode(next);
    // Best-effort persistence — UI never blocks on it.
    setWorkspaceMode(next).catch(() => {});
    router.push(next === "free" ? "/free" : "/enterprise");
  };

  const joined = user?.created_at
    ? new Date(user.created_at).toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" })
    : "—";

  return (
    <div className="min-h-screen px-6 md:px-10 py-10 max-w-[840px]">
      <motion.header
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="mb-8"
      >
        <h1 className="font-display text-3xl md:text-4xl font-bold text-text-primary tracking-tight">Settings</h1>
        <p className="text-sm text-text-secondary mt-1">Manage your account, workspace mode, and preferences.</p>
      </motion.header>

      <div className="space-y-5">
        {/* Profile */}
        <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.05 }}>
          <Section title="Profile" desc="Your Axiom identity.">
            <div className="flex items-center gap-4 pb-5 mb-5 border-b border-white/[0.06]">
              <div className="h-14 w-14 rounded-2xl bg-primary/12 border border-primary/30 flex items-center justify-center">
                <span className="text-primary font-bold text-xl">{user?.username?.charAt(0).toUpperCase() ?? "U"}</span>
              </div>
              <div>
                <p className="text-[15px] font-semibold text-text-primary">{user?.username ?? "User"}</p>
                <p className="text-[12px] text-text-muted">Member</p>
              </div>
            </div>
            <dl className="grid sm:grid-cols-2 gap-4">
              {[
                { Icon: UserIcon, label: "Username", value: user?.username ?? "—" },
                { Icon: Mail, label: "Email", value: user?.email ?? "—" },
                { Icon: Calendar, label: "Joined", value: joined },
                { Icon: Shield, label: "Account", value: "Standard" },
              ].map((f) => (
                <div key={f.label} className="flex items-center gap-3">
                  <div className="h-9 w-9 rounded-lg bg-white/[0.03] border border-white/[0.06] flex items-center justify-center shrink-0">
                    <f.Icon size={15} className="text-text-muted" />
                  </div>
                  <div className="min-w-0">
                    <dt className="text-[10px] uppercase tracking-wider text-text-ghost font-semibold">{f.label}</dt>
                    <dd className="text-[13px] text-text-primary truncate">{f.value}</dd>
                  </div>
                </div>
              ))}
            </dl>
          </Section>
        </motion.div>

        {/* Workspace mode */}
        <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.1 }}>
          <Section title="Workspace mode" desc="Switch how you work with Axiom. Your data and runs stay with you.">
            <div className="grid sm:grid-cols-2 gap-3">
              {MODES.map((m) => {
                const active = m.id === mode;
                return (
                  <button
                    key={m.id}
                    onClick={() => choose(m.id)}
                    disabled={switching !== null}
                    className="group relative text-left rounded-2xl p-5 border transition-all duration-300 disabled:opacity-60"
                    style={{
                      borderColor: active ? `${m.accent}55` : "rgba(255,255,255,0.06)",
                      background: active ? `${m.accent}0d` : "rgba(255,255,255,0.02)",
                    }}
                  >
                    <div className="flex items-center justify-between mb-3">
                      <div className="h-10 w-10 rounded-xl grid place-items-center" style={{ background: `${m.accent}1a`, border: `1px solid ${m.accent}40` }}>
                        <m.Icon size={18} style={{ color: m.accent }} strokeWidth={1.75} />
                      </div>
                      {active ? (
                        <span className="inline-flex items-center gap-1 text-[11px] font-semibold" style={{ color: m.accent }}>
                          <Check size={13} /> Current
                        </span>
                      ) : (
                        <ArrowRight size={15} className="text-text-ghost group-hover:text-text-secondary group-hover:translate-x-0.5 transition-all" />
                      )}
                    </div>
                    <p className="text-[15px] font-semibold text-text-primary">{m.name}</p>
                    <p className="text-[12px] text-text-muted mt-0.5">{m.tagline}</p>
                  </button>
                );
              })}
            </div>
          </Section>
        </motion.div>

        {/* Preferences */}
        <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.15 }}>
          <Section title="Preferences">
            <ul className="divide-y divide-white/[0.05]">
              {[
                { Icon: Moon, label: "Theme", value: "Dark", hint: "Axiom is dark-first" },
                { Icon: Bell, label: "Run notifications", value: "On", hint: "Notify when a pipeline finishes" },
              ].map((p) => (
                <li key={p.label} className="flex items-center justify-between py-3.5 first:pt-0 last:pb-0">
                  <span className="flex items-center gap-3">
                    <p.Icon size={16} className="text-text-muted" />
                    <span>
                      <span className="text-[13px] text-text-primary font-medium block">{p.label}</span>
                      <span className="text-[11px] text-text-ghost">{p.hint}</span>
                    </span>
                  </span>
                  <span className="text-[12px] font-medium px-2.5 py-1 rounded-lg bg-white/[0.04] border border-white/[0.06] text-text-secondary">{p.value}</span>
                </li>
              ))}
            </ul>
          </Section>
        </motion.div>

        {/* Danger / session */}
        <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.2 }}>
          <Section title="Session">
            <button
              onClick={() => { logout(); router.replace("/auth"); }}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl border border-destructive/25 text-destructive text-[13px] font-medium hover:bg-destructive/10 transition-all"
            >
              <LogOut size={15} /> Sign out
            </button>
          </Section>
        </motion.div>
      </div>
    </div>
  );
}
