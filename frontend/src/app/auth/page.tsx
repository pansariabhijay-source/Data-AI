"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { useAuthStore } from "@/store/useAuthStore";
import { useAppStore } from "@/store/useAppStore";
import { authLogin, authSignup, getWorkspace } from "@/lib/api";
import AgentNetwork from "@/components/ui/AgentNetwork";
import {
  ArrowRight,
  Loader2,
  Eye,
  EyeOff,
  Mail,
  Lock,
  User,
  ShieldCheck,
  Workflow,
  FileBarChart,
} from "lucide-react";

const VALUE_PROPS = [
  { Icon: Workflow, label: "Eight autonomous agents, end to end" },
  { Icon: FileBarChart, label: "Models, explainability & reports" },
  { Icon: ShieldCheck, label: "Your data, your workspace, persisted" },
];

export default function AuthPage() {
  const router = useRouter();
  const setAuth = useAuthStore((s) => s.setAuth);
  const selectMode = useAppStore((s) => s.selectMode);

  const [isLogin, setIsLogin] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showPassword, setShowPassword] = useState(false);

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [username, setUsername] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    // If the proxy gate bounced the user here from a protected page, send them
    // back exactly where they were headed once authenticated. Read it straight
    // from the URL (in the handler) to avoid a Suspense-bound useSearchParams.
    const rawNext = new URLSearchParams(window.location.search).get("next");
    const safeNext = rawNext && rawNext.startsWith("/") && !rawNext.startsWith("//") ? rawNext : null;

    try {
      if (isLogin) {
        const data = await authLogin(email, password);
        setAuth(data.user, data.token);
        if (safeNext) {
          router.replace(safeNext);
          return;
        }
        // Returning users still resume their last workspace silently in the
        // background (so the navbar/store know it), but the flow lands them on
        // the landing page first — they choose where to go from there.
        try {
          const prefs = await getWorkspace();
          if (prefs.selected_mode) selectMode(prefs.selected_mode);
        } catch {
          /* preferences are best-effort */
        }
        router.replace("/");
      } else {
        const data = await authSignup(email, username, password);
        setAuth(data.user, data.token);
        // New accounts arrive on the landing page (unless deep-linked).
        router.replace(safeNext ?? "/");
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Authentication failed");
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen grid lg:grid-cols-[1.05fr_1fr] relative overflow-hidden">
      <div className="aurora" />

      {/* ── Brand / value panel (desktop) ──────────────────────────────────── */}
      <aside className="relative hidden lg:flex flex-col justify-between px-14 py-12 border-r border-white/[0.06] overflow-hidden">
        <div
          className="absolute inset-0 pointer-events-none"
          style={{ background: "radial-gradient(ellipse 80% 60% at 30% 10%, rgba(99,102,241,0.12), transparent 70%)" }}
        />

        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.25, 0.4, 0, 1] }}
          className="relative z-10 flex items-center gap-2.5"
        >
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary to-primary-container flex items-center justify-center shadow-[0_6px_24px_rgba(99,102,241,0.4)]">
            <span className="text-white font-bold">A</span>
          </div>
          <span className="text-[17px] font-semibold tracking-tight text-text-primary font-display">Axiom</span>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.94 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.1, ease: [0.25, 0.4, 0, 1] }}
          className="relative z-10 grid place-items-center my-6"
        >
          <AgentNetwork size={420} />
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2, ease: [0.25, 0.4, 0, 1] }}
          className="relative z-10"
        >
          <h2 className="font-display text-[36px] tracking-[-0.01em] text-text-primary leading-[1.05]">
            Your autonomous
            <br />
            <span className="gradient-text">data scientist.</span>
          </h2>
          <ul className="mt-6 space-y-3">
            {VALUE_PROPS.map(({ Icon, label }) => (
              <li key={label} className="flex items-center gap-3 text-[13.5px] text-text-secondary">
                <span className="w-7 h-7 shrink-0 rounded-lg bg-primary/10 border border-primary/25 grid place-items-center">
                  <Icon size={14} className="text-primary-dim" strokeWidth={1.75} />
                </span>
                {label}
              </li>
            ))}
          </ul>
        </motion.div>
      </aside>

      {/* ── Auth card ──────────────────────────────────────────────────────── */}
      <main className="relative z-10 flex items-center justify-center px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.25, 0.4, 0, 1] }}
          className="w-full max-w-[400px]"
        >
          {/* Mobile brand */}
          <div className="lg:hidden flex items-center gap-2.5 mb-10">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary to-primary-container flex items-center justify-center">
              <span className="text-white font-bold">A</span>
            </div>
            <span className="text-[17px] font-semibold tracking-tight text-text-primary font-display">Axiom</span>
          </div>

          <h1 className="font-display text-[34px] tracking-[-0.01em] leading-tight text-text-primary">
            {isLogin ? "Welcome back" : "Create your workspace"}
          </h1>
          <p className="text-text-secondary text-[14px] mt-1.5">
            {isLogin ? "Sign in to pick up where you left off." : "Start building autonomous ML pipelines in minutes."}
          </p>

          {/* Segmented toggle */}
          <div className="mt-7 relative grid grid-cols-2 p-1 rounded-xl bg-white/[0.03] border border-white/[0.07]">
            <motion.div
              className="absolute inset-y-1 w-[calc(50%-4px)] rounded-lg bg-white/[0.07] border border-white/[0.08]"
              animate={{ left: isLogin ? 4 : "calc(50% + 0px)" }}
              transition={{ type: "spring", stiffness: 380, damping: 32 }}
            />
            {[
              { id: true, label: "Sign in" },
              { id: false, label: "Sign up" },
            ].map((t) => (
              <button
                key={t.label}
                onClick={() => { setIsLogin(t.id); setError(null); }}
                className={`relative z-10 py-2 text-[13px] font-semibold rounded-lg transition-colors ${
                  isLogin === t.id ? "text-text-primary" : "text-text-muted hover:text-text-secondary"
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>

          <form onSubmit={handleSubmit} className="mt-6 flex flex-col gap-4">
            <AnimatePresence mode="popLayout">
              {!isLogin && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.2 }}
                >
                  <Field
                    icon={<User size={16} />}
                    type="text"
                    label="Username"
                    placeholder="ada_lovelace"
                    value={username}
                    onChange={setUsername}
                    autoComplete="username"
                  />
                </motion.div>
              )}
            </AnimatePresence>

            <Field
              icon={<Mail size={16} />}
              type="email"
              label="Email"
              placeholder="you@company.com"
              value={email}
              onChange={setEmail}
              autoComplete="email"
            />

            <Field
              icon={<Lock size={16} />}
              type={showPassword ? "text" : "password"}
              label="Password"
              placeholder="••••••••"
              value={password}
              onChange={setPassword}
              autoComplete={isLogin ? "current-password" : "new-password"}
              trailing={
                <button
                  type="button"
                  onClick={() => setShowPassword((s) => !s)}
                  className="text-text-muted hover:text-text-secondary transition-colors"
                  tabIndex={-1}
                >
                  {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              }
            />

            <AnimatePresence>
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -4 }}
                  className="text-destructive text-[12.5px] bg-destructive-muted border border-destructive/20 py-2.5 px-3.5 rounded-lg"
                >
                  {error}
                </motion.div>
              )}
            </AnimatePresence>

            <button
              type="submit"
              disabled={loading}
              className="btn-primary mt-1 w-full !py-3 text-[14px] flex items-center justify-center gap-2.5 group disabled:opacity-60 disabled:cursor-not-allowed"
            >
              {loading ? (
                <Loader2 className="animate-spin" size={18} />
              ) : (
                <>
                  {isLogin ? "Sign in" : "Create account"}
                  <ArrowRight size={17} className="transition-transform group-hover:translate-x-0.5" />
                </>
              )}
            </button>
          </form>

          <p className="mt-6 text-center text-[12px] text-text-muted">
            {isLogin ? "New to Axiom? " : "Already have an account? "}
            <button
              onClick={() => { setIsLogin((v) => !v); setError(null); }}
              className="text-primary-dim hover:text-primary-fixed font-medium transition-colors"
            >
              {isLogin ? "Create a workspace" : "Sign in"}
            </button>
          </p>
        </motion.div>
      </main>
    </div>
  );
}

function Field({
  icon,
  label,
  type,
  placeholder,
  value,
  onChange,
  autoComplete,
  trailing,
}: {
  icon: React.ReactNode;
  label: string;
  type: string;
  placeholder: string;
  value: string;
  onChange: (v: string) => void;
  autoComplete?: string;
  trailing?: React.ReactNode;
}) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-[11.5px] font-medium text-text-muted ml-0.5">{label}</span>
      <div className="relative flex items-center rounded-xl bg-white/[0.025] border border-white/[0.08] focus-within:border-primary/50 focus-within:bg-white/[0.04] transition-colors">
        <span className="pl-3.5 text-text-muted">{icon}</span>
        <input
          type={type}
          required
          placeholder={placeholder}
          value={value}
          autoComplete={autoComplete}
          onChange={(e) => onChange(e.target.value)}
          className="flex-1 bg-transparent border-0 focus:ring-0 focus:outline-none py-3 px-3 text-[14px] text-text-primary placeholder:text-text-ghost"
        />
        {trailing && <span className="pr-3.5">{trailing}</span>}
      </div>
    </label>
  );
}
