"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Bell, Settings, Search, LogOut, User as UserIcon, ChevronDown } from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import { useAuthStore } from "@/store/useAuthStore";

function Brand({ workspace }: { workspace?: string }) {
  return (
    <Link href="/" className="flex items-center gap-2.5 group">
      <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary to-primary-container flex items-center justify-center shadow-[0_4px_18px_rgba(79,70,229,0.35)]">
        <span className="text-white font-bold text-sm">A</span>
      </div>
      <span className="text-[15px] font-semibold tracking-tight text-text-primary font-display">Axiom</span>
      {workspace && (
        <>
          <span className="text-text-ghost">/</span>
          <span className="text-[13px] font-medium text-text-secondary">{workspace}</span>
        </>
      )}
    </Link>
  );
}

function UserCluster() {
  const { user, logout } = useAuthStore();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  return (
    <div className="flex items-center gap-1">
      <button title="Notifications" className="relative p-2 rounded-lg text-text-muted hover:text-text-primary hover:bg-white/5 transition-all">
        <Bell size={17} strokeWidth={1.75} />
        <span className="absolute top-1.5 right-1.5 w-1.5 h-1.5 rounded-full bg-accent" />
      </button>
      <button
        onClick={() => router.push("/enterprise/settings")}
        title="Settings"
        className="p-2 rounded-lg text-text-muted hover:text-text-primary hover:bg-white/5 transition-all"
      >
        <Settings size={17} strokeWidth={1.75} />
      </button>

      <div ref={ref} className="relative ml-1">
        <button
          onClick={() => setOpen((o) => !o)}
          className="flex items-center gap-1.5 rounded-full pl-0.5 pr-1.5 py-0.5 hover:bg-white/5 transition-all"
        >
          <span className="h-8 w-8 rounded-full border border-primary/40 bg-primary/10 flex items-center justify-center">
            <span className="text-primary font-semibold text-sm">
              {user?.username?.charAt(0).toUpperCase() ?? "U"}
            </span>
          </span>
          <ChevronDown size={13} className={`text-text-muted transition-transform ${open ? "rotate-180" : ""}`} />
        </button>

        <AnimatePresence>
          {open && (
            <motion.div
              initial={{ opacity: 0, y: -6, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -6, scale: 0.98 }}
              transition={{ duration: 0.16, ease: [0.25, 0.4, 0, 1] }}
              className="absolute right-0 mt-2 w-60 rounded-2xl border border-white/[0.08] bg-void/95 backdrop-blur-2xl shadow-2xl shadow-black/40 overflow-hidden"
            >
              <div className="px-4 py-3.5 border-b border-white/[0.06]">
                <p className="text-[13px] font-semibold text-text-primary truncate">{user?.username ?? "User"}</p>
                <p className="text-[11px] text-text-muted truncate">{user?.email ?? ""}</p>
              </div>
              <div className="p-1.5">
                <button
                  onClick={() => { setOpen(false); router.push("/enterprise/settings"); }}
                  className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-[13px] text-text-secondary hover:text-text-primary hover:bg-white/5 transition-all"
                >
                  <UserIcon size={15} /> Account & preferences
                </button>
                <button
                  onClick={() => { setOpen(false); logout(); router.replace("/auth"); }}
                  className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-[13px] text-text-secondary hover:text-destructive hover:bg-destructive/8 transition-all"
                >
                  <LogOut size={15} /> Sign out
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

export default function Navbar() {
  const { mode } = useAppStore();
  const { isAuthenticated, user } = useAuthStore();
  const pathname = usePathname();
  const router = useRouter();

  const isLanding = pathname === "/";
  const isEnterprise = mode === "enterprise";
  const workspace = user ? `${user.username}` : undefined;

  // ── Enterprise console top bar (left sidebar owns navigation) ──────────────
  if (isAuthenticated && isEnterprise && !isLanding) {
    return (
      <header className="fixed top-0 left-0 w-full z-50 flex justify-between items-center px-6 lg:px-8 h-16 bg-void/70 backdrop-blur-xl border-b border-white/[0.06]">
        <Brand workspace={workspace} />
        <div className="flex items-center gap-3">
          <button className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg border border-glass-border bg-glass text-text-muted hover:text-text-secondary text-[12px] transition-all w-56">
            <Search size={14} />
            <span>Search runs, reports…</span>
            <kbd className="ml-auto text-[10px] font-mono px-1.5 py-0.5 rounded bg-white/5 border border-white/10">⌘K</kbd>
          </button>
          <UserCluster />
        </div>
      </header>
    );
  }

  // ── Landing + Free top bar ─────────────────────────────────────────────────
  return (
    <motion.header
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: [0.25, 0.4, 0, 1] }}
      className="fixed top-0 left-0 right-0 z-50 backdrop-blur-2xl bg-void/60 border-b border-white/8"
    >
      <div className="max-w-[1340px] mx-auto px-6 lg:px-8 h-16 flex items-center justify-between">
        <Brand workspace={!isLanding ? workspace : undefined} />

        <div className="flex items-center gap-2.5">
          {isAuthenticated ? (
            <UserCluster />
          ) : (
            <>
              <Link
                href="/auth"
                className="px-4 py-2 rounded-xl text-[13px] font-medium text-text-secondary hover:text-text-primary transition-colors"
              >
                Sign in
              </Link>
              <button
                onClick={() => router.push("/auth")}
                className="btn-primary text-[13px]"
              >
                Get started
              </button>
            </>
          )}
        </div>
      </div>
    </motion.header>
  );
}
