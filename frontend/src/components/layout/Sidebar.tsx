"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  LayoutGrid,
  History,
  FileText,
  Cpu,
  Workflow,
  Settings,
  ChevronLeft,
  ChevronRight,
  BarChart3,
} from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import { useAuthStore } from "@/store/useAuthStore";

type NavItem = {
  href: string;
  label: string;
  icon: typeof LayoutGrid;
  /** Restrict to a specific mode. Undefined = visible in every mode. */
  mode?: "enterprise";
};

// Information architecture per the product brief — six surfaces, nothing more.
const NAV_ITEMS: NavItem[] = [
  { href: "/enterprise", label: "Home", icon: LayoutGrid },
  { href: "/enterprise/runs", label: "Runs", icon: History },
  { href: "/enterprise/reports", label: "Reports", icon: FileText },
  { href: "/enterprise/visualizations", label: "Visualizations", icon: BarChart3 },
  { href: "/enterprise/agents", label: "Agents", icon: Cpu },
  { href: "/enterprise/workflow", label: "Workflow Builder", icon: Workflow, mode: "enterprise" },
  { href: "/enterprise/settings", label: "Settings", icon: Settings },
];

export default function Sidebar() {
  const pathname = usePathname();
  const { sidebarCollapsed, setSidebarCollapsed, mode } = useAppStore();
  const user = useAuthStore((s) => s.user);

  const items = NAV_ITEMS.filter((i) => !i.mode || i.mode === mode);

  return (
    <motion.aside
      initial={false}
      animate={{ width: sidebarCollapsed ? 72 : 248 }}
      transition={{ duration: 0.3, ease: [0.25, 0.4, 0, 1] }}
      className="fixed left-0 top-0 bottom-0 z-40 flex flex-col bg-void/50 backdrop-blur-2xl border-r border-white/[0.06]"
      style={{ paddingTop: 76 }}
    >
      {/* Workspace label */}
      <div className={`mb-6 ${sidebarCollapsed ? "px-0" : "px-5"}`}>
        <AnimatePresence mode="wait">
          {!sidebarCollapsed && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.18 }}
            >
              <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-text-ghost">
                Workspace
              </p>
              <p className="mt-1 text-[13px] font-semibold text-text-primary truncate">
                {user ? `${user.username}'s workspace` : "Axiom"}
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-0.5">
        {items.map((item) => {
          const isActive =
            pathname === item.href ||
            (item.href !== "/enterprise" && pathname.startsWith(item.href));
          const Icon = item.icon;

          return (
            <Link
              key={item.href}
              href={item.href}
              className={`sidebar-nav-item ${isActive ? "sidebar-nav-active" : ""} ${
                sidebarCollapsed ? "justify-center !mx-2 !px-0" : ""
              }`}
              title={sidebarCollapsed ? item.label : undefined}
            >
              <Icon
                size={18}
                strokeWidth={1.75}
                className={`shrink-0 ${isActive ? "text-text-primary" : ""}`}
              />
              <AnimatePresence>
                {!sidebarCollapsed && (
                  <motion.span
                    initial={{ opacity: 0, width: 0 }}
                    animate={{ opacity: 1, width: "auto" }}
                    exit={{ opacity: 0, width: 0 }}
                    transition={{ duration: 0.15 }}
                    className="overflow-hidden whitespace-nowrap"
                  >
                    {item.label}
                  </motion.span>
                )}
              </AnimatePresence>
            </Link>
          );
        })}
      </nav>

      {/* Footer: identity + collapse — no upsell, no template cruft */}
      <div className="mt-auto pb-5">
        <AnimatePresence>
          {!sidebarCollapsed && user && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="mx-3 mb-3 flex items-center gap-3 rounded-xl border border-white/[0.06] bg-white/[0.02] px-3 py-2.5"
            >
              <div className="h-8 w-8 shrink-0 rounded-full bg-primary/12 border border-primary/30 flex items-center justify-center">
                <span className="text-primary font-semibold text-[13px]">
                  {user.username?.charAt(0).toUpperCase() ?? "U"}
                </span>
              </div>
              <div className="min-w-0">
                <p className="text-[12px] font-medium text-text-primary truncate">{user.username}</p>
                <p className="text-[10px] text-text-muted truncate">Enterprise · active</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <button
          onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-xl text-text-ghost hover:text-text-secondary hover:bg-white/[0.04] transition-all duration-200"
        >
          {sidebarCollapsed ? <ChevronRight size={15} /> : <ChevronLeft size={15} />}
          <AnimatePresence>
            {!sidebarCollapsed && (
              <motion.span
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="text-[11px] font-medium"
              >
                Collapse
              </motion.span>
            )}
          </AnimatePresence>
        </button>
      </div>
    </motion.aside>
  );
}
