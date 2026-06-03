"use client";

import { motion } from "framer-motion";
import { Sparkles, Crown } from "lucide-react";
import { useAppStore } from "@/store/useAppStore";

export default function ModeSwitch() {
  const { mode, toggleMode } = useAppStore();
  const isEnterprise = mode === "enterprise";

  return (
    <button
      onClick={toggleMode}
      className={`relative flex items-center gap-1 p-1 rounded-full border backdrop-blur-xl transition-all duration-300 hover:scale-[1.02] ${
        isEnterprise
          ? "border-pro-glass-border bg-pro-surface/60 hover:border-pro/30"
          : "border-glass-border bg-void/60 hover:border-success/30"
      }`}
      title={`Switch to ${isEnterprise ? "Free" : "Pro"} mode`}
    >
      {/* Sliding pill */}
      <motion.div
        layout
        className="absolute top-1 h-[calc(100%-8px)] w-[calc(50%-2px)] rounded-full"
        style={{
          background: isEnterprise
            ? "linear-gradient(135deg, rgba(129,140,248,0.2), rgba(167,139,250,0.2))"
            : "linear-gradient(135deg, rgba(52,211,153,0.2), rgba(0,229,200,0.2))",
          left: isEnterprise ? "calc(50% + 1px)" : "4px",
          boxShadow: isEnterprise
            ? "0 0 12px rgba(129,140,248,0.15)"
            : "0 0 12px rgba(0,229,200,0.15)",
        }}
        transition={{ type: "spring", stiffness: 400, damping: 30 }}
      />

      {/* Free option */}
      <div
        className={`relative z-10 flex items-center gap-1.5 px-3 py-1.5 rounded-full text-[11px] font-semibold tracking-wide transition-colors duration-300 ${
          !isEnterprise ? "text-success" : "text-text-ghost"
        }`}
      >
        <Sparkles size={12} />
        Free
      </div>

      {/* Pro option */}
      <div
        className={`relative z-10 flex items-center gap-1.5 px-3 py-1.5 rounded-full text-[11px] font-semibold tracking-wide transition-colors duration-300 ${
          isEnterprise ? "text-pro-bright" : "text-text-ghost"
        }`}
      >
        <Crown size={12} />
        Pro
      </div>
    </button>
  );
}
