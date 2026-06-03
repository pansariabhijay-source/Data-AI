"use client";

import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  Database, Eraser, Wrench, Split, Brain, ShieldAlert,
  TrendingUp, PackageCheck, Cpu, type LucideIcon,
} from "lucide-react";
import { AGENT_ORDER, AGENT_META, type AgentId } from "@/lib/types";

const ICONS: Record<AgentId, LucideIcon> = {
  data_collection: Database,
  preprocessing: Eraser,
  feature_engineering: Wrench,
  data_splitting: Split,
  model_training: Brain,
  error_detection: ShieldAlert,
  improvement: TrendingUp,
  finalization: PackageCheck,
};

interface AgentNetworkProps {
  /** Rendered viewport size in px (square). */
  size?: number;
  className?: string;
  /** When true (default) the active agent cycles automatically to feel "alive". */
  animate?: boolean;
}

/**
 * Animated agent constellation — eight specialized agents orbiting the AXIOM
 * core, with live data pulses travelling the pipeline ring. Pure SVG + Framer
 * Motion; no external dependencies.
 */
export default function AgentNetwork({ size = 560, className = "", animate = true }: AgentNetworkProps) {
  const VB = 600;
  const cx = VB / 2;
  const cy = VB / 2;
  const R = 215;
  const coreR = 58;
  const nodeR = 34;

  const nodes = useMemo(
    () =>
      AGENT_ORDER.map((id, i) => {
        const angle = (-90 + i * (360 / AGENT_ORDER.length)) * (Math.PI / 180);
        return {
          id,
          x: cx + R * Math.cos(angle),
          y: cy + R * Math.sin(angle),
          color: AGENT_META[id].color,
          label: AGENT_META[id].shortLabel,
          Icon: ICONS[id],
        };
      }),
    [cx, cy],
  );

  // Closed ring path threading every node in pipeline order — drives the comet.
  const ringPath = useMemo(() => {
    const pts = nodes.map((n) => `${n.x.toFixed(1)},${n.y.toFixed(1)}`);
    return `M ${pts.join(" L ")} Z`;
  }, [nodes]);

  const [active, setActive] = useState(0);
  useEffect(() => {
    if (!animate) return;
    const t = setInterval(() => setActive((a) => (a + 1) % nodes.length), 1400);
    return () => clearInterval(t);
  }, [animate, nodes.length]);

  return (
    <div className={className} style={{ width: size, height: size, maxWidth: "100%" }}>
      <svg viewBox={`0 0 ${VB} ${VB}`} width="100%" height="100%" role="img" aria-label="Axiom agent network">
        <defs>
          <radialGradient id="an-core" cx="50%" cy="40%" r="65%">
            <stop offset="0%" stopColor="#c7d2fe" />
            <stop offset="45%" stopColor="#818cf8" />
            <stop offset="100%" stopColor="#4f46e5" />
          </radialGradient>
          <radialGradient id="an-core-halo" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="rgba(99,102,241,0.45)" />
            <stop offset="70%" stopColor="rgba(99,102,241,0.05)" />
            <stop offset="100%" stopColor="transparent" />
          </radialGradient>
          <filter id="an-soft" x="-60%" y="-60%" width="220%" height="220%">
            <feGaussianBlur stdDeviation="6" />
          </filter>
          <linearGradient id="an-ring" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#6366f1" />
            <stop offset="50%" stopColor="#a5b4fc" />
            <stop offset="100%" stopColor="#a78bfa" />
          </linearGradient>
        </defs>

        {/* Core halo */}
        <circle cx={cx} cy={cy} r={170} fill="url(#an-core-halo)" />

        {/* Spokes: core → each agent */}
        {nodes.map((n, i) => (
          <line
            key={`spoke-${n.id}`}
            x1={cx} y1={cy} x2={n.x} y2={n.y}
            stroke={n.color}
            strokeWidth={active === i ? 1.6 : 0.8}
            strokeOpacity={active === i ? 0.55 : 0.16}
            style={{ transition: "stroke-opacity 0.6s, stroke-width 0.6s" }}
          />
        ))}

        {/* Pipeline ring */}
        <path d={ringPath} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth={1.2} />
        <path
          d={ringPath}
          fill="none"
          stroke="url(#an-ring)"
          strokeWidth={1.6}
          strokeOpacity={0.5}
          strokeDasharray="3 9"
          strokeLinecap="round"
        >
          <animate attributeName="stroke-dashoffset" from="24" to="0" dur="1.2s" repeatCount="indefinite" />
        </path>

        {/* Data comets travelling the ring */}
        {animate && [0, 0.5].map((begin, i) => (
          <circle key={`comet-${i}`} r={4.5} fill="#a5b4fc">
            <animateMotion dur="7s" begin={`${begin * 7}s`} repeatCount="indefinite" path={ringPath} rotate="auto" />
            <animate attributeName="opacity" values="0;1;1;0" dur="7s" begin={`${begin * 7}s`} repeatCount="indefinite" />
          </circle>
        ))}

        {/* Pulses flowing core → active node */}
        {animate && (
          <motion.circle
            key={`pulse-${active}`}
            r={3.5}
            fill={nodes[active].color}
            initial={{ cx, cy, opacity: 0 }}
            animate={{ cx: nodes[active].x, cy: nodes[active].y, opacity: [0, 1, 1, 0] }}
            transition={{ duration: 1.2, ease: "easeInOut" }}
          />
        )}

        {/* Agent nodes */}
        {nodes.map((n, i) => {
          const isActive = active === i;
          return (
            <g key={n.id} transform={`translate(${n.x}, ${n.y})`}>
              {/* glow */}
              <motion.circle
                r={nodeR + 8}
                fill={n.color}
                opacity={0.12}
                filter="url(#an-soft)"
                animate={isActive ? { scale: [1, 1.25, 1], opacity: [0.18, 0.32, 0.18] } : { scale: 1, opacity: 0.1 }}
                transition={{ duration: 1.4, repeat: isActive ? Infinity : 0, ease: "easeInOut" }}
              />
              {/* active expanding ring */}
              {isActive && (
                <motion.circle
                  r={nodeR}
                  fill="none"
                  stroke={n.color}
                  strokeWidth={1.5}
                  initial={{ scale: 1, opacity: 0.6 }}
                  animate={{ scale: 1.9, opacity: 0 }}
                  transition={{ duration: 1.4, repeat: Infinity, ease: "easeOut" }}
                />
              )}
              {/* body */}
              <circle
                r={nodeR}
                fill="rgba(11,13,19,0.92)"
                stroke={n.color}
                strokeWidth={isActive ? 2 : 1.2}
                strokeOpacity={isActive ? 1 : 0.45}
                style={{ transition: "stroke-opacity 0.5s, stroke-width 0.5s" }}
              />
              {/* icon */}
              <g transform="translate(-11, -15)" color={n.color}>
                <n.Icon size={22} strokeWidth={1.75} color={n.color} opacity={isActive ? 1 : 0.7} />
              </g>
              {/* label */}
              <text
                y={nodeR + 16}
                textAnchor="middle"
                fontSize="11"
                fontWeight={600}
                fill={isActive ? n.color : "#5b6472"}
                style={{ transition: "fill 0.5s", fontFamily: "var(--font-sans)" }}
              >
                {n.label}
              </text>
            </g>
          );
        })}

        {/* Core */}
        <motion.g
          animate={{ scale: [1, 1.03, 1] }}
          transition={{ duration: 3.5, repeat: Infinity, ease: "easeInOut" }}
          style={{ transformOrigin: `${cx}px ${cy}px` }}
        >
          <circle cx={cx} cy={cy} r={coreR + 10} fill="none" stroke="rgba(99,102,241,0.25)" strokeWidth={1}>
            <animateTransform attributeName="transform" type="rotate" from={`0 ${cx} ${cy}`} to={`360 ${cx} ${cy}`} dur="22s" repeatCount="indefinite" />
          </circle>
          <circle cx={cx} cy={cy} r={coreR} fill="url(#an-core)" />
          <circle cx={cx} cy={cy} r={coreR} fill="none" stroke="rgba(255,255,255,0.35)" strokeWidth={1} />
          <g transform={`translate(${cx - 17}, ${cy - 24})`}>
            <Cpu size={34} strokeWidth={1.6} color="#ffffff" />
          </g>
          <text x={cx} y={cy + 22} textAnchor="middle" fontSize="12" fontWeight={800} fill="#ffffff" letterSpacing="1.5" style={{ fontFamily: "var(--font-display)" }}>
            AXIOM
          </text>
        </motion.g>
      </svg>
    </div>
  );
}
