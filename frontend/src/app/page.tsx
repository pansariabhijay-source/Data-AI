"use client";

import { useRouter } from "next/navigation";
import { Check, ArrowRight, Sparkles, Crown } from "lucide-react";
import { useAuthStore } from "@/store/useAuthStore";
import { useAppStore } from "@/store/useAppStore";
import type { Mode } from "@/lib/types";

const SERIF = "'Instrument Serif', serif";

const NAV_LINKS = [
  { label: "Home", href: "#top", active: true },
  { label: "Platform", href: "#platform" },
  { label: "Pricing", href: "#platform" },
  { label: "Reach Us", href: "#contact" },
];

const VIDEO_SRC =
  "https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260314_131748_f2ca2a28-fed7-44c8-b9a9-bd9acdd5ec31.mp4";

const TIERS: {
  id: Mode;
  name: string;
  tagline: string;
  price: string;
  priceNote: string;
  accent: string;
  glow: string;
  Icon: typeof Sparkles;
  features: string[];
  cta: string;
}[] = [
  {
    id: "free",
    name: "Free",
    tagline: "Browser quick-start",
    price: "$0",
    priceNote: "forever",
    accent: "#d3d8e1",
    glow: "rgba(211,216,225,0.14)",
    Icon: Sparkles,
    features: [
      "Full 8-agent autonomous pipeline",
      "Classification, regression & clustering",
      "Up to 10,000 rows per dataset",
      "Downloadable model artifacts & reports",
    ],
    cta: "Start free",
  },
  {
    id: "enterprise",
    name: "Enterprise",
    tagline: "The full command center",
    price: "Custom",
    priceNote: "talk to us",
    accent: "#aab3c4",
    glow: "rgba(170,179,196,0.16)",
    Icon: Crown,
    features: [
      "Visual workflow builder & agent console",
      "Live execution with full run history",
      "Experiment tracking & model comparison",
      "REST API access + artifact export",
    ],
    cta: "Open console",
  },
];

export default function LandingPage() {
  const router = useRouter();
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const hasSelectedMode = useAppStore((s) => s.hasSelectedMode);
  const selectMode = useAppStore((s) => s.selectMode);
  const mode = useAppStore((s) => s.mode);

  // Generic entry — resume workspace if known, else pick a mode at /welcome.
  const beginJourney = () => {
    if (!isAuthenticated) router.push("/auth");
    else if (hasSelectedMode) router.push(mode === "free" ? "/free" : "/enterprise");
    else router.push("/welcome");
  };

  // Tier-specific entry — remember the chosen tier, sign in if needed.
  const enter = (tier: Mode) => {
    if (!isAuthenticated) {
      router.push(`/auth?next=${tier === "free" ? "/free" : "/enterprise"}`);
      return;
    }
    selectMode(tier);
    router.push(tier === "free" ? "/free" : "/enterprise");
  };

  return (
    <main id="top" className="relative w-full overflow-hidden bg-background">
      {/* ═══════════════ HERO — fullscreen looping video ═══════════════ */}
      <section className="relative h-screen w-full overflow-hidden">
        <video
          className="absolute inset-0 z-0 h-full w-full object-cover"
          autoPlay
          loop
          muted
          playsInline
          preload="auto"
          src={VIDEO_SRC}
        />

        <div className="relative z-10 flex h-full flex-col">
          {/* ── Navigation ── */}
          <nav className="mx-auto flex w-full max-w-7xl items-center justify-between px-8 py-6">
            <button
              onClick={() => router.push("/")}
              className="text-3xl tracking-tight text-foreground"
              style={{ fontFamily: SERIF }}
            >
              <img src="/logo.png" alt="Axiom" className="h-10 w-auto object-contain" />
            </button>

            <div className="hidden items-center gap-9 md:flex">
              {NAV_LINKS.map((link) => (
                <a
                  key={link.label}
                  href={link.href}
                  className={`text-sm transition-colors hover:text-foreground ${
                    link.active ? "text-foreground" : "text-muted-foreground"
                  }`}
                >
                  {link.label}
                </a>
              ))}
            </div>

            <button
              onClick={beginJourney}
              className="liquid-glass cursor-pointer rounded-full px-6 py-2.5 text-sm text-foreground hover:scale-[1.03]"
            >
              Begin Journey
            </button>
          </nav>

          {/* ── Hero copy ── */}
          <div className="flex flex-1 flex-col items-center justify-center px-6 pb-24 text-center">
            <h1
              className="animate-fade-rise max-w-7xl text-5xl font-normal leading-[0.95] tracking-[-2.46px] text-foreground sm:text-7xl md:text-8xl"
              style={{ fontFamily: SERIF }}
            >
              Where data <em className="not-italic text-muted-foreground">learns</em> to{" "}
              <em className="not-italic text-muted-foreground">think for itself.</em>
            </h1>

            <p className="animate-fade-rise-delay mt-8 max-w-2xl text-base leading-relaxed text-muted-foreground sm:text-lg">
              Axiom is the autonomous data scientist for deep thinkers, bold creators, and
              quiet rebels. Upload a dataset and eight agents handle the rest — clean models,
              clear reports, and sharp focus, with zero configuration.
            </p>

            <button
              onClick={beginJourney}
              className="liquid-glass animate-fade-rise-delay-2 mt-12 cursor-pointer rounded-full px-14 py-5 text-base text-foreground hover:scale-[1.03]"
            >
              Begin Journey
            </button>
          </div>
        </div>
      </section>

      {/* ═══════════════ CHOOSE YOUR PATH — Free vs Enterprise ═══════════════ */}
      <section id="platform" className="relative mx-auto max-w-6xl px-6 py-28">
        <div className="text-center">
          <span className="text-[11px] font-semibold uppercase tracking-[0.32em] text-muted-foreground">
            Two ways to run Axiom
          </span>
          <h2
            className="mt-5 text-5xl tracking-[-1.5px] text-foreground md:text-6xl"
            style={{ fontFamily: SERIF }}
          >
            Choose your <em className="not-italic text-muted-foreground">path.</em>
          </h2>
          <p className="mx-auto mt-5 max-w-xl text-base leading-relaxed text-muted-foreground">
            Start free in your browser — no card, no setup. Or step into the full
            enterprise command center when you need control.
          </p>
        </div>

        <div className="mt-14 grid gap-6 md:grid-cols-2">
          {TIERS.map((tier) => (
            <div
              key={tier.id}
              className="liquid-glass group relative flex flex-col rounded-[28px] p-9 transition-transform duration-500 hover:-translate-y-1"
            >
              {/* accent wash */}
              <div
                className="pointer-events-none absolute inset-0 rounded-[28px] opacity-0 transition-opacity duration-500 group-hover:opacity-100"
                style={{ background: `radial-gradient(ellipse 90% 60% at 30% 0%, ${tier.glow}, transparent 70%)` }}
              />
              <div className="relative z-10 flex flex-1 flex-col">
                <div className="flex items-center justify-between">
                  <span
                    className="grid h-12 w-12 place-items-center rounded-2xl"
                    style={{ background: `${tier.accent}1a`, border: `1px solid ${tier.accent}40` }}
                  >
                    <tier.Icon size={22} style={{ color: tier.accent }} strokeWidth={1.75} />
                  </span>
                  <span className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
                    {tier.tagline}
                  </span>
                </div>

                <h3 className="mt-7 text-4xl text-foreground" style={{ fontFamily: SERIF }}>
                  {tier.name}
                </h3>
                <div className="mt-2 flex items-end gap-2">
                  <span className="text-3xl text-foreground" style={{ fontFamily: SERIF }}>
                    {tier.price}
                  </span>
                  <span className="mb-1 text-sm text-muted-foreground">/ {tier.priceNote}</span>
                </div>

                <ul className="mt-7 flex-1 space-y-3.5">
                  {tier.features.map((f) => (
                    <li key={f} className="flex items-center gap-3 text-sm text-foreground/80">
                      <Check size={15} style={{ color: tier.accent }} className="shrink-0" />
                      {f}
                    </li>
                  ))}
                </ul>

                <button
                  onClick={() => enter(tier.id)}
                  className="group/btn mt-9 inline-flex items-center justify-center gap-2.5 rounded-full px-7 py-4 text-sm font-medium text-[#03141d] transition-transform duration-300 hover:scale-[1.02]"
                  style={{ background: tier.accent }}
                >
                  {tier.cta}
                  <ArrowRight size={16} className="transition-transform group-hover/btn:translate-x-1" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ═══════════════ FOOTER ═══════════════ */}
      <footer id="contact" className="border-t border-white/10 py-12">
        <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-4 px-6 sm:flex-row">
          <img src="/logo.png" alt="Axiom" className="h-8 w-auto object-contain" />
          <span className="text-xs text-muted-foreground">
            Autonomous ML infrastructure · built for quiet rebels
          </span>
          <button
            onClick={beginJourney}
            className="liquid-glass cursor-pointer rounded-full px-6 py-2.5 text-sm text-foreground hover:scale-[1.03]"
          >
            Begin Journey
          </button>
        </div>
      </footer>
    </main>
  );
}
