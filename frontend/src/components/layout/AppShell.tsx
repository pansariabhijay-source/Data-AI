"use client";

import { useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { useAppStore } from "@/store/useAppStore";
import { useAuthStore } from "@/store/useAuthStore";
import Sidebar from "@/components/layout/Sidebar";
import Navbar from "@/components/layout/Navbar";

export default function AppShell({ children }: { children: React.ReactNode }) {
  const { sidebarCollapsed, setMode, hasSelectedMode, mode } = useAppStore();
  const { isAuthenticated, token, logout } = useAuthStore();
  const pathname = usePathname();
  const router = useRouter();
  const [mounted, setMounted] = useState(false);
  const [tokenChecked, setTokenChecked] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // ── Validate the stored token against the backend on first mount ────────
  // This catches the case where the browser has a stale/expired token in
  // localStorage but the backend no longer recognizes it.
  useEffect(() => {
    if (!mounted) return;
    if (!token) {
      setTokenChecked(true);
      return;
    }
    let cancelled = false;
    fetch("/api/auth/me", {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => {
        if (cancelled) return;
        if (res.status === 401) {
          // Token is stale/expired — clear everything so user can re-login
          logout();
        }
        setTokenChecked(true);
      })
      .catch(() => {
        // Network error — don't lock the user out, let them try
        if (!cancelled) setTokenChecked(true);
      });
    return () => {
      cancelled = true;
    };
  }, [mounted, token, logout]);

  // ── Route classification ──────────────────────────────────────────────────
  const isLanding = pathname === "/";
  const isAuthPage = pathname === "/auth";
  const isWelcome = pathname === "/welcome";
  const isFreeRoute = pathname === "/free" || pathname.startsWith("/free/");
  const isEnterpriseRoute =
    pathname === "/enterprise" || pathname.startsWith("/enterprise/");

  // Authentication comes first. Only the marketing landing page and the auth
  // page are public — every product surface (free flow included) requires an
  // account, then a deliberate mode choice at /welcome.
  const isPublic = isLanding || isAuthPage;
  const needsMode = isFreeRoute || isEnterpriseRoute || pathname.startsWith("/pipeline") || pathname.startsWith("/report");

  // Keep the persisted UI mode aligned with the section the user is actually in.
  useEffect(() => {
    if (isFreeRoute) setMode("free");
    else if (isEnterpriseRoute) setMode("enterprise");
  }, [isFreeRoute, isEnterpriseRoute, setMode]);

  // 1) Gate protected routes behind authentication.
  useEffect(() => {
    if (mounted && tokenChecked && !isAuthenticated && !isPublic) {
      router.replace("/auth");
    }
  }, [mounted, tokenChecked, isAuthenticated, isPublic, pathname, router]);

  // 2) Send already-authenticated users away from the auth page into the flow.
  //    A returning user who already chose a mode resumes in their workspace;
  //    everyone else makes a deliberate choice at /welcome.
  useEffect(() => {
    if (mounted && tokenChecked && isAuthenticated && isAuthPage) {
      if (hasSelectedMode) router.replace(mode === "free" ? "/free" : "/enterprise");
      else router.replace("/welcome");
    }
  }, [mounted, tokenChecked, isAuthenticated, isAuthPage, hasSelectedMode, mode, router]);

  // 3) Require a deliberate mode selection before entering any app surface.
  useEffect(() => {
    if (mounted && tokenChecked && isAuthenticated && !hasSelectedMode && needsMode) {
      router.replace("/welcome");
    }
  }, [mounted, tokenChecked, isAuthenticated, hasSelectedMode, needsMode, router]);

  // Prevent flash of unauthenticated / pre-selection content.
  const blocked =
    !tokenChecked ||
    (!isAuthenticated && !isPublic) ||
    (isAuthenticated && !hasSelectedMode && needsMode);
  if (!mounted || blocked) {
    return <div className="min-h-screen bg-void" />;
  }

  // The Stitch sidebar + topbar belong to the enterprise console only.
  const showSidebar = isEnterpriseRoute;

  if (!showSidebar) {
    return (
      <div>
        <div className="mesh-bg" />
        <div className="grid-overlay" />
        {children}
      </div>
    );
  }

  // Enterprise layout with Stitch sidebar + topbar
  return (
    <div className="flex min-h-screen">
      <div className="mesh-bg" />
      <div className="grid-overlay" />
      <Navbar />
      <Sidebar />
      <main
        className="flex-1 transition-all duration-300 relative z-10 pt-16"
        style={{
          marginLeft: sidebarCollapsed ? 64 : 256,
        }}
      >
        {children}
      </main>
    </div>
  );
}

