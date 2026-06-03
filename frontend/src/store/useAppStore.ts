import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { Mode, Experiment, DatasetPreview } from "@/lib/types";
import { WORKSPACE_COOKIE, setCookie, deleteCookie } from "@/lib/cookies";

interface AppState {
  // ── Mode ──
  mode: Mode;
  /** Whether the user has made a deliberate mode choice at /welcome. Gates app entry. */
  hasSelectedMode: boolean;
  setMode: (mode: Mode) => void;
  toggleMode: () => void;
  /** Deliberate selection from the mode-selection screen — records the choice + marks onboarding done. */
  selectMode: (mode: Mode) => void;
  /** Clears the selection so the user is routed back through mode selection (used by "Switch mode"). */
  resetModeSelection: () => void;

  // ── Upload ──
  dataset: DatasetPreview | null;
  setDataset: (d: DatasetPreview | null) => void;

  // ── Active Run ──
  activeRunId: string | null;
  setActiveRunId: (id: string | null) => void;

  // ── Experiments ──
  experiments: Experiment[];
  addExperiment: (e: Experiment) => void;
  updateExperiment: (runId: string, updates: Partial<Experiment>) => void;
  getExperiment: (runId: string) => Experiment | undefined;

  // ── UI State ──
  sidebarCollapsed: boolean;
  setSidebarCollapsed: (v: boolean) => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      // Mode
      mode: "free",
      hasSelectedMode: false,
      setMode: (mode) => set({ mode }),
      toggleMode: () => set((s) => ({ mode: s.mode === "free" ? "enterprise" : "free" })),
      selectMode: (mode) => {
        // Mirror the chosen workspace so the proxy can deep-link a returning,
        // authenticated user straight to /free or /enterprise.
        setCookie(WORKSPACE_COOKIE, mode);
        set({ mode, hasSelectedMode: true });
      },
      resetModeSelection: () => {
        deleteCookie(WORKSPACE_COOKIE);
        set({ hasSelectedMode: false });
      },

      // Upload
      dataset: null,
      setDataset: (dataset) => set({ dataset }),

      // Active Run
      activeRunId: null,
      setActiveRunId: (activeRunId) => set({ activeRunId }),

      // Experiments
      experiments: [],
      addExperiment: (e) => set((s) => ({ experiments: [e, ...s.experiments].slice(0, 100) })),
      updateExperiment: (runId, updates) =>
        set((s) => ({
          experiments: s.experiments.map((e) =>
            e.run_id === runId ? { ...e, ...updates } : e
          ),
        })),
      getExperiment: (runId) => get().experiments.find((e) => e.run_id === runId),

      // UI
      sidebarCollapsed: false,
      setSidebarCollapsed: (sidebarCollapsed) => set({ sidebarCollapsed }),
    }),
    {
      name: "axiom-app-store",
      partialize: (state) => ({
        mode: state.mode,
        hasSelectedMode: state.hasSelectedMode,
        sidebarCollapsed: state.sidebarCollapsed,
        experiments: state.experiments.slice(0, 20),
      }),
      // Keep the workspace cookie in sync with the rehydrated selection.
      onRehydrateStorage: () => (state) => {
        if (state?.hasSelectedMode) {
          setCookie(WORKSPACE_COOKIE, state.mode);
        } else {
          deleteCookie(WORKSPACE_COOKIE);
        }
      },
    }
  )
);
