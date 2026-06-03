import { create } from "zustand";
import type { AgentId, AgentOutput, AgentStatus, LogEntry, VisualizationResult } from "@/lib/types";
import { AGENT_ORDER } from "@/lib/types";

interface AgentState {
  status: AgentStatus;
  output: AgentOutput | null;
}

interface AgentStoreState {
  agents: Record<AgentId, AgentState>;
  runId: string | null;

  // Actions
  setRunId: (id: string | null) => void;
  setAgentStatus: (id: AgentId, status: AgentStatus) => void;
  setAgentOutput: (id: AgentId, output: AgentOutput) => void;
  appendLog: (id: AgentId, log: LogEntry) => void;
  addVisualization: (id: AgentId, viz: VisualizationResult) => void;
  resetAll: () => void;
  resetAgent: (id: AgentId) => void;

  // Selectors
  getAgent: (id: AgentId) => AgentState;
  getCompletedAgents: () => AgentId[];
  getRunningAgent: () => AgentId | null;
  isAllCompleted: () => boolean;
}

const createInitialAgents = (): Record<AgentId, AgentState> => {
  const agents: Partial<Record<AgentId, AgentState>> = {};
  for (const id of AGENT_ORDER) {
    agents[id] = { status: "idle", output: null };
  }
  return agents as Record<AgentId, AgentState>;
};

export const useAgentStore = create<AgentStoreState>()((set, get) => ({
  agents: createInitialAgents(),
  runId: null,

  setRunId: (runId) => set({ runId }),

  setAgentStatus: (id, status) =>
    set((s) => ({
      agents: { ...s.agents, [id]: { ...s.agents[id], status } },
    })),

  setAgentOutput: (id, output) =>
    set((s) => ({
      agents: { ...s.agents, [id]: { status: output.status, output } },
    })),

  appendLog: (id, log) =>
    set((s) => {
      const agent = s.agents[id];
      if (!agent.output) return s;
      return {
        agents: {
          ...s.agents,
          [id]: {
            ...agent,
            output: { ...agent.output, logs: [...agent.output.logs, log] },
          },
        },
      };
    }),

  addVisualization: (id, viz) =>
    set((s) => {
      const agent = s.agents[id];
      if (!agent.output) return s;
      return {
        agents: {
          ...s.agents,
          [id]: {
            ...agent,
            output: {
              ...agent.output,
              visualizations: [...agent.output.visualizations, viz],
            },
          },
        },
      };
    }),

  resetAll: () => set({ agents: createInitialAgents(), runId: null }),

  resetAgent: (id) =>
    set((s) => ({
      agents: { ...s.agents, [id]: { status: "idle", output: null } },
    })),

  getAgent: (id) => get().agents[id],
  getCompletedAgents: () =>
    AGENT_ORDER.filter((id) => get().agents[id].status === "completed"),
  getRunningAgent: () =>
    AGENT_ORDER.find((id) => get().agents[id].status === "running") ?? null,
  isAllCompleted: () =>
    AGENT_ORDER.every((id) => get().agents[id].status === "completed"),
}));
