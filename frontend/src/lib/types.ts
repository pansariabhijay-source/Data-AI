// ── Axiom Platform Types ──────────────────────────────────────────────────────

export type Mode = "free" | "enterprise";

export type AgentId =
  | "data_collection"
  | "preprocessing"
  | "feature_engineering"
  | "data_splitting"
  | "model_training"
  | "error_detection"
  | "improvement"
  | "finalization";

export type AgentStatus = "idle" | "queued" | "running" | "completed" | "failed" | "skipped";

export interface AgentMeta {
  id: AgentId;
  label: string;
  shortLabel: string;
  description: string;
  icon: string;
  color: string;
}

export const AGENT_META: Record<AgentId, AgentMeta> = {
  data_collection: {
    id: "data_collection",
    label: "Data Collection",
    shortLabel: "Ingest",
    description: "Load, profile, and validate the dataset",
    icon: "Database",
    color: "#e6e9ef",
  },
  preprocessing: {
    id: "preprocessing",
    label: "Preprocessing",
    shortLabel: "Clean",
    description: "Handle missing values, outliers, and duplicates",
    icon: "Eraser",
    color: "#d3d8e1",
  },
  feature_engineering: {
    id: "feature_engineering",
    label: "Feature Engineering",
    shortLabel: "Engineer",
    description: "Encode, transform, and select features",
    icon: "Wrench",
    color: "#c2c8d4",
  },
  data_splitting: {
    id: "data_splitting",
    label: "Data Splitting",
    shortLabel: "Split",
    description: "Create train, validation, and test sets",
    icon: "Split",
    color: "#b4bcca",
  },
  model_training: {
    id: "model_training",
    label: "Model Training",
    shortLabel: "Train",
    description: "Train and evaluate multiple ML models",
    icon: "Brain",
    color: "#c8ced9",
  },
  error_detection: {
    id: "error_detection",
    label: "Error Detection",
    shortLabel: "Audit",
    description: "Detect overfitting, leakage, and anomalies",
    icon: "ShieldAlert",
    color: "#d8dce4",
  },
  improvement: {
    id: "improvement",
    label: "Improvement Loop",
    shortLabel: "Improve",
    description: "Optimize hyperparameters and retry failed models",
    icon: "TrendingUp",
    color: "#bcc3d0",
  },
  finalization: {
    id: "finalization",
    label: "Finalization",
    shortLabel: "Finalize",
    description: "Generate reports, SHAP explanations, and export",
    icon: "PackageCheck",
    color: "#eef1f6",
  },
};

export const AGENT_ORDER: AgentId[] = [
  "data_collection",
  "preprocessing",
  "feature_engineering",
  "data_splitting",
  "model_training",
  "error_detection",
  "improvement",
  "finalization",
];

export interface AgentOutput {
  agent_id: AgentId;
  status: AgentStatus;
  started_at?: string;
  completed_at?: string;
  duration_seconds?: number;
  memory_mb?: number;
  summary: Record<string, unknown>;
  metrics: Record<string, number>;
  logs: LogEntry[];
  artifacts: Record<string, string>;
  visualizations: VisualizationResult[];
  error?: string;
}

export interface LogEntry {
  time: string;
  level: "info" | "warning" | "error" | "debug";
  msg: string;
  agent?: string;
}

export interface VisualizationResult {
  name: string;
  type: string;
  description: string;
  base64_png: string;
  category: "basic" | "advanced";
}

export interface WorkflowNode {
  id: string;
  agentId: AgentId;
  position: { x: number; y: number };
  config: Record<string, unknown>;
}

export interface WorkflowEdge {
  id: string;
  source: string;
  target: string;
}

export interface WorkflowConfig {
  name: string;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  createdAt: string;
}

export interface Experiment {
  run_id: string;
  status: "starting" | "running" | "completed" | "failed";
  mode: Mode;
  started_at: string;
  completed_at?: string;
  data_path: string;
  target_column?: string;
  problem_type?: string;
  best_model?: string;
  best_metric_name?: string;
  best_metric_value?: number;
  agents_run: AgentId[];
  agent_outputs: Partial<Record<AgentId, AgentOutput>>;
}

export interface DatasetPreview {
  filename: string;
  path: string;
  columns: string[];
  dtypes: Record<string, string>;
  n_rows: number;
  preview: Record<string, unknown>[];
}
