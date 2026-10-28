// Typed shapes for the `/api/v1` contract. Kept narrow on purpose: the UI must
// show "unavailable" rather than guess a field the backend did not send.

export interface Health {
  status: "ok" | "degraded";
  profile: string;
  anchor: "mock" | "real";
  version: string;
  device: string;
  loaded: boolean;
  active_op: string | null;
  uptime_seconds: number;
  error: string | null;
}

export interface MemoryRow {
  text: string | null;
  source: string | null;
  score?: number | null;
  channels?: Record<string, number> | null;
}

export interface InferenceResult {
  prompt: string;
  response: string;
  profile: string | null;
  anchor: string;
  backend: string;
  memory: MemoryRow[];
}

export interface RecallResult {
  query: string;
  results: MemoryRow[];
}

export interface ModelsResponse {
  models: string[];
  selected: string;
  loaded: boolean;
}

export interface Job {
  id: string;
  kind: string;
  params: Record<string, unknown>;
  status: "queued" | "running" | "done" | "failed" | "cancelled";
  created_at: number;
  updated_at: number;
  progress: number;
  result: unknown;
  error: string | null;
}

export interface ExperimentRow {
  id: string;
  bytes: number;
  modified: number;
  kind: string;
}

export interface ApiError {
  error: string;
}
