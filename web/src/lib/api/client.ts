// Minimal typed fetch client. No state library: plain hooks + this module.
import type {
  ExperimentRow,
  Health,
  InferenceResult,
  Job,
  ModelsResponse,
  RecallResult,
} from "../schemas";

export class ApiRequestError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiRequestError";
  }
}

const BASE = "/api/v1";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  const text = await response.text();
  const body = text ? JSON.parse(text) : {};
  if (!response.ok) {
    const message =
      typeof body?.error === "string" ? body.error : `request failed (${response.status})`;
    throw new ApiRequestError(response.status, message);
  }
  return body as T;
}

export const api = {
  health: () => request<Health>("/health"),
  models: () => request<ModelsResponse>("/models"),
  loadModel: (name: string) =>
    request<{ profile: string; load_seconds: number }>("/models/load", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),
  infer: (prompt: string, signal?: AbortSignal) =>
    request<InferenceResult>("/inference", {
      method: "POST",
      body: JSON.stringify({ prompt }),
      signal,
    }),
  teach: (text: string) =>
    request<{ facts_stored: number }>("/teach", {
      method: "POST",
      body: JSON.stringify({ text }),
    }),
  recall: (query: string, topK = 3) =>
    request<RecallResult>("/recall", {
      method: "POST",
      body: JSON.stringify({ query, top_k: topK }),
    }),
  clear: () => request<{ cleared: boolean; profile: string }>("/clear", { method: "POST" }),
  jobs: () => request<{ jobs: Job[] }>("/jobs"),
  createJob: (kind: string) =>
    request<Job>("/jobs", { method: "POST", body: JSON.stringify({ kind }) }),
  cancelJob: (id: string) =>
    request<Job>(`/jobs/${id}/cancel`, { method: "POST", body: "{}" }),
  experiments: () => request<{ experiments: ExperimentRow[] }>("/experiments"),
  bench: () =>
    request<{ status: string; reason: string; results: unknown[] }>("/bench", {
      method: "POST",
      body: "{}",
    }),
};
