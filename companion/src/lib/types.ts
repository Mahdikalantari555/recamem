export interface Health {
  status: string;
  profile: string;
  anchor: string;
  version: string;
  device: string;
  loaded: boolean;
  active_op: string | null;
  error: string | null;
}

export interface PetAttribute {
  value: number | null;
  sources: string[];
}

export interface CompanionState {
  anchor: string;
  backend: string;
  profile: string;
  facts_total: number | null;
  facts_active: number | null;
  facts_retired: number | null;
  feedback_counts: { approve: number; reject: number; total: number };
  memory_health: Record<string, unknown>;
  ttt_status: Record<string, unknown>;
  retrieval_health: Record<string, unknown>;
  retrieval_hit_rate: number | null;
  last_sleep_at: number | null;
  selected_checkpoint: string | null;
  measured_at: number;
  signals: string[];
}

export interface Pet {
  mood: PetAttribute;
  energy: PetAttribute;
  bond: PetAttribute;
  knowledge: PetAttribute;
  note: string;
}

export interface CompanionEvent {
  id: string;
  action: string;
  engine_call: string;
  target: number | null;
  timestamp: number;
  before: Record<string, unknown>;
  after: Record<string, unknown>;
}

export interface Checkpoint {
  path: string;
  name: string;
  eligible: boolean;
  reason: string;
  held_out_bpb: number | null;
  gates: Record<string, string>;
  config?: unknown;
  seed?: number;
  dataset?: string;
  commit?: string;
  command?: string;
}

export interface Job {
  id: string;
  kind: string;
  status: string;
  progress: number;
  result: Record<string, unknown> | null;
  error: string | null;
  created_at: number;
  updated_at: number;
  params?: Record<string, unknown>;
}

export interface Turn {
  prompt: string;
  response: string;
  memory: Array<{ text?: string; source?: string; score?: number }>;
  anchor: string;
  backend: string;
  delivery: string;
}
