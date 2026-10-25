import { Health, MessageOptions, Mode } from "./api";

// Instant and Effort, as the composer offers them. The worker's modes.py has the numbers;
// the hints here are what one run takes on the RTX 4090.

export const MODES: { id: Mode; label: string; hint: string }[] = [
  { id: "instant", label: "Instant", hint: "1 reward, 1 round · first video in about 2 min" },
  { id: "effort", label: "Effort", hint: "4 rewards a round, 2+ rounds, live coach · about 10 min" },
];

export const MODEL_LABELS: Record<string, string> = {
  "claude-opus-5-5": "Opus 5.5",
  "claude-sonnet-5-5": "Sonnet 5.5",
  "claude-haiku-5-5": "Haiku 5.5",
};

// The model each mode uses unless the message picks another (modes.MODES on the worker).
export const MODE_MODEL: Record<Mode, string> = {
  instant: "claude-sonnet-5-5",
  effort: "claude-opus-5-5",
};

// A new task searches; a follow-up is quick.
export const TASK_OPTIONS: MessageOptions = { mode: "effort", model: MODE_MODEL.effort };
export const FOLLOWUP_OPTIONS: MessageOptions = { mode: "instant", model: MODE_MODEL.instant };

export const modelLabel = (id: string | null | undefined) =>
  id ? (MODEL_LABELS[id] ?? id) : "";

export const modeLabel = (id: string | null | undefined) =>
  MODES.find((m) => m.id === id)?.label ?? "Effort";

// The models the server takes, strongest first; older servers don't list them.
export const modelsOf = (health: Health | null) =>
  health?.models?.length ? health.models : Object.keys(MODEL_LABELS);
