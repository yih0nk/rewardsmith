export type Status = "queued" | "running" | "paused" | "done" | "failed" | "cancelled";

// Live training progress, written by the training process as it goes.
export type ProgressStage = "compiling" | "training" | "rendering" | "done";

export interface CurvePoint {
  steps: number;
  wall_s: number;
  // null when a diverged run reported NaN.
  episode_reward: number | null;
  terms: Record<string, number | null>;
}

export interface Progress {
  stage: ProgressStage;
  // Can pass total_steps (brax rounds up); fraction is capped at 1.
  steps: number;
  total_steps: number;
  fraction: number;
  wall_s: number;
  curve: CurvePoint[];
}

// written → training → trained → scored, or gate_failed / failed, or stopped
// (successive halving cut it short mid-training: metrics but no video or score).
export type CandidateStatus =
  | "written"
  | "gate_failed"
  | "training"
  | "trained"
  | "scored"
  | "stopped"
  | "failed";

export interface CheckResult {
  pass: boolean;
  verified?: boolean;
  error?: string;
  clauses: { metric: string; op: string; value: number; actual: number | null; pass: boolean }[];
}

export interface Candidate {
  check_result?: CheckResult | null;
  id: number;
  idx: number;
  reward_src: string;
  diff: string | null;
  // What the writer or editor says this candidate tries.
  idea: string | null;
  status: CandidateStatus | (string & {});
  error: string | null;
  metrics: Record<string, unknown> | null;
  video_path: string | null;
  score: number | null;
  critique: string | null;
  // 1 when the critic suspects reward hacking.
  reward_hacking: number | null;
  // Set when someone flags the candidate as reward hacking (the hall of fame).
  flagged_at: number | null;
  flag_note: string | null;
  // null until the training process writes its first progress file. Absent on hall of fame entries.
  progress?: Progress | null;
  // Passed / total runs of the success check on varied robots, seeds and pushes; null until scored.
  robustness?: Record<string, { passed: number; total: number }> | null;
  // Server seconds when the candidate's draft was stored (before its code check).
  created_at?: number;
}

export interface HallEntry extends Candidate {
  round_n: number;
  request_id: string;
  request_text: string;
}

export interface Round {
  id: number;
  n: number;
  status: string;
  best_candidate_id: number | null;
  candidates: Candidate[];
  // Server seconds when the round started.
  created_at?: number;
}

export interface RequestSummary {
  id: string;
  text: string;
  status: Status;
  best_score: number | null;
  best_candidate_id?: number | null;
  succeeded?: number | null;
  success_check_error?: string | null;
  // A pause or cancel the worker will act on at its next checkpoint.
  control?: "pause" | "cancel" | null;
  created_at: number;
  // The task (conversation) this turn belongs to, and its place in it (1 = the task itself).
  // Older servers leave them out: every request is then its own one-turn task.
  conversation_id?: string | null;
  turn?: number | null;
  // instant | effort, and the writer/editor/critic model. null on runs from before modes.
  mode?: Mode | null;
  model?: string | null;
  // A follow-up's candidate to refine; null = the task's best when it starts.
  target_candidate_id?: number | null;
  // 1 when a follow-up's note changed the scene, so it trained from scratch.
  scene_changed?: number | null;
}

export type Mode = "instant" | "effort";

// How the worker names a follow-up's target (server get_request).
export interface TurnTarget {
  id: number;
  request_id: string;
  turn: number;
  round_n: number;
  idx: number;
  score: number | null;
}

// One clause of the pass/fail check Claude writes before training (agents/checker.py).
export interface SuccessClause {
  metric: string;
  op: string;
  value: number;
}

// A step the server timed itself (db.stage). round_n is null for scene and check.
export interface ServerStage {
  name: "scene" | "check" | "writing" | "compiling" | "training" | "judging";
  round_n: number | null;
  started_at: number;
  ended_at: number | null;
}

export interface RequestDetail extends RequestSummary {
  success_spec?: SuccessClause[] | null;
  rounds: Round[];
  // round_n and candidate (0-based idx) are set on events about one candidate (coach notes).
  events: {
    id: number;
    kind: string;
    message: string;
    created_at: number;
    round_n?: number | null;
    candidate?: number | null;
  }[];
  // Every timed step so far; older servers leave it out.
  stages?: ServerStage[];
  // The candidate a follow-up refines, once the worker picked it.
  target?: TurnTarget | null;
}

export interface Health {
  ok: boolean;
  fake_sim: boolean;
  fake_llm: boolean;
  // Rewards Claude writes per round in Effort mode (older servers leave it out).
  candidates_per_round?: number;
  max_rounds?: number;
  // The models a message may pick, strongest first (modes.py).
  models?: string[];
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

// Server clock minus client clock, in seconds, from the last response's Date header.
// Stage timers use it so a pod with a drifting clock doesn't start them at a wrong value.
let clockOffset = 0;
export const serverClockOffset = () => clockOffset;

export function noteServerDate(header: string | null, clientMs = Date.now()) {
  const server = header ? Date.parse(header) : NaN;
  // The header has 1 s resolution: ignore offsets that small.
  if (Number.isFinite(server))
    clockOffset = Math.abs(server - clientMs) > 2000 ? (server - clientMs) / 1000 : 0;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail =
      typeof body?.detail === "string"
        ? body.detail
        : Array.isArray(body?.detail)
          ? body.detail.map((item: { msg: string }) => item.msg).join("; ")
          : res.statusText;
    throw new ApiError(
      res.status,
      `${res.status}: ${detail || "API request failed"}`,
    );
  }
  return res.json() as Promise<T>;
}

export const getHealth = (): Promise<Health> =>
  fetch("/api/health").then((r) => json<Health>(r));

// Worker paths may use Windows separators; URLs always use forward slashes.
export const artifactUrl = (path: string) =>
  `/artifacts/${path.replaceAll("\\", "/").split("/").map(encodeURIComponent).join("/")}`;

export const listRequests = (): Promise<RequestSummary[]> =>
  fetch("/api/requests").then((r) => json<RequestSummary[]>(r));

export const getRequest = (id: string): Promise<RequestDetail> =>
  fetch(`/api/requests/${encodeURIComponent(id)}`).then((r) => {
    noteServerDate(r.headers?.get?.("date") ?? null);
    return json<RequestDetail>(r);
  });

// Per-message options: Instant or Effort, and the writer/editor/critic model.
export interface MessageOptions {
  mode: Mode;
  model: string;
}

// A new task: a conversation whose first turn is this request (same id).
export const createRequest = (text: string, options?: MessageOptions) =>
  fetch("/api/requests", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, ...options }),
  }).then((r) => json<{ id: string; conversation_id?: string }>(r));

// A follow-up turn. An empty text means "keep going"; no target = the task's best candidate.
export const createFollowUp = (
  conversationId: string,
  text: string,
  options: MessageOptions,
  targetCandidateId?: number | null,
) =>
  fetch(`/api/conversations/${encodeURIComponent(conversationId)}/turns`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      ...options,
      ...(targetCandidateId != null ? { target_candidate_id: targetCandidateId } : {}),
    }),
  }).then((r) => json<{ id: string; conversation_id: string }>(r));

export const listHallOfFame = (): Promise<HallEntry[]> =>
  fetch("/api/hall-of-fame").then((r) => json<HallEntry[]>(r));

export const flagCandidate = (id: number, note: string) =>
  fetch(`/api/candidates/${id}/flag`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ note }),
  }).then((r) => json<{ ok: boolean }>(r));

export const unflagCandidate = (id: number) =>
  fetch(`/api/candidates/${id}/flag`, { method: "DELETE" }).then((r) =>
    json<{ ok: boolean }>(r),
  );

export type RunAction = "pause" | "resume" | "cancel";

// Pause, resume or cancel a run. Returns the new state: "pausing" / "cancelling" while
// the worker finishes its current step. A 409 means the run's state doesn't allow it.
export const controlRequest = (id: string, action: RunAction) =>
  fetch(`/api/requests/${encodeURIComponent(id)}/${action}`, { method: "POST" }).then(
    (r) => json<{ state: string }>(r),
  );
