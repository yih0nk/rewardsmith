export type Status = "queued" | "running" | "done" | "failed";

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
}

export interface RequestSummary {
  id: string;
  text: string;
  status: Status;
  best_score: number | null;
  best_candidate_id?: number | null;
  succeeded?: number | null;
  success_check_error?: string | null;
  created_at: number;
}

export interface RequestDetail extends RequestSummary {
  rounds: Round[];
  events: { id: number; kind: string; message: string; created_at: number }[];
}

export interface Health {
  ok: boolean;
  fake_sim: boolean;
  fake_llm: boolean;
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
  fetch(`/api/requests/${encodeURIComponent(id)}`).then((r) =>
    json<RequestDetail>(r),
  );

export const createRequest = (text: string) =>
  fetch("/api/requests", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  }).then((r) => json<{ id: string }>(r));

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
