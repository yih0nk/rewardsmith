import { mockApi } from "./mockApi";

// The real backend is the default. Opt into the isolated demo with VITE_MOCK_API=1.
export const MOCK_MODE =
  (import.meta as ImportMeta & { env: Record<string, string> }).env
    .VITE_MOCK_API === "1";
export type Status = "queued" | "running" | "done" | "failed";

export interface Candidate {
  id: number;
  idx: number;
  reward_src: string;
  diff: string | null;
  status: string;
  error: string | null;
  metrics: Record<string, unknown> | null;
  video_path: string | null;
  score: number | null;
  critique: string | null;
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
  MOCK_MODE
    ? Promise.resolve({ ok: true, fake_sim: true, fake_llm: true })
    : fetch("/api/health").then((r) => json<Health>(r));

// Worker paths may use Windows separators; URLs always use forward slashes.
export const artifactUrl = (path: string) =>
  `/artifacts/${path.replaceAll("\\", "/").split("/").map(encodeURIComponent).join("/")}`;

export const listRequests = (): Promise<RequestSummary[]> =>
  MOCK_MODE
    ? mockApi.list()
    : fetch("/api/requests").then((r) => json<RequestSummary[]>(r));

export const getRequest = (id: string): Promise<RequestDetail> =>
  MOCK_MODE
    ? mockApi.get(id)
    : fetch(`/api/requests/${encodeURIComponent(id)}`).then((r) =>
        json<RequestDetail>(r),
      );

export const createRequest = (text: string) =>
  MOCK_MODE
    ? mockApi.create(text)
    : fetch("/api/requests", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      }).then((r) => json<{ id: string }>(r));
