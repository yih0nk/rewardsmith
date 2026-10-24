import { mockApi } from "./mockApi";

// Set VITE_MOCK_API=0 to connect to the backend through the Vite proxy.
export const MOCK_MODE =
  (import.meta as ImportMeta & { env: Record<string, string> }).env
    .VITE_MOCK_API !== "0";
export type Status = "queued" | "running" | "done" | "failed";

export interface Candidate {
  id: number;
  idx: number;
  reward_src: string;
  diff: string | null;
  status: string;
  error: string | null;
  metrics: Record<string, number> | null;
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

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export const listRequests = (): Promise<RequestSummary[]> =>
  MOCK_MODE
    ? mockApi.list()
    : fetch("/api/requests").then((r) => json<RequestSummary[]>(r));

export const getRequest = (id: string): Promise<RequestDetail> =>
  MOCK_MODE
    ? mockApi.get(id)
    : fetch(`/api/requests/${id}`).then((r) => json<RequestDetail>(r));

export const createRequest = (text: string) =>
  MOCK_MODE
    ? mockApi.create(text)
    : fetch("/api/requests", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      }).then((r) => json<{ id: string }>(r));
