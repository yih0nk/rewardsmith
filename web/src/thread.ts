import { ReactNode } from "react";
import { RequestSummary } from "./api";

// The board as conversations: one per task, each a thread of turns. A turn is one
// request: the user's message and the run it started.
//
// Today every request is its own conversation with a single turn. When the backend
// adds follow-ups (a request that refines an earlier one), conversationsFromRequests
// is the one place that groups requests into threads.

export interface Turn {
  requestId: string;
  text: string;
  createdAt: number;
}

export interface Conversation {
  // The first turn's request id; also the #hash that opens it.
  id: string;
  title: string;
  status: RequestSummary["status"];
  // Best critic score across the conversation's turns, or null.
  bestScore: number | null;
  createdAt: number;
  turns: Turn[];
}

// Newest conversation first, like a chat app's sidebar.
export function conversationsFromRequests(
  requests: RequestSummary[],
): Conversation[] {
  return [...requests]
    .sort((a, b) => b.created_at - a.created_at)
    .map((r) => ({
      id: r.id,
      title: r.text,
      status: r.status,
      bestScore: r.best_score ?? null,
      createdAt: r.created_at,
      turns: [{ requestId: r.id, text: r.text, createdAt: r.created_at }],
    }));
}

// One GPU, one run at a time, oldest queued first (the worker's claim order).
export interface QueuePlace {
  // The task running now, if any.
  running: string | null;
  // Queued tasks that will run before this one.
  ahead: string[];
}

export function queuePlace(
  requests: RequestSummary[],
  id: string,
): QueuePlace | null {
  const me = requests.find((r) => r.id === id);
  if (!me || me.status !== "queued") return null;
  return {
    running: requests.find((r) => r.status === "running")?.text ?? null,
    ahead: requests
      .filter(
        (r) =>
          r.status === "queued" &&
          r.id !== id &&
          (r.created_at < me.created_at ||
            (r.created_at === me.created_at && r.id < id)),
      )
      .sort((a, b) => a.created_at - b.created_at)
      .map((r) => r.text),
  };
}

// What a later "refine this" reply points at: the whole run, or one candidate in it.
export type ReplyTarget =
  | { kind: "run"; requestId: string }
  | {
      kind: "candidate";
      requestId: string;
      round: number;
      candidateId: number;
      // Zero-based, like the API; shown as idx + 1.
      idx: number;
    };

// Renders the reply control for a target. Unset today: no follow-up backend yet,
// so nothing renders where it would go.
export type RenderReply = (target: ReplyTarget) => ReactNode;

// Stable DOM ids, so a reply (or a link) can point at a run or a candidate.
export const runAnchor = (requestId: string) => `run-${requestId}`;
export const candidateAnchor = (candidateId: number) =>
  `candidate-${candidateId}`;

// Options a message will carry once the backend takes them (Instant/Effort, model).
// The composer has a slot for their controls; nothing reads them yet.
export interface ComposerOptions {
  mode?: string;
  model?: string;
}
