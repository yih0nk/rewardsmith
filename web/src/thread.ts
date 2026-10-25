import { ReactNode } from "react";
import { MessageOptions, RequestSummary } from "./api";

// The board as conversations: one per task, each a thread of turns. A turn is one
// request: the user's message and the run it started. Turn 1 is the task; later
// turns are follow-ups that refine one of its candidates.
//
// conversationsFromRequests groups requests by their conversation_id. Requests from
// servers without conversations are each their own one-turn conversation.

export interface Turn {
  requestId: string;
  text: string;
  createdAt: number;
  // 1 for the task itself.
  turn: number;
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

const turnOf = (r: RequestSummary) => r.turn ?? 1;

// Newest conversation first, like a chat app's sidebar. A conversation's status is its
// latest turn's; its title is the task (turn 1).
export function conversationsFromRequests(
  requests: RequestSummary[],
): Conversation[] {
  const groups = new Map<string, RequestSummary[]>();
  for (const r of requests) {
    const id = r.conversation_id ?? r.id;
    groups.set(id, [...(groups.get(id) ?? []), r]);
  }
  return [...groups.entries()]
    .map(([id, rs]) => {
      const turns = [...rs].sort(
        (a, b) => turnOf(a) - turnOf(b) || a.created_at - b.created_at,
      );
      const first = turns[0];
      const last = turns[turns.length - 1];
      const scores = turns
        .map((t) => t.best_score)
        .filter((s): s is number => s != null);
      return {
        id,
        title: first.text,
        status: last.status,
        bestScore: scores.length ? Math.max(...scores) : null,
        createdAt: first.created_at,
        turns: turns.map((t) => ({
          requestId: t.id,
          text: t.text,
          createdAt: t.created_at,
          turn: turnOf(t),
        })),
      };
    })
    .sort((a, b) => b.createdAt - a.createdAt);
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
  const running = requests.find((r) => r.status === "running");
  return {
    running: running ? taskName(requests, running) : null,
    ahead: requests
      .filter(
        (r) =>
          r.status === "queued" &&
          r.id !== id &&
          (r.created_at < me.created_at ||
            (r.created_at === me.created_at && r.id < id)),
      )
      .sort((a, b) => a.created_at - b.created_at)
      .map((r) => taskName(requests, r)),
  };
}

// A run as the queue names it: its task, and the note when it is a follow-up
// ("Walk to the ball (follow-up: faster)"), so a note alone never stands for a task.
export function taskName(requests: RequestSummary[], r: RequestSummary): string {
  if (turnOf(r) === 1) return r.text;
  const task = requests.find(
    (q) => q.id === r.conversation_id || (q.conversation_id === r.conversation_id && turnOf(q) === 1),
  );
  return task ? `${task.text} (follow-up: ${r.text})` : r.text;
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
      // false when the candidate has no score yet: there is nothing to refine.
      scored?: boolean;
    };

// Renders the reply control for a target ("Refine this"). Nothing renders where it is unset.
export type RenderReply = (target: ReplyTarget) => ReactNode;

// Stable DOM ids, so a reply (or a link) can point at a run or a candidate.
export const runAnchor = (requestId: string) => `run-${requestId}`;
export const candidateAnchor = (candidateId: number) =>
  `candidate-${candidateId}`;

// Options a message carries (Instant/Effort, model). See modes.ts for the defaults.
export type ComposerOptions = MessageOptions;
