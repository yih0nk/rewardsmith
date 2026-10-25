import { Candidate, RequestDetail, Round, ServerStage } from "./api";
import { QueuePlace } from "./thread";

// Where a run is, in words a first-time viewer can follow, and since when.
// Times are server seconds (time.time()); the board ticks them on the client clock.

export type StepKey =
  | "setup"
  | "writing"
  | "gate"
  | "compiling"
  | "training"
  | "rendering"
  | "judging";

export type StageKey =
  "queued" | StepKey | "between" | "done" | "no_results" | "failed";

// The steps one run goes through, in order. Writing to judging repeats every round.
export const STEPS: { key: StepKey; short: string }[] = [
  { key: "setup", short: "Set up task" },
  { key: "writing", short: "Write rewards" },
  { key: "gate", short: "Test code" },
  { key: "compiling", short: "Compile" },
  { key: "training", short: "Train" },
  { key: "rendering", short: "Record video" },
  { key: "judging", short: "Judge" },
];

export interface RunStage {
  key: StageKey;
  // One sentence: what is happening right now.
  label: string;
  // What it means or how long it usually takes; optional.
  hint?: string;
  round: number | null;
  // Server seconds when this stage began, or null when the API doesn't say.
  startedAt: number | null;
}

const PROGRESS_ORDER = {
  compiling: 0,
  training: 1,
  rendering: 2,
  done: 3,
} as const;

const plural = (n: number, one: string, many = `${one}s`) =>
  `${n} ${n === 1 ? one : many}`;

const finite = (v: unknown): v is number =>
  typeof v === "number" && Number.isFinite(v);

// The last "Training N candidate(s)" event of a round: when its training processes started.
function trainEvent(detail: RequestDetail, round: Round): number | null {
  const since = round.created_at ?? -Infinity;
  const events = detail.events.filter(
    (e) => e.kind === "train" && e.created_at >= since,
  );
  return events.at(-1)?.created_at ?? null;
}

// Training processes report seconds since they started (progress.wall_s).
// Added to the train event, that dates the stage they reached.
function sinceTrain(start: number | null, seconds: (number | null)[]) {
  const known = seconds.filter(finite);
  return start != null && known.length ? start + Math.max(...known) : null;
}

const earliest = (times: (number | null | undefined)[]) => {
  const known = times.filter(finite);
  return known.length ? Math.min(...known) : null;
};

export function isScored(c: Candidate) {
  return finite(c.score);
}

// "Queued, starts when “walk backwards” finishes": who it is waiting for, from the
// request list. Without a place in the queue, a generic line.
function queuedLabel(queue: QueuePlace | null | undefined) {
  if (!queue)
    return {
      label: "Waiting in line for the GPU worker",
      hint: "Runs train one at a time. This one starts when the worker is free.",
    };
  const hint = "One GPU, so tasks train one at a time.";
  const [next, ...rest] = queue.ahead;
  if (queue.running && !next)
    return { label: `Queued, starts when “${queue.running}” finishes`, hint };
  if (next)
    return {
      label: `Queued behind ${plural(queue.ahead.length + (queue.running ? 1 : 0), "task")}`,
      hint: `${queue.running ? `“${queue.running}” is running, then ` : ""}“${next}”${
        rest.length ? ` and ${plural(rest.length, "more task")}` : ""
      } go first. ${hint}`,
    };
  return { label: "Queued, starting in a few seconds", hint };
}

// Board steps the server times on its own. Training, compiling and recording share the
// server's one "training" step, so those keep the per-candidate progress times.
const SERVER_STAGES: Partial<Record<StageKey, ServerStage["name"][]>> = {
  setup: ["scene", "check"],
  writing: ["writing"],
  gate: ["compiling"],
  judging: ["judging"],
};

export function runStage(
  detail: RequestDetail,
  candidatesPerRound?: number | null,
  queue?: QueuePlace | null,
): RunStage {
  const stage = inferStage(detail, candidatesPerRound, queue);
  const names = SERVER_STAGES[stage.key];
  if (!names || detail.status !== "running") return stage;
  const open = (detail.stages ?? []).filter(
    (s) =>
      names.includes(s.name) &&
      s.ended_at == null &&
      (stage.key === "setup" || s.round_n === stage.round),
  );
  const started = earliest(open.map((s) => s.started_at));
  return started == null ? stage : { ...stage, startedAt: started };
}

function inferStage(
  detail: RequestDetail,
  candidatesPerRound?: number | null,
  queue?: QueuePlace | null,
): RunStage {
  const rounds = detail.rounds;
  const last = rounds.at(-1);
  const tried = rounds.reduce((n, r) => n + r.candidates.length, 0);
  const lastEvent = detail.events.at(-1);

  if (detail.status === "queued")
    return {
      key: "queued",
      ...queuedLabel(queue),
      round: null,
      startedAt: detail.created_at,
    };
  if (detail.status === "failed") {
    const error = [...detail.events].reverse().find((e) => e.kind === "error");
    return {
      key: "failed",
      label: "The run stopped with an error",
      hint: error?.message,
      round: last?.n ?? null,
      startedAt: error?.created_at ?? lastEvent?.created_at ?? null,
    };
  }
  if (detail.status === "done") {
    const scored = rounds.some((r) => r.candidates.some(isScored));
    return scored
      ? {
          key: "done",
          label: `Finished after ${plural(rounds.length, "round")}${
            finite(detail.best_score)
              ? ` · best score ${detail.best_score.toFixed(1)} / 10`
              : ""
          }`,
          round: last?.n ?? null,
          startedAt: lastEvent?.created_at ?? null,
        }
      : {
          key: "no_results",
          label: `Finished, but none of the ${plural(tried, "reward")} worked`,
          hint: "Every candidate failed before Claude could judge it.",
          round: last?.n ?? null,
          startedAt: lastEvent?.created_at ?? null,
        };
  }

  // Running. Before round 1, Claude sets up the scene and the pass/fail check.
  if (!last) {
    const scene = detail.events.some((e) => e.kind === "scene");
    return {
      key: "setup",
      label: scene
        ? "Claude is writing the pass/fail check"
        : "Claude is reading your prompt and setting up the scene",
      hint: "Done once per run, before any training.",
      round: null,
      startedAt: detail.created_at,
    };
  }
  if (last.status === "done")
    return {
      key: "between",
      label: `Round ${last.n} is done. Starting round ${last.n + 1}`,
      round: last.n,
      startedAt: null,
    };

  const cands = last.candidates;
  if (cands.length === 0) {
    const previous = rounds.at(-2);
    const k = previous?.candidates.length || candidatesPerRound || null;
    return {
      key: "writing",
      label: previous
        ? `Claude is revising the best rewards into ${k ? plural(k, "new one", "new ones") : "new ones"}`
        : `Claude is writing ${k ? plural(k, "reward function") : "reward functions"}`,
      hint: previous
        ? "It reads the critique of the best rewards so far and edits their code."
        : "Each one is a different idea, in Python, for scoring the robot's motion.",
      round: last.n,
      startedAt: last.created_at ?? null,
    };
  }

  const training = cands.filter((c) => c.status === "training");
  if (training.length) {
    const tStart = trainEvent(detail, last);
    // The least advanced one decides the stage: the round moves on when all of them do.
    const stage = Math.min(
      ...training.map((c) =>
        c.progress ? (PROGRESS_ORDER[c.progress.stage] ?? 0) : 0,
      ),
    );
    if (stage === 0)
      return {
        key: "compiling",
        label: `Compiling the simulator for ${plural(training.length, "robot")}`,
        hint: "JAX turns each reward into GPU code. Usually about 1–2 min.",
        round: last.n,
        startedAt: tStart,
      };
    if (stage === 1)
      return {
        key: "training",
        label: `Training ${plural(training.length, "robot")} in parallel`,
        hint: "Each robot learns from its own reward with PPO, a standard RL algorithm.",
        round: last.n,
        startedAt: sinceTrain(
          tStart,
          training.map((c) => c.progress?.curve[0]?.wall_s ?? null),
        ),
      };
    return {
      key: "rendering",
      label: "Recording a video of each trained robot",
      round: last.n,
      startedAt: sinceTrain(
        tStart,
        training.map((c) => c.progress?.curve.at(-1)?.wall_s ?? null),
      ),
    };
  }
  const judging = cands.filter((c) => c.status === "trained");
  if (judging.length)
    return {
      key: "judging",
      label: `Claude is watching ${plural(judging.length, "video")} and scoring each one`,
      hint: "Scores run from 0 to 10: how well the motion matches your prompt.",
      round: last.n,
      startedAt: sinceTrain(
        trainEvent(detail, last),
        judging.map((c) => c.progress?.wall_s ?? null),
      ),
    };
  if (cands.some((c) => c.status === "written"))
    return {
      key: "gate",
      label: `Test-running the code of ${plural(cands.length, "reward")}`,
      hint: "Each reward must compile and run a few steps. Claude fixes errors, up to 3 tries.",
      round: last.n,
      startedAt: earliest(cands.map((c) => c.created_at)),
    };
  return {
    key: "between",
    label: `Wrapping up round ${last.n}`,
    round: last.n,
    startedAt: null,
  };
}

// 0..STEPS.length: how many steps of the current round are done. Finished runs count all.
export function stepIndex(key: StageKey): number {
  if (key === "done" || key === "no_results") return STEPS.length;
  if (key === "queued") return -1;
  if (key === "between") return STEPS.length;
  return STEPS.findIndex((s) => s.key === key);
}

// "7s", "1m 05s", "1h 02m". Negative or unknown input reads as 0s.
export function formatElapsed(seconds: number): string {
  const s = Math.max(0, Math.floor(finite(seconds) ? seconds : 0));
  if (s < 60) return `${s}s`;
  if (s < 3600)
    return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}m`;
}

// Seconds from a server timestamp to now, on the client clock corrected by the
// server's clock offset (server minus client, in seconds).
export function elapsedSince(
  startedAt: number,
  nowMs: number,
  offsetS = 0,
): number {
  return Math.max(0, nowMs / 1000 + offsetS - startedAt);
}

export type CandidateKind =
  "waiting" | "live" | "scored" | "failed" | "stopped";

// A candidate's state in plain words, for cards and status lines. `live` comes from
// TrainingProgress.liveState while it trains.
export function candidateState(
  c: Candidate,
  liveLabel?: string | null,
): { kind: CandidateKind; text: string; why?: string } {
  if (isScored(c))
    return { kind: "scored", text: `Scored ${c.score!.toFixed(1)} / 10` };
  switch (c.status) {
    case "written":
      return { kind: "waiting", text: "Testing its code" };
    case "gate_failed":
      return {
        kind: "failed",
        text: "Code never ran",
        why: "It failed the code check 3 times, so it was not trained.",
      };
    case "stopped":
      return {
        kind: "stopped",
        text: "Stopped early",
        why: "Claude stopped it mid-training to give the GPU to stronger candidates.",
      };
    case "failed":
      if (c.error?.startsWith("The critic could not"))
        return {
          kind: "failed",
          text: "Couldn't be judged",
          why: "Claude's scoring call failed.",
        };
      if (c.error?.startsWith("Stopped: the run failed"))
        return {
          kind: "failed",
          text: "Cut short",
          why: "The whole run stopped before it finished.",
        };
      return { kind: "failed", text: "Crashed while training" };
    case "trained":
      return { kind: "live", text: "Being judged by Claude" };
    default:
      return { kind: "live", text: liveLabel ?? c.status.replaceAll("_", " ") };
  }
}

export const candidateName = (round: number, idx: number) =>
  `Round ${round} · Candidate ${idx + 1}`;
