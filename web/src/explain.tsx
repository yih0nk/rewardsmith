import { ReactNode, useId, useState } from "react";
import { SuccessClause } from "./api";

// Plain-language explanations for the words the pipeline uses.
export const GLOSSARY = {
  critic:
    "Claude watches frames from the robot's video and scores how well the motion matches your prompt, from 0 (not at all) to 10 (exactly).",
  gate: "Before training, each reward's code must compile and run a few simulator steps. If it crashes, Claude reads the error and fixes it, up to 3 tries.",
  halving:
    "Successive halving: partway through training, Claude compares the robots and stops the weaker ones early to save GPU time.",
  carry:
    "Each new round starts from the best 2 rewards so far: Claude edits their code using the critique of their videos.",
  episodeReward:
    "The reward the robot earned per episode, on its reward function's own scale. It shows learning progress; it can't be compared across candidates.",
  successCheck:
    "A pass/fail test Claude writes from your prompt before any training, over measured motion (speed, turning, height, falls). It never changes after results come in.",
  candidate:
    "One reward function Claude wrote, and the robot trained with it. Each round trains several side by side.",
} as const;

// A word with its explanation: a dotted underline, the explanation on hover or focus.
export function Explain({
  term,
  children,
}: {
  term: keyof typeof GLOSSARY;
  children: ReactNode;
}) {
  const id = useId();
  // Open the tip leftward when it would run past the right edge of the window.
  const [flip, setFlip] = useState(false);
  const place = (el: Element) =>
    setFlip(el.getBoundingClientRect().left + 460 > window.innerWidth);
  return (
    <span
      className={`explain ${flip ? "flip" : ""}`}
      onPointerEnter={(e) => place(e.currentTarget)}
      onFocus={(e) => place(e.currentTarget)}
    >
      <button type="button" className="explain-term" aria-describedby={id}>
        {children}
      </button>
      <span className="explain-tip" role="tooltip" id={id}>
        {GLOSSARY[term]}
      </span>
    </span>
  );
}

// ---- Metrics: names, units and how to read them ----

interface MetricInfo {
  label: string;
  format: (v: number) => string;
  help?: string;
}

const num = (v: number, digits = 2) => String(Number(v.toFixed(digits)));
const unit =
  (u: string, digits = 2) =>
  (v: number) =>
    `${num(v, digits)} ${u}`;
const percent = (v: number) => `${Math.round(v * 100)}%`;
const yesNo = (v: number) => (v >= 0.5 ? "yes" : "no");
const ofOne = (v: number) => `${num(v)} of 1`;

const METRICS: Record<string, MetricInfo> = {
  forward_velocity: {
    label: "Forward speed",
    format: unit("m/s"),
    help: "negative = backwards",
  },
  lateral_velocity: {
    label: "Sideways speed",
    format: unit("m/s"),
    help: "positive = to its left",
  },
  yaw_rate: {
    label: "Turning rate",
    format: unit("rad/s"),
    help: "positive = counter-clockwise",
  },
  base_height: {
    label: "Body height",
    format: unit("m"),
    help: "about 0.28 m standing",
  },
  upright: {
    label: "Uprightness",
    format: ofOne,
    help: "1 = level, 0 = on its side",
  },
  fell: { label: "Fell over", format: yesNo },
  abs_forward_velocity: {
    label: "Speed forwards or backwards",
    format: unit("m/s"),
  },
  abs_lateral_velocity: {
    label: "Sideways speed, either way",
    format: unit("m/s"),
  },
  abs_yaw_rate: { label: "Turning rate, either way", format: unit("rad/s") },
  episode_reward: {
    label: "Training reward",
    format: (v) => `${num(v)} pts`,
    help: "on its reward's own scale",
  },
  fake_quality: { label: "Fake quality", format: (v) => `${num(v, 1)} / 10` },
  rollout_steps: {
    label: "Video length",
    format: (v) => `${Math.round(v)} steps (${num(v / 50, 1)} s)`,
  },
  target_reached: {
    label: "Reached the target",
    format: percent,
    help: "share of test episodes",
  },
  target_start_dist: { label: "Start distance to target", format: unit("m") },
  target_min_dist: { label: "Closest to target", format: unit("m") },
  target_final_dist: { label: "Final distance to target", format: unit("m") },
  target_time_to_reach: { label: "Time to reach target", format: unit("s", 1) },
  target_fell: {
    label: "Fell on the way",
    format: percent,
    help: "share of test episodes",
  },
  train_s: { label: "Training time", format: unit("s", 0) },
  compile_s: { label: "Compile time", format: unit("s", 0) },
  robust_s: { label: "Robustness test time", format: unit("s", 0) },
  train_randomized: { label: "Trained on varied robots", format: yesNo },
  train_pushes: { label: "Pushed during training", format: yesNo },
};

// Shown elsewhere: reward terms in the breakdown, robustness counts in their own line.
export const isListedMetric = (name: string) =>
  !name.startsWith("term_") && !name.startsWith("robust_");

export function metricInfo(name: string): MetricInfo {
  return (
    METRICS[name] ?? {
      label: name.replaceAll("_", " "),
      format: (v) => num(v, 4),
    }
  );
}

// ---- Success-check metrics: one labeler for the table and the sentence ----
//
// Scene checks use outcome metrics (agents/checker.py): reach:OBJ, push:OBJ:ZONE,
// climb:BOX, stand_on:BOX, avoid:OBJ, stand_still, move, forward, backward, spin,
// each optionally @SECONDS, combined as all:A&B (same episode) or sequence:A>B>C
// (in order). They and target_reached / target_fell / scene_fell are rates over
// fresh episodes, 0 to 1. Each gets a table label ("Push ball into goal") and a
// phrase that completes "Pass = the robot …" ("pushes the ball into the goal").

interface Outcome {
  label: string;
  does: string;
}

const NAMED: Record<string, Outcome> = {
  target_reached: { label: "Reached target", does: "reaches the target" },
  target_fell: { label: "Episodes with a fall", does: "falls" },
  scene_fell: { label: "Episodes with a fall", does: "falls" },
};

const words = (name?: string) => (name ?? "").replaceAll("_", " ");

function outcome(metric: string): Outcome | null {
  if (NAMED[metric]) return NAMED[metric];
  const combined = (parts: string[], label: string, does: string) => {
    const each = parts.map(outcome);
    if (each.some((o) => !o)) return null;
    return {
      label: each.map((o) => o!.label).join(label),
      does: each.map((o) => o!.does).join(does),
    };
  };
  if (metric.startsWith("sequence:"))
    return combined(metric.slice(9).split(">"), ", then ", ", then ");
  if (metric.startsWith("all:"))
    return combined(metric.slice(4).split("&"), " and ", " and ");
  const [event, duration] = metric.split("@");
  const [kind, object, zone] = event.split(":");
  const o = words(object);
  const kinds: Record<string, Outcome> = {
    reach: { label: `Reach ${o}`, does: `reaches the ${o}` },
    push: {
      label: `Push ${o} into ${words(zone)}`,
      does: `pushes the ${o} into the ${words(zone)}`,
    },
    climb: { label: `Climb onto ${o}`, does: `climbs onto the ${o}` },
    stand_on: {
      label: `Stand still on ${o}`,
      does: `stands still on the ${o}`,
    },
    avoid: {
      label: `Avoid ${o} with progress`,
      does: `avoids the ${o} while making progress`,
    },
    stand_still: { label: "Stand still", does: "stands still" },
    move: { label: "Move from the start", does: "moves away from the start" },
    forward: { label: "Move forward", does: "moves forward" },
    backward: { label: "Move backward", does: "moves backward" },
    spin: { label: "Complete a turn", does: "completes a full turn" },
  };
  const found = kinds[kind];
  if (!found) return null;
  const hold = duration ? ` for ${duration} s` : "";
  return { label: found.label + hold, does: found.does + hold };
}

// Rates across fresh episodes, shown as percentages.
export const isRateMetric = (metric: string) => outcome(metric) != null;

// The table's name for a check metric.
export const checkLabel = (metric: string) =>
  outcome(metric)?.label ??
  (metric === "target_final_dist"
    ? "Final target distance"
    : metricInfo(metric).label);

// A measured or required value, with its unit.
export function checkValue(metric: string, v: number | null): string {
  if (v == null || !Number.isFinite(v)) return "Not measured";
  if (isRateMetric(metric)) return `${Number((v * 100).toFixed(1))}%`;
  return metricInfo(metric).format(v);
}

// Per-episode fall rates beat the single test-run "fell": when both are in a check,
// the board shows the rate and keeps "fell" out of the table and the sentence.
const EPISODE_FALLS = ["scene_fell", "target_fell"];

export function visibleClauses<T extends { metric: string }>(
  clauses: T[],
): { shown: T[]; hidden: T[] } {
  const rate = clauses.some((c) => EPISODE_FALLS.includes(c.metric));
  return {
    shown: clauses.filter((c) => !(rate && c.metric === "fell")),
    hidden: clauses.filter((c) => rate && c.metric === "fell"),
  };
}

// ---- The success check as a sentence ----

const speed = (v: number) => `${num(Math.abs(v))} m/s`;
const turn = (v: number) =>
  `${num(Math.abs(v))} rad/s (${Math.round((Math.abs(v) * 180) / Math.PI)}°/s)`;

// "at least" for >=, "more than" for >, "at most" for <=, "fewer than" / "below" for <.
const RATE_WORDS: Record<string, string> = {
  ">=": "at least",
  ">": "more than",
  "<=": "at most",
  "<": "fewer than",
};
const LEVEL_WORDS: Record<string, string> = {
  ">=": "at least",
  ">": "above",
  "<=": "at most",
  "<": "below",
};

// One clause, completing "Pass = the robot …". Every op reads the right way round:
// >= and > are "at least" / "above", <= and < are "at most" / "below".
export function describeClause({ metric, op, value }: SuccessClause): string {
  const above = op.startsWith(">");
  const inclusive = op.endsWith("=");
  const level = LEVEL_WORDS[op] ?? op;
  // A bound on speed: "faster than" / "at least" for a minimum, "slower than" /
  // "at most" for a maximum. Going backwards below -0.3 m/s is a minimum speed.
  const pace = (minimum: boolean, than: string) =>
    minimum
      ? `${inclusive ? "at least" : "faster than"} ${than}`
      : `${inclusive ? "at most" : "slower than"} ${than}`;

  const rate = outcome(metric);
  if (rate)
    return `${rate.does} in ${RATE_WORDS[op] ?? op} ${checkValue(metric, value)} of episodes`;
  switch (metric) {
    case "fell":
      return above ? "falls over" : "doesn't fall over";
    case "target_final_dist":
      return above
        ? `ends ${inclusive ? "at least" : "more than"} ${num(value)} m from the target`
        : `ends within ${num(value)} m of the target`;
    case "upright":
      return above
        ? `stays upright (uprightness ${level} ${num(value)} of 1)`
        : `tips over (uprightness ${level} ${num(value)} of 1)`;
    case "base_height":
      return `keeps its body ${level} ${num(value)} m`;
    case "forward_velocity":
      if (!above && value <= 0)
        return `walks backwards ${pace(true, speed(value))}`;
      if (above && value >= 0)
        return `walks forwards ${pace(true, speed(value))}`;
      return above
        ? `goes backwards no faster than ${speed(value)}`
        : `goes forwards no faster than ${speed(value)}`;
    case "abs_forward_velocity":
      return `moves forwards or backwards ${pace(above, speed(value))}`;
    case "lateral_velocity":
      if (above && value >= 0) return `steps left ${pace(true, speed(value))}`;
      if (!above && value <= 0)
        return `steps right ${pace(true, speed(value))}`;
      return `moves sideways ${level} ${num(value)} m/s (left is positive)`;
    case "abs_lateral_velocity":
      return `moves sideways ${pace(above, speed(value))}`;
    case "yaw_rate":
      if (above && value >= 0)
        return `turns counter-clockwise ${pace(true, turn(value))}`;
      if (!above && value <= 0)
        return `turns clockwise ${pace(true, turn(value))}`;
      return `turns ${level} ${num(value)} rad/s (counter-clockwise is positive)`;
    case "abs_yaw_rate":
      return `turns either way ${pace(above, turn(value))}`;
    default:
      return `has ${metricInfo(metric).label.toLowerCase()} ${level} ${checkValue(metric, value)}`;
  }
}

const joinAnd = (parts: string[]) =>
  parts.length < 2
    ? parts.join("")
    : `${parts.slice(0, -1).join(", ")} and ${parts.at(-1)}`;

export function successSentence(spec: SuccessClause[] | null | undefined) {
  if (!spec?.length) return null;
  return `Pass = the robot ${joinAnd(spec.map(describeClause))}.`;
}
