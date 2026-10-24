import { ReactNode, useId } from "react";
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
  return (
    <span className="explain">
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

// ---- The success check as a sentence ----

const speed = (v: number) => `${num(Math.abs(v))} m/s`;
const turn = (v: number) =>
  `${num(Math.abs(v))} rad/s (${Math.round((Math.abs(v) * 180) / Math.PI)}°/s)`;

// One clause, completing "Pass = the robot …".
export function describeClause({ metric, op, value }: SuccessClause): string {
  const above = op === ">";
  const faster = above ? "faster" : "slower";
  switch (metric) {
    case "fell":
      return above ? "falls over" : "doesn't fall over";
    case "upright":
      return above
        ? `stays upright (uprightness above ${num(value)} of 1)`
        : `tips over (uprightness below ${num(value)} of 1)`;
    case "base_height":
      return `keeps its body ${above ? "above" : "below"} ${num(value)} m`;
    case "forward_velocity":
      if (!above && value <= 0)
        return `walks backwards faster than ${speed(value)}`;
      if (above && value >= 0)
        return `walks forwards faster than ${speed(value)}`;
      return above
        ? `goes backwards no faster than ${speed(value)}`
        : `goes forwards no faster than ${speed(value)}`;
    case "abs_forward_velocity":
      return `moves forwards or backwards ${faster} than ${speed(value)}`;
    case "lateral_velocity":
      if (above && value >= 0) return `steps left faster than ${speed(value)}`;
      if (!above && value <= 0)
        return `steps right faster than ${speed(value)}`;
      return `moves sideways ${above ? "above" : "below"} ${num(value)} m/s (left is positive)`;
    case "abs_lateral_velocity":
      return `moves sideways ${faster} than ${speed(value)}`;
    case "yaw_rate":
      if (above && value >= 0)
        return `turns counter-clockwise faster than ${turn(value)}`;
      if (!above && value <= 0)
        return `turns clockwise faster than ${turn(value)}`;
      return `turns ${above ? "above" : "below"} ${num(value)} rad/s (counter-clockwise is positive)`;
    case "abs_yaw_rate":
      return `turns either way ${faster} than ${turn(value)}`;
    default:
      return `has ${metric.replaceAll("_", " ")} ${above ? "above" : "below"} ${num(value)}`;
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
