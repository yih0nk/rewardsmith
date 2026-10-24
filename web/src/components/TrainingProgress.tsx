import { Candidate, CurvePoint } from "../api";

export interface LiveState {
  label: string;
  // 0..1 for the progress bar, or null when there is nothing to measure yet.
  fraction: number | null;
}

const clamp01 = (v: unknown) =>
  typeof v === "number" && Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0;

// What an unfinished candidate is doing right now, or null once it has a score or failed.
export function liveState(c: Candidate): LiveState | null {
  if (c.score != null) return null;
  if (c.status === "scored" || c.status === "failed" || c.status === "gate_failed")
    return null;
  // Trained means the video exists and Claude is scoring it, whatever the progress file says.
  if (c.status === "trained") return { label: "Judging", fraction: 1 };
  const p = c.progress;
  if (!p) {
    // The training process takes ~10 s to write its first progress file.
    return c.status === "training"
      ? { label: "Starting…", fraction: 0 }
      : { label: "Waiting to train", fraction: null };
  }
  const fraction = clamp01(p.fraction);
  switch (p.stage) {
    case "compiling":
      return { label: "Compiling", fraction };
    case "rendering":
      return { label: "Rendering video", fraction: 1 };
    case "done":
      // Its round trains candidates together: until the others finish, it isn't judged yet.
      return { label: c.status === "training" ? "Finishing up" : "Judging", fraction: 1 };
    default:
      // Floor, so the bar never says 100% while steps are still running.
      return { label: `Training ${Math.floor(fraction * 100)}%`, fraction };
  }
}

export const formatReward = (v: number) =>
  Math.abs(v) >= 100 ? String(Math.round(v)) : String(Number(v.toFixed(2)));

export interface SparklineShape {
  points: { x: number; y: number }[];
  latest: number;
  min: number;
  max: number;
}

// Episode reward over training steps, scaled into a width x height box. Null rewards are skipped.
export function sparkline(
  curve: CurvePoint[] | null | undefined,
  width: number,
  height: number,
  pad = 2,
): SparklineShape | null {
  const pts = (curve ?? []).filter(
    (p): p is CurvePoint & { episode_reward: number } =>
      typeof p?.episode_reward === "number" && Number.isFinite(p.episode_reward),
  );
  if (pts.length === 0) return null;
  const rewards = pts.map((p) => p.episode_reward);
  const min = Math.min(...rewards);
  const max = Math.max(...rewards);
  const steps = pts.map((p, i) => (Number.isFinite(p.steps) ? p.steps : i));
  const s0 = Math.min(...steps);
  const s1 = Math.max(...steps);
  const w = width - 2 * pad;
  const h = height - 2 * pad;
  const round = (v: number) => Math.round(v * 10) / 10;
  return {
    points: pts.map((p, i) => ({
      x: round(pad + (s1 > s0 ? ((steps[i] - s0) / (s1 - s0)) * w : w / 2)),
      // A flat curve sits in the middle rather than on an edge.
      y: round(pad + (max > min ? (1 - (p.episode_reward - min) / (max - min)) * h : h / 2)),
    })),
    latest: rewards[rewards.length - 1],
    min,
    max,
  };
}

export function Sparkline({
  curve,
  width = 120,
  height = 28,
  caption = "reward",
}: {
  curve: CurvePoint[] | null | undefined;
  width?: number;
  height?: number;
  caption?: string;
}) {
  const shape = sparkline(curve, width, height);
  if (!shape) return null;
  const last = shape.points[shape.points.length - 1];
  const summary = `Episode reward over ${shape.points.length} ${
    shape.points.length === 1 ? "eval" : "evals"
  }, from ${formatReward(shape.min)} to ${formatReward(shape.max)}, latest ${formatReward(shape.latest)}`;
  return (
    <span className="sparkline">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        width={width}
        height={height}
        role="img"
        aria-label={summary}
      >
        {shape.points.length > 1 && (
          <polyline
            points={shape.points.map((p) => `${p.x},${p.y}`).join(" ")}
          />
        )}
        <circle cx={last.x} cy={last.y} r={2} />
      </svg>
      <span className="sparkline-value" aria-hidden="true">
        {caption} <strong>{formatReward(shape.latest)}</strong>
      </span>
    </span>
  );
}

// Stage label, a progress bar and the reward curve for a candidate that is still training.
export default function TrainingProgress({
  c,
  large = false,
}: {
  c: Candidate;
  large?: boolean;
}) {
  const live = liveState(c);
  if (!live) return null;
  const p = c.progress;
  return (
    <div className={`training-progress ${large ? "large" : ""}`}>
      {large && <strong className="training-stage">{live.label}</strong>}
      {live.fraction != null && (
        <div
          className="meter"
          role="progressbar"
          aria-label={`Candidate ${c.idx + 1} training progress`}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(live.fraction * 100)}
          aria-valuetext={live.label}
        >
          <i style={{ width: `${live.fraction * 100}%` }} />
        </div>
      )}
      {large && p && p.total_steps > 0 && (
        <span className="subtle">
          {Math.min(p.steps, p.total_steps).toLocaleString("en-US")} /{" "}
          {p.total_steps.toLocaleString("en-US")} steps ·{" "}
          {Math.round(p.wall_s)} s
        </span>
      )}
      <Sparkline
        curve={p?.curve}
        width={large ? 240 : 90}
        height={large ? 56 : 28}
      />
    </div>
  );
}
