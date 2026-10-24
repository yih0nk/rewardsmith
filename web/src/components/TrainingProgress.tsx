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
  if (c.status === "scored" || c.status === "failed" || c.status === "gate_failed" || c.status === "stopped")
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

// A fraction drawn in characters, e.g. "[#####.....]".
export const textBar = (fraction: number, width: number) => {
  const n = Math.round(clamp01(fraction) * width);
  return `[${"#".repeat(n)}${".".repeat(width - n)}]`;
};

export const formatReward = (v: number) =>
  Math.abs(v) >= 100 ? String(Math.round(v)) : String(Number(v.toFixed(2)));

export interface SparklineShape {
  points: { x: number; y: number }[];
  latest: number;
  min: number;
  max: number;
  firstStep: number;
  lastStep: number;
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
    firstStep: s0,
    lastStep: s1,
  };
}

const describe = (shape: SparklineShape) =>
  `Episode reward over ${shape.points.length} ${
    shape.points.length === 1 ? "eval" : "evals"
  }, from ${formatReward(shape.min)} to ${formatReward(shape.max)}, latest ${formatReward(shape.latest)}`;

export const formatSteps = (n: number) =>
  n >= 1e6
    ? `${Number((n / 1e6).toFixed(1))}M`
    : n >= 1e3
      ? `${Number((n / 1e3).toFixed(1))}k`
      : String(Math.round(n));

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
  const summary = describe(shape);
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
          <span aria-hidden="true">{textBar(live.fraction, large ? 30 : 16)}</span>
        </div>
      )}
      {large && p && p.total_steps > 0 && (
        <span className="subtle">
          {Math.min(p.steps, p.total_steps).toLocaleString("en-US")} /{" "}
          {p.total_steps.toLocaleString("en-US")} steps ·{" "}
          {Math.round(p.wall_s)} s
        </span>
      )}
      {!large && <Sparkline curve={p?.curve} width={90} height={28} />}
    </div>
  );
}

// The stage's full learning curve: episode reward against training steps.
// The SVG stretches to its box, so labels and the last-point dot are HTML.
export function RewardCurve({
  curve,
  live = false,
}: {
  curve: CurvePoint[] | null | undefined;
  live?: boolean;
}) {
  const W = 600;
  const H = 140;
  const shape = sparkline(curve, W, H, 8);
  if (!shape)
    return (
      <figure className="curve empty">
        <figcaption>
          <span className="label">Episode reward</span>
        </figcaption>
        <div className="curve-plot">
          <span className="curve-note">
            {live ? "First evaluation in a few seconds…" : "No learning curve recorded"}
          </span>
        </div>
      </figure>
    );
  const pts = shape.points;
  const last = pts[pts.length - 1];
  const line = pts.map((p, i) => `${i ? "L" : "M"}${p.x} ${p.y}`).join(" ");
  const area =
    pts.length > 1 ? `${line} L${last.x} ${H} L${pts[0].x} ${H} Z` : "";
  return (
    <figure className={`curve ${live ? "live" : ""}`}>
      <figcaption>
        <span className="label">Episode reward</span>
        <span className="curve-latest">
          <span className="label">{live ? "now" : "final"}</span>
          <strong className="num">{formatReward(shape.latest)}</strong>
        </span>
      </figcaption>
      <div className="curve-plot">
        <span className="curve-y top num">{formatReward(shape.max)}</span>
        <span className="curve-y bottom num">{formatReward(shape.min)}</span>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="none"
          role="img"
          aria-label={describe(shape)}
        >
          <path
            className="curve-grid"
            d={`M0 ${H / 4}H${W}M0 ${H / 2}H${W}M0 ${(3 * H) / 4}H${W}`}
          />
          {area && <path className="curve-area" d={area} />}
          <path className="curve-line" d={line} />
        </svg>
        <i
          className="curve-dot"
          style={{ left: `${(last.x / W) * 100}%`, top: `${(last.y / H) * 100}%` }}
        />
      </div>
      <div className="curve-x num">
        <span>{formatSteps(shape.firstStep)}</span>
        <span>steps</span>
        <span>{formatSteps(shape.lastStep)}</span>
      </div>
    </figure>
  );
}
