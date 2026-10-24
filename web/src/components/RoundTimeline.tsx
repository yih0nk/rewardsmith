import { Round } from "../api";

const scored = (r: Round) =>
  r.candidates.filter(
    (c): c is typeof c & { score: number } =>
      typeof c.score === "number" && Number.isFinite(c.score),
  );

// A round's headline score: its winner's, or failing that its best scored candidate's.
export function roundScore(r: Round): number | null {
  const winner = r.candidates.find((c) => c.id === r.best_candidate_id);
  if (typeof winner?.score === "number" && Number.isFinite(winner.score))
    return winner.score;
  const scores = scored(r).map((c) => c.score);
  return scores.length ? Math.max(...scores) : null;
}

// Each round's score next to the change from the last scored round before it.
export function roundDeltas(rounds: Round[]) {
  let previous: number | null = null;
  return rounds.map((r) => {
    const score = roundScore(r);
    const delta = score != null && previous != null ? score - previous : null;
    if (score != null) previous = score;
    return { round: r, score, delta };
  });
}

// Every candidate's score as a character on a 0–10 strip in half-point cells:
// "o" for a candidate, "O" for the round's winner.
export function scoreStrip(r: Round, cells = 21): string[] {
  const strip = Array<string>(cells).fill("·");
  for (const c of scored(r)) {
    const i = Math.round(
      (Math.min(10, Math.max(0, c.score)) / 10) * (cells - 1),
    );
    if (c.id === r.best_candidate_id) strip[i] = "O";
    else if (strip[i] !== "O") strip[i] = "o";
  }
  return strip;
}

export const formatDelta = (d: number) =>
  `${d > 0 ? "+" : d < 0 ? "−" : "±"}${Math.abs(d).toFixed(1)}`;

// One row per round: the winner's score, the change since the previous
// round and every candidate's score on a shared 0–10 strip.
export default function RoundTimeline({
  rounds,
  current,
  bestScore,
  pending,
  onSelect,
}: {
  rounds: Round[];
  current: number | undefined;
  bestScore: number | null;
  pending: string | null;
  onSelect: (n: number) => void;
}) {
  return (
    <div className="rounds">
      {roundDeltas(rounds).map(({ round: r, score, delta }) => {
        const active = current === r.n;
        const best = score != null && score === bestScore;
        return (
          <button
            key={r.id}
            aria-pressed={active}
            className={`round ${active ? "active" : ""} ${best ? "best" : ""}`}
            onClick={() => onSelect(r.n)}
          >
            <span className="label">Round {String(r.n).padStart(2, "0")}</span>
            <span className="round-score num">
              {score?.toFixed(1) ?? "—"}
              <small>/10</small>
            </span>
            <span
              className={`delta num ${delta == null ? "" : delta > 0.05 ? "up" : delta < -0.05 ? "down" : "flat"}`}
            >
              {delta != null && (
                <>
                  {formatDelta(delta)}
                  <span className="sr-only"> from the previous round</span>
                </>
              )}
            </span>
            <span className="track" aria-hidden="true">
              {scoreStrip(r).map((ch, i) =>
                ch === "O" ? <b key={i}>O</b> : ch,
              )}
            </span>
            <span className="round-meta">
              {r.candidates.length} candidates · {r.status}
            </span>
            <span className="round-best">{best && "Best"}</span>
          </button>
        );
      })}
      {pending && (
        <div className="round pending" role="status">
          <span className="spinner" aria-hidden="true" />
          <strong>{pending}</strong>
          <span className="round-meta">Waiting for the next worker update</span>
        </div>
      )}
    </div>
  );
}
