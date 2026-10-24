import { Candidate } from "../api";
import { candidateName, candidateState } from "../stages";
import TrainingProgress, {
  Sparkline,
  liveState,
  textBar,
} from "./TrainingProgress";

// A div rather than one big button, so the progress bar keeps its role;
// the button stretches over the card, so the whole card still selects it.
export default function CandidateCard({
  c,
  round,
  best,
  overall = false,
  selected,
  onSelect,
}: {
  c: Candidate;
  round: number;
  // Its round's winner.
  best: boolean;
  // The best of the whole run.
  overall?: boolean;
  selected: boolean;
  onSelect: () => void;
}) {
  const live = liveState(c);
  const state = candidateState(c, live?.label);
  return (
    <div
      className={`candidate-option ${selected ? "active" : ""} ${best ? "winner" : ""} ${state.kind}`}
    >
      <button
        className="candidate-select"
        aria-pressed={selected}
        onClick={onSelect}
      >
        <span className="candidate-head">
          <span className="label">{candidateName(round, c.idx)}</span>
          {overall ? (
            <span className="winner-tag">Best overall</span>
          ) : (
            best && <span className="winner-tag">Round winner</span>
          )}
        </span>
        {c.score != null ? (
          <strong className="candidate-score num">
            {c.score.toFixed(1)}
            <small> / 10</small>
          </strong>
        ) : (
          <strong className="candidate-state">
            {state.kind === "failed" && <span aria-hidden="true">✕ </span>}
            {state.text}
          </strong>
        )}
        {state.why && <span className="candidate-error">{state.why}</span>}
      </button>
      {c.score != null && (
        <span className="score-bar" aria-hidden="true">
          {textBar(c.score / 10, 10)}
        </span>
      )}
      {live ? (
        <TrainingProgress c={c} />
      ) : (
        // Once scored, keep the curve small as a record of how it learned.
        (c.progress?.curve?.length ?? 0) > 1 && (
          <Sparkline
            curve={c.progress?.curve}
            width={90}
            caption="how it learned"
          />
        )
      )}
    </div>
  );
}
