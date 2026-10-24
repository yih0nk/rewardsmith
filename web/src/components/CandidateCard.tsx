import { Candidate } from "../api";
import TrainingProgress, {
  Sparkline,
  liveState,
  textBar,
} from "./TrainingProgress";

// A div rather than one big button, so the progress bar keeps its role;
// the button stretches over the card, so the whole card still selects it.
export default function CandidateCard({
  c,
  best,
  selected,
  onSelect,
}: {
  c: Candidate;
  best: boolean;
  selected: boolean;
  onSelect: () => void;
}) {
  const live = liveState(c);
  const failed = c.status === "failed" || c.status === "gate_failed";
  return (
    <div
      className={`candidate-option ${selected ? "active" : ""} ${best ? "winner" : ""} ${failed ? "failed" : ""} ${live ? "live" : ""}`}
    >
      <button
        className="candidate-select"
        aria-pressed={selected}
        onClick={onSelect}
      >
        <span className="candidate-head">
          <span className="label">Candidate {c.idx + 1}</span>
          {best && <span className="winner-tag">Winner</span>}
        </span>
        {c.score != null ? (
          <strong className="candidate-score num">
            {c.score.toFixed(1)}
            <small> / 10</small>
          </strong>
        ) : (
          <strong className="candidate-state">
            {live?.label ?? c.status.replaceAll("_", " ")}
          </strong>
        )}
        {c.error && (
          <span className="candidate-error">
            {c.status === "gate_failed"
              ? "Validation failed"
              : c.status === "stopped"
                ? "Stopped early"
                : "Failed"}
          </span>
        )}
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
