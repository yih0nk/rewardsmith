import { Candidate } from "../api";
import TrainingProgress, { Sparkline, liveState } from "./TrainingProgress";

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
  return (
    <div className={`candidate-option ${selected ? "active" : ""}`}>
      <button
        className="candidate-select"
        aria-pressed={selected}
        onClick={onSelect}
      >
        <span>
          Candidate {c.idx + 1}
          {best ? " · Round winner" : ""}
        </span>
        <strong>
          {c.score != null
            ? `${c.score.toFixed(1)} / 10`
            : (live?.label ?? c.status.replaceAll("_", " "))}
        </strong>
        {c.error && (
          <span className="candidate-error">
            {c.status === "gate_failed" ? "Validation failed" : "Training failed"}
          </span>
        )}
      </button>
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
