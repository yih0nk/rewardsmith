import { Candidate } from "../api";
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
  return (
    <button
      className={`candidate-option ${selected ? "active" : ""}`}
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
          : c.status.replaceAll("_", " ")}
      </strong>
      {c.error && (
        <span className="candidate-error">
          {c.status === "gate_failed" ? "Validation failed" : "Training failed"}
        </span>
      )}
    </button>
  );
}
