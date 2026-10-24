import { RequestSummary } from "../api";
interface Props {
  requests: RequestSummary[];
  selected: string | null;
  onSelect: (id: string) => void;
}
export default function RequestList({ requests, selected, onSelect }: Props) {
  return (
    <nav aria-label="Recent experiments">
      {requests.length === 0 && <p className="subtle">No experiments yet.</p>}
      {requests.map((run) => (
        <button
          key={run.id}
          aria-pressed={run.id === selected}
          onClick={() => onSelect(run.id)}
          className={`run-link ${run.id === selected ? "selected" : ""}`}
        >
          <span
            className={`status-dot ${run.status === "done" ? "green" : run.status === "failed" ? "red" : ""}`}
          />
          <span>
            {run.text}
            <small className="run-status">
              {run.status}
              {run.best_score != null
                ? ` · ${run.best_score.toFixed(1)} / 10`
                : ""}
            </small>
          </span>
        </button>
      ))}
    </nav>
  );
}
