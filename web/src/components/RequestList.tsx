import { RequestSummary } from "../api";
interface Props {
  requests: RequestSummary[];
  selected: string | null;
  onSelect: (id: string) => void;
}
// One row per run: status, best score, request. The selected row is filled green.
export default function RequestList({ requests, selected, onSelect }: Props) {
  return (
    <nav aria-label="Recent experiments" className="runs">
      {requests.length === 0 && <p className="subtle">No experiments yet.</p>}
      {requests.map((run) => (
        <button
          key={run.id}
          aria-pressed={run.id === selected}
          onClick={() => onSelect(run.id)}
          className={`run-link ${run.id === selected ? "selected" : ""}`}
        >
          <span className={`run-status ${run.status}`}>{run.status}</span>
          <span className="run-score num">
            {run.best_score != null ? (
              <>
                {run.best_score.toFixed(1)}
                <span aria-hidden="true">/10</span>
                <span className="sr-only"> out of 10</span>
              </>
            ) : (
              "—"
            )}
          </span>
          <span className="run-text">{run.text}</span>
        </button>
      ))}
    </nav>
  );
}
