import { RequestSummary } from "../api";
interface Props {
  requests: RequestSummary[];
  selected: string | null;
  onSelect: (id: string) => void;
}
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
          <span className="run-text">{run.text}</span>
          <small className="run-status">
            <span className={`lamp ${run.status}`} />
            {run.status}
            {run.best_score != null && (
              <span className="run-score">
                {run.best_score.toFixed(1)}
                <span aria-hidden="true">/10</span>
                <span className="sr-only"> out of 10</span>
              </span>
            )}
          </small>
        </button>
      ))}
    </nav>
  );
}
