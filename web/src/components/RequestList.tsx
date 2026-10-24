import { RequestSummary } from "../api";

interface Props {
  requests: RequestSummary[];
  selected: string | null;
  onSelect: (id: string) => void;
}

export default function RequestList({ requests, selected, onSelect }: Props) {
  return (
    <>
      <h2 className="label">Requests</h2>
      {requests.length === 0 && <p className="muted">Nothing yet. Type a behavior above to start.</p>}
      <ul className="requests">
        {requests.map((r) => (
          <li key={r.id}>
            <button className={r.id === selected ? "req active" : "req"} onClick={() => onSelect(r.id)}>
              <span>{r.text}</span>
              <span className="muted">
                {r.status}
                {r.best_score != null && ` · ${r.best_score.toFixed(1)}`}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </>
  );
}
