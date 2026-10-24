import { RequestDetail as Detail } from "../api";
import CandidateCard from "./CandidateCard";

export default function RequestDetail({ detail }: { detail: Detail }) {
  return (
    <div>
      <h2>{detail.text}</h2>
      <p className="muted">
        {detail.status} · best score {detail.best_score?.toFixed(1) ?? "n/a"} · link: {location.origin}/#{detail.id}
      </p>
      {detail.rounds.map((round) => (
        <div key={round.id} className="round">
          <h3 className="label">Round {round.n}</h3>
          <div className="grid">
            {round.candidates.map((c) => (
              <CandidateCard key={c.id} c={c} best={c.id === round.best_candidate_id} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
