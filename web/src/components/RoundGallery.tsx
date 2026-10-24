import { Round, artifactUrl } from "../api";

// Each round's winning rollout side by side, so round 1 and round 5 can be compared.
export default function RoundGallery({
  rounds,
  onPick,
}: {
  rounds: Round[];
  onPick: (round: number, candidateId: number) => void;
}) {
  const tiles = rounds.flatMap((round) => {
    const best = round.candidates.find((c) => c.id === round.best_candidate_id);
    return best?.video_path ? [{ round, best, video: best.video_path }] : [];
  });
  if (tiles.length < 2) return null;
  return (
    <section aria-label="Best rollout of each round">
      <div className="section-heading">
        <h2>
          Every round, side by side<span>{tiles.length} rollouts</span>
        </h2>
      </div>
      <div className="gallery">
        {tiles.map(({ round, best, video }) => (
          <figure key={round.id}>
            <video src={artifactUrl(video)} autoPlay muted loop playsInline />
            <figcaption>
              <button
                className="link"
                onClick={() => onPick(round.n, best.id)}
              >
                Round {round.n} · {best.score?.toFixed(1) ?? "—"} / 10
              </button>
            </figcaption>
          </figure>
        ))}
      </div>
    </section>
  );
}
