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
  // Several looping videos at once is a lot of motion; let them sit still when asked.
  const still =
    typeof window !== "undefined" &&
    window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  return (
    <section aria-label="Best rollout of each round">
      <div className="section-heading">
        <h3>Every round, side by side</h3>
        <span className="subtle">{tiles.length} winning rollouts</span>
      </div>
      <div className="gallery">
        {tiles.map(({ round, best, video }) => (
          <figure key={round.id}>
            <video
              src={artifactUrl(video)}
              autoPlay={!still}
              controls={still}
              muted
              loop
              playsInline
            />
            <figcaption>
              <button className="link" onClick={() => onPick(round.n, best.id)}>
                Round {round.n}
                <span className="num">
                  {best.score?.toFixed(1) ?? "—"} / 10
                </span>
              </button>
            </figcaption>
          </figure>
        ))}
      </div>
    </section>
  );
}
