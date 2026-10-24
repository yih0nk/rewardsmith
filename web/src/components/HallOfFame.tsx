import { useEffect, useState } from "react";
import { HallEntry, artifactUrl, listHallOfFame } from "../api";
import RewardTerms from "./RewardTerms";

// Candidates flagged as reward hacking: the robot scored without doing the behavior.
export default function HallOfFame({
  onOpen,
  onBack,
}: {
  onOpen: (requestId: string) => void;
  onBack: () => void;
}) {
  const [entries, setEntries] = useState<HallEntry[] | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    const refresh = () =>
      listHallOfFame()
        .then((data) => {
          if (active) {
            setEntries(data);
            setError("");
          }
        })
        .catch((err) => {
          if (active) setError(`Cannot load the hall of fame: ${String(err)}`);
        });
    refresh();
    const timer = setInterval(refresh, 5000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">HALL OF FAME</div>
          <h1>Reward hacking, caught on camera.</h1>
          <p>
            Candidates that earned their score by gaming the reward instead of
            doing the behavior.
          </p>
        </div>
      </div>
      {error && (
        <div className="error" role="alert">
          {error} Retrying automatically.
        </div>
      )}
      {entries === null ? (
        !error && (
          <div className="loading" role="status">
            Loading the hall of fame…
          </div>
        )
      ) : entries.length === 0 ? (
        <section className="empty-state">
          <h2>Nothing flagged yet.</h2>
          <p>
            When a robot games its reward, open that candidate and choose “Flag
            as reward hacking”. It will show up here.
          </p>
          <button className="primary" onClick={onBack}>
            Back to experiments
          </button>
        </section>
      ) : (
        <div className="hall">
          {entries.map((entry) => (
            <HallCard key={entry.id} entry={entry} onOpen={onOpen} />
          ))}
        </div>
      )}
    </>
  );
}

export function HallCard({
  entry,
  onOpen,
}: {
  entry: HallEntry;
  onOpen: (requestId: string) => void;
}) {
  return (
    <article className="hall-card">
      {entry.video_path ? (
        <video src={artifactUrl(entry.video_path)} controls muted loop playsInline />
      ) : (
        <div className="video-empty">
          <span>No rollout video</span>
        </div>
      )}
      <div className="hall-body">
        <span className="eyebrow">
          ROUND {entry.round_n} · CANDIDATE {entry.idx + 1} ·{" "}
          {entry.score?.toFixed(1) ?? "—"} / 10
        </span>
        <h2>{entry.request_text}</h2>
        <p className="hall-note">“{entry.flag_note}”</p>
        <RewardTerms metrics={entry.metrics} />
        <button className="link" onClick={() => onOpen(entry.request_id)}>
          Open experiment ↗
        </button>
      </div>
    </article>
  );
}
