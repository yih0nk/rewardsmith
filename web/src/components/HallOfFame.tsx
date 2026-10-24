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
      <section className="cmd hall-hero" aria-labelledby="hall-title">
        <p className="eyebrow">Hall of fame</p>
        <h1 id="hall-title">
          Reward hacking, <span className="em">caught on camera.</span>
        </h1>
        <p className="hero-sub">
          Candidates that earned their score by gaming the reward instead of
          doing the behavior.
        </p>
      </section>
      {error && (
        <div className="error" role="alert">
          {error} Retrying automatically.
        </div>
      )}
      {entries === null ? (
        !error && (
          <p className="cmd-note" role="status">
            Loading the hall of fame…
          </p>
        )
      ) : entries.length === 0 ? (
        <section className="cmd empty-state">
          <h2 className="cmd-title">Nothing flagged yet.</h2>
          <p>
            When a robot games its reward, open that candidate and choose “Flag
            as reward hacking”. It will show up here.
          </p>
          <div>
            <button className="primary" onClick={onBack}>
              Back to experiments
            </button>
          </div>
        </section>
      ) : (
        <section className="cmd" aria-labelledby="flagged-title">
          <h2 className="cmd-title" id="flagged-title">
            Flagged candidates
            <span className="subtle">{entries.length} in the hall</span>
          </h2>
          <div className="hall">
            {entries.map((entry) => (
              <HallCard key={entry.id} entry={entry} onOpen={onOpen} />
            ))}
          </div>
        </section>
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
      <div className="hall-media">
        <div className="well">
          {entry.video_path ? (
            <video
              src={artifactUrl(entry.video_path)}
              controls
              muted
              loop
              playsInline
              preload="metadata"
            />
          ) : (
            <div className="well-empty">
              <strong>No rollout video</strong>
            </div>
          )}
        </div>
      </div>
      <div className="hall-body">
        <span className="eyebrow">
          ROUND {entry.round_n} · CANDIDATE {entry.idx + 1} ·{" "}
          {entry.score?.toFixed(1) ?? "—"} / 10
        </span>
        <h3>{entry.request_text}</h3>
        <figure className="critic hall-note">
          <blockquote>{entry.flag_note}</blockquote>
          <figcaption>— Flag note</figcaption>
        </figure>
        <RewardTerms metrics={entry.metrics} />
        <button className="link" onClick={() => onOpen(entry.request_id)}>
          Open experiment →
        </button>
      </div>
    </article>
  );
}
