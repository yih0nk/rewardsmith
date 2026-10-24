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
      <section className="masthead">
        <div className="eyebrow">Hall of fame</div>
        <h1>
          Reward hacking, <em>caught on camera.</em>
        </h1>
        <p>
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
          <div className="loading hall-loading" role="status">
            <span className="label">Loading the hall of fame…</span>
            <i />
            <i />
          </div>
        )
      ) : entries.length === 0 ? (
        <section className="empty-state">
          <svg viewBox="0 0 240 64" aria-hidden="true">
            <path
              className="grid"
              d="M0 16H240M0 32H240M0 48H240M40 0V64M80 0V64M120 0V64M160 0V64M200 0V64"
            />
            <path className="trace" d="M0 40H240" />
          </svg>
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
              <span className="label">No signal</span>
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
        <h2>{entry.request_text}</h2>
        <blockquote className="hall-note">{entry.flag_note}</blockquote>
        <RewardTerms metrics={entry.metrics} />
        <button className="link" onClick={() => onOpen(entry.request_id)}>
          Open experiment →
        </button>
      </div>
    </article>
  );
}
