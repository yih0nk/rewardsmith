import { useState } from "react";
import {
  RequestDetail as Detail,
  Health,
  MOCK_MODE,
  artifactUrl,
} from "../api";
import CandidateCard from "./CandidateCard";
import RobotScene from "./RobotScene";

export default function RequestDetail({
  detail,
  health,
}: {
  detail: Detail;
  health: Health | null;
}) {
  const [round, setRound] = useState<number | null>(null);
  const [candidateId, setCandidateId] = useState<number | null>(null);
  const [playing, setPlaying] = useState(true);
  const bestRound =
    detail.status === "done" && detail.best_score != null
      ? detail.rounds.find((r) =>
          r.candidates.some((c) => c.score === detail.best_score),
        )
      : undefined;
  const current =
    detail.rounds.find((r) => r.n === round) ??
    bestRound ??
    detail.rounds.at(-1);
  const candidate =
    current?.candidates.find((c) => c.id === candidateId) ??
    current?.candidates.find((c) => c.id === current.best_candidate_id) ??
    current?.candidates.at(-1);
  const running = detail.status === "running" || detail.status === "queued";
  const stage =
    detail.status === "queued"
      ? "Queued · waiting for worker"
      : (detail.events.at(-1)?.message ?? "Waiting for worker updates");
  const metrics = Object.entries(candidate?.metrics ?? {}).filter(
    (entry): entry is [string, number] =>
      typeof entry[1] === "number" && Number.isFinite(entry[1]),
  );
  const terms = metrics.filter(([key]) => key.startsWith("term_"));
  return (
    <>
      <div className="experiment-title">
        <div>
          <span className="eyebrow">CURRENT EXPERIMENT</span>
          <h2>{detail.text}</h2>
        </div>
        <span className={`badge ${running ? "working" : ""}`}>
          <span
            className={`status-dot ${detail.status === "failed" ? "red" : detail.status === "done" ? "green" : ""}`}
          />
          {detail.status === "queued"
            ? "Queued"
            : running
              ? "In progress"
              : detail.status === "failed"
                ? "Failed"
                : "Completed"}
        </span>
      </div>
      <div className="pipeline">
        <span className="pipeline-label" role="status">
          {detail.status === "failed"
            ? "Experiment failed"
            : running
              ? stage
              : "Experiment complete"}
        </span>
        <span className="round-count">
          {detail.rounds.filter((r) => r.status === "done").length} rounds
          completed
        </span>
      </div>
      <div className="result-grid">
        <section className="preview-card" aria-label="Robot rollout">
          <div className="panel-heading">
            <span>
              Policy preview{" "}
              <span className="subtle">
                / Round {current?.n ?? "—"}
                {candidate ? ` · Candidate ${candidate.idx + 1}` : ""}
              </span>
            </span>
            <span className="mini-badge">
              {candidate?.video_path
                ? "ROLLOUT"
                : MOCK_MODE
                  ? "ILLUSTRATION"
                  : "NO FOOTAGE"}
            </span>
          </div>
          {candidate?.video_path ? (
            <video
              key={candidate.id}
              src={artifactUrl(candidate.video_path)}
              controls
              muted
              loop
              playsInline
            />
          ) : MOCK_MODE ? (
            <RobotScene playing={playing} round={current?.n ?? 0} />
          ) : (
            <div className="video-empty">
              <strong>
                {candidate?.status === "gate_failed"
                  ? "Reward did not pass validation"
                  : candidate?.status === "failed"
                    ? "Candidate failed"
                    : health?.fake_sim
                      ? "Fake simulator · no video generated"
                      : "No rollout video yet"}
              </strong>
              <span>
                {running
                  ? "Results update automatically while the worker runs."
                  : "This candidate has no recorded footage."}
              </span>
            </div>
          )}
          <div className="playback">
            {MOCK_MODE && !candidate?.video_path && (
              <button
                aria-label={playing ? "Pause animation" : "Play animation"}
                onClick={() => setPlaying(!playing)}
              >
                {playing ? "Ⅱ" : "▶"}
              </button>
            )}
            <span>
              {candidate?.video_path
                ? "Recorded rollout · use video controls"
                : MOCK_MODE
                  ? "Animated placeholder · no training footage"
                  : "No recorded footage"}
            </span>
            <span className="playback-end">MuJoCo Playground</span>
          </div>
        </section>
        <section className="evaluation">
          <div className="eyebrow">
            {candidate
              ? `CANDIDATE · ${candidate.status.replaceAll("_", " ")}`
              : "WAITING FOR CANDIDATES"}
          </div>
          <div className="score">
            {candidate?.score?.toFixed(1) ?? "—"}
            <span>/ 10</span>
            <span className="score-label">Critic score</span>
          </div>
          <div className="metrics">
            {metrics
              .filter(([key]) => !key.startsWith("term_"))
              .map(([key, value]) => (
                <div className="metric" key={key}>
                  <div>
                    <span>{key.replaceAll("_", " ")}</span>
                    <strong>{Number(value.toFixed(4))}</strong>
                  </div>
                </div>
              ))}
            {metrics.length === 0 && (
              <p className="subtle">No metrics reported yet.</p>
            )}
          </div>
          <div className="critic">
            <span>✧ CRITIC NOTES</span>
            <p>
              {candidate?.critique ??
                (running
                  ? "Waiting for the critic to evaluate this candidate."
                  : "No critique was returned for this candidate.")}
            </p>
          </div>
        </section>
      </div>
      <div className="section-heading">
        <h2>
          Learning, one round at a time
          <span>{detail.rounds.length} rounds</span>
        </h2>
        <span className="subtle">Select a round to explore</span>
      </div>
      <div className="attempts">
        {detail.rounds.map((r) => {
          const c =
            r.candidates.find((c) => c.id === r.best_candidate_id) ??
            r.candidates.at(-1);
          return (
            <button
              key={r.id}
              aria-pressed={current?.n === r.n}
              className={`attempt ${current?.n === r.n ? "active" : ""}`}
              onClick={() => {
                setRound(r.n);
                setCandidateId(null);
              }}
            >
              <div className="attempt-top">
                <span>ROUND {String(r.n).padStart(2, "0")}</span>
                {c?.score != null && c.score === detail.best_score && (
                  <span className="best">↗ BEST RESULT</span>
                )}
              </div>
              <div className="attempt-score">
                {c?.score?.toFixed(1) ?? "—"}
                <small>/ 10</small>
                <span>
                  {r.candidates.length} candidates · {r.status}
                </span>
              </div>
            </button>
          );
        })}
        {running && (
          <div className="attempt pending">
            <span className="spinner" />
            <strong>{stage}</strong>
            <span>Waiting for the next worker update</span>
          </div>
        )}
      </div>
      {current && (
        <div className="candidate-list" aria-label="Candidates">
          {current.candidates.map((c) => (
            <CandidateCard
              key={c.id}
              c={c}
              best={c.id === current.best_candidate_id}
              selected={candidate?.id === c.id}
              onSelect={() => setCandidateId(c.id)}
            />
          ))}
        </div>
      )}
      {candidate?.error && (
        <pre className="error" role="alert">
          {candidate.error}
        </pre>
      )}
      {terms.length > 0 && (
        <section className="reward-terms">
          <h2>Reward breakdown</h2>
          {terms.map(([key, value]) => (
            <div className="metric" key={key}>
              <div>
                <span>{key.slice(5).replaceAll("_", " ")}</span>
                <strong>{Number(value.toFixed(4))}</strong>
              </div>
              <div className="meter">
                <i
                  style={{
                    width: `${(Math.abs(value) / Math.max(...terms.map(([, v]) => Math.abs(v)), 0.001)) * 100}%`,
                    background: value < 0 ? "#b17c67" : undefined,
                  }}
                />
              </div>
            </div>
          ))}
          <p className="subtle">
            Bar lengths show relative magnitude within this candidate; negative
            terms are shown in brown.
          </p>
        </section>
      )}
      <div className="technical">
        {candidate && (
          <>
            <details>
              <summary>
                Reward function <span>View generated Python ↗</span>
              </summary>
              <pre>{candidate.reward_src}</pre>
            </details>
            {candidate.diff && (
              <details>
                <summary>
                  What changed <span>Revision diff ↗</span>
                </summary>
                <pre>{candidate.diff}</pre>
              </details>
            )}
          </>
        )}
        <details>
          <summary>
            Activity log <span>{detail.events.length} events ↗</span>
          </summary>
          {detail.events.length ? (
            detail.events.map((event) => (
              <p key={event.id}>
                <time
                  dateTime={new Date(event.created_at * 1000).toISOString()}
                >
                  {new Date(event.created_at * 1000).toLocaleTimeString()}
                </time>{" "}
                · {event.kind}: {event.message}
              </p>
            ))
          ) : (
            <p>No events yet.</p>
          )}
        </details>
      </div>
    </>
  );
}
