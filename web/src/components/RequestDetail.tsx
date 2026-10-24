import { useRef, useState } from "react";
import { RequestDetail as Detail, Health, artifactUrl } from "../api";
import CandidateCard from "./CandidateCard";
import DiffView from "./DiffView";
import FlagControl from "./FlagControl";
import RewardTerms from "./RewardTerms";
import RoundGallery from "./RoundGallery";
import RoundTimeline, { formatDelta, roundScore } from "./RoundTimeline";
import TrainingProgress, { RewardCurve, liveState } from "./TrainingProgress";

export default function RequestDetail({
  detail,
  health,
}: {
  detail: Detail;
  health: Health | null;
}) {
  const [round, setRound] = useState<number | null>(null);
  const [candidateId, setCandidateId] = useState<number | null>(null);
  // Rounds and candidates are picked below the stage; bring it back into view when it has scrolled away.
  const stageRef = useRef<HTMLElement>(null);
  const show = (n: number, id: number | null) => {
    setRound(n);
    setCandidateId(id);
    const stage = stageRef.current;
    if (stage && stage.getBoundingClientRect().top < 0) {
      const still = window.matchMedia?.(
        "(prefers-reduced-motion: reduce)",
      ).matches;
      stage.scrollIntoView({
        behavior: still ? "auto" : "smooth",
        block: "start",
      });
    }
  };
  const scoredAll = detail.rounds.flatMap((r) =>
    r.candidates
      .filter((c) => typeof c.score === "number" && Number.isFinite(c.score))
      .map((c) => ({ round: r, c, score: c.score as number })),
  );
  const top = scoredAll.find(({ c }) => c.id === detail.best_candidate_id) ??
    scoredAll.reduce<(typeof scoredAll)[number] | null>((a, b) => {
      if (!a) return b;
      const aPass = a.c.check_result?.pass !== false;
      const bPass = b.c.check_result?.pass !== false;
      return bPass !== aPass ? (bPass ? b : a) : (b.score > a.score ? b : a);
    }, null);
  const bestRound = detail.status === "done" ? top?.round : undefined;
  const current =
    detail.rounds.find((r) => r.n === round) ??
    bestRound ??
    detail.rounds.at(-1);
  const candidate =
    current?.candidates.find((c) => c.id === candidateId) ??
    current?.candidates.find((c) => c.id === current.best_candidate_id) ??
    // Mid-round, show a candidate still training (the furthest along) so the stage has something moving.
    current?.candidates
      .filter((c) => liveState(c)?.fraction != null)
      .sort(
        (a, b) =>
          Number(b.status === "training") - Number(a.status === "training") ||
          liveState(b)!.fraction! - liveState(a)!.fraction!,
      )[0] ??
    current?.candidates.at(-1);
  const running = detail.status === "running" || detail.status === "queued";
  const live = candidate ? liveState(candidate) : null;
  const stage =
    detail.status === "queued"
      ? "Queued · waiting for worker"
      : (detail.events.at(-1)?.message ?? "Waiting for worker updates");
  const firstScore =
    detail.rounds.map(roundScore).find((s): s is number => s != null) ?? null;
  const roundsDone = detail.rounds.filter((r) => r.status === "done").length;
  const candidatesTotal = detail.rounds.reduce(
    (n, r) => n + r.candidates.length,
    0,
  );
  const statusText =
    detail.status === "queued"
      ? "Queued"
      : running
        ? "In progress"
        : detail.status === "failed"
          ? "Failed"
          : "Completed";
  const metrics = Object.entries(candidate?.metrics ?? {}).filter(
    (entry): entry is [string, number] =>
      typeof entry[1] === "number" && Number.isFinite(entry[1]),
  );
  return (
    <article className="experiment">
      <header className="exp-head">
        <div>
          <span className="eyebrow">
            Experiment <span className="num">#{detail.id}</span>
          </span>
          <h2>{detail.text}</h2>
        </div>
        <span className={`badge ${detail.status}`}>
          <span className={`lamp ${detail.status}`} />
          {statusText}
        </span>
      </header>
      <div className={`pipeline ${running ? "running" : ""}`}>
        <span className="pipeline-label" role="status">
          {detail.status === "failed"
            ? "Experiment failed"
            : running
              ? stage
              : "Experiment complete"}
        </span>
        <span className="round-count">
          {roundsDone} {roundsDone === 1 ? "round" : "rounds"} completed
        </span>
      </div>
      {detail.success_check_error ? (
        <p role="status">Success could not be verified: the checker failed.</p>
      ) : detail.status === "done" && detail.succeeded != null ? (
        <p role="status">{detail.succeeded ? "Success verified" : "Success criteria not met"}</p>
      ) : null}
      <dl className="summary">
        <div className="summary-best">
          <dt className="label">Best score</dt>
          <dd className="num">
            {top ? top.score.toFixed(1) : "—"}
            <small>/10</small>
          </dd>
          {top && (
            <dd className="summary-note">
              Round {top.round.n} · Candidate {top.c.idx + 1}
            </dd>
          )}
        </div>
        <div>
          <dt className="label">Since round 1</dt>
          <dd
            className={`num delta ${top && firstScore != null ? (top.score - firstScore > 0.05 ? "up" : "flat") : ""}`}
          >
            {top && firstScore != null
              ? formatDelta(top.score - firstScore)
              : "—"}
          </dd>
          {top && firstScore != null && (
            <dd className="summary-note">
              {firstScore.toFixed(1)} → {top.score.toFixed(1)}
            </dd>
          )}
        </div>
        <div>
          <dt className="label">Rounds</dt>
          <dd className="num">{roundsDone}</dd>
          <dd className="summary-note">{detail.rounds.length} started</dd>
        </div>
        <div>
          <dt className="label">Rewards tried</dt>
          <dd className="num">{candidatesTotal}</dd>
          <dd className="summary-note">
            {scoredAll.length} scored by the critic
          </dd>
        </div>
      </dl>

      <div className="result-grid">
        <section className="stage" aria-label="Robot rollout" ref={stageRef}>
          <div className="stage-head">
            <span className="label">
              Round {current?.n ?? "—"}
              {candidate ? ` · Candidate ${candidate.idx + 1}` : ""}
            </span>
            <span
              className={`tag ${candidate?.video_path ? "rollout" : live ? "live" : ""}`}
            >
              {candidate?.video_path ? "ROLLOUT" : live ? "LIVE" : "NO FOOTAGE"}
            </span>
          </div>
          <div className="well">
            {candidate?.video_path ? (
              <video
                key={candidate.id}
                src={artifactUrl(candidate.video_path)}
                controls
                muted
                loop
                playsInline
              />
            ) : candidate && live ? (
              <TrainingProgress c={candidate} large />
            ) : (
              <div className="well-empty">
                <span className="label">No signal</span>
                <strong>
                  {candidate?.status === "gate_failed"
                    ? "Reward did not pass validation"
                    : candidate?.status === "failed"
                      ? "Candidate failed"
                      : candidate?.status === "stopped"
                        ? "Stopped early: another candidate looked more promising"
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
          </div>
          <RewardCurve
            key={candidate?.id}
            curve={candidate?.progress?.curve}
            live={Boolean(live)}
          />
          <div className="stage-foot">
            <span>
              {candidate?.video_path
                ? "Recorded rollout · use video controls"
                : live
                  ? "Training live · updates automatically"
                  : "No recorded footage"}
            </span>
            <span>MuJoCo Playground · Unitree Go1</span>
          </div>
        </section>
        <section className="evaluation" aria-label="Evaluation">
          <div className="label">
            {candidate
              ? `Candidate · ${candidate.status.replaceAll("_", " ")}`
              : "Waiting for candidates"}
          </div>
          <div className={`score ${candidate?.score == null ? "none" : ""}`}>
            <span className="num">{candidate?.score?.toFixed(1) ?? "—"}</span>
            <span className="score-of num">/ 10</span>
          </div>
          <span className="score-label label">Critic score</span>
          {candidate?.check_result && (
            <p>Success check: {candidate.check_result.verified === false ? "unavailable" : candidate.check_result.pass ? "passed" : "failed"}</p>
          )}
          <dl className="metrics">
            {metrics
              .filter(([key]) => !key.startsWith("term_"))
              .map(([key, value]) => (
                <div className="metric" key={key}>
                  <dt>{key.replaceAll("_", " ")}</dt>
                  <dd className="num">{Number(value.toFixed(4))}</dd>
                </div>
              ))}
          </dl>
          {metrics.length === 0 && (
            <p className="subtle">No metrics reported yet.</p>
          )}
          <figure className="critic">
            <figcaption className="label">Critic notes</figcaption>
            <blockquote>
              {candidate?.critique ??
                (running
                  ? "Waiting for the critic to evaluate this candidate."
                  : "No critique was returned for this candidate.")}
            </blockquote>
          </figure>
          {(candidate?.status === "trained" ||
            candidate?.status === "scored") && (
            <FlagControl key={candidate.id} candidate={candidate} />
          )}
        </section>
      </div>

      <div className="section-heading">
        <h3>Learning, one round at a time</h3>
        <span className="subtle">
          {detail.rounds.length} rounds · select one to explore
        </span>
      </div>
      <RoundTimeline
        rounds={detail.rounds}
        current={current?.n}
        bestScore={top?.score ?? null}
        bestRound={top?.round.n}
        pending={running ? stage : null}
        onSelect={(n) => show(n, null)}
      />

      {current && (
        <>
          <div className="section-heading">
            <h3>Candidates in round {current.n}</h3>
            <span className="subtle">
              {current.candidates.length} reward functions · select one to load
              it on the stage
            </span>
          </div>
          <div className="candidate-list" aria-label="Candidates">
            {current.candidates.map((c) => (
              <CandidateCard
                key={c.id}
                c={c}
                best={c.id === current.best_candidate_id}
                selected={candidate?.id === c.id}
                onSelect={() => show(current.n, c.id)}
              />
            ))}
          </div>
        </>
      )}
      {candidate?.error && (
        <pre className="error" role="alert">
          {candidate.error}
        </pre>
      )}
      <RewardTerms metrics={candidate?.metrics ?? null} />
      <RoundGallery rounds={detail.rounds} onPick={(n, id) => show(n, id)} />
      <div className="technical">
        {candidate && (
          <>
            <details>
              <summary>
                Reward function <span>View generated Python</span>
              </summary>
              <pre className="code">{candidate.reward_src}</pre>
            </details>
            {candidate.diff && (
              <details>
                <summary>
                  What changed <span>Revision diff</span>
                </summary>
                <DiffView diff={candidate.diff} />
              </details>
            )}
          </>
        )}
        <details>
          <summary>
            Activity log <span>{detail.events.length} events</span>
          </summary>
          {detail.events.length ? (
            <ol className="log">
              {detail.events.map((event) => (
                <li key={event.id}>
                  <time
                    dateTime={new Date(event.created_at * 1000).toISOString()}
                  >
                    {new Date(event.created_at * 1000).toLocaleTimeString()}
                  </time>
                  <span className="log-kind">{event.kind}</span>
                  <span>{event.message}</span>
                </li>
              ))}
            </ol>
          ) : (
            <p>No events yet.</p>
          )}
        </details>
      </div>
    </article>
  );
}
