import { useRef, useState } from "react";
import { RequestDetail as Detail, Health, artifactUrl } from "../api";
import {
  Explain,
  isListedMetric,
  metricInfo,
  successSentence,
  describeClause,
} from "../explain";
import { candidateName, candidateState, isScored } from "../stages";
import { QueuePlace, RenderReply, runAnchor } from "../thread";
import CandidateCard from "./CandidateCard";
import DiffView from "./DiffView";
import FlagControl from "./FlagControl";
import RewardTerms from "./RewardTerms";
import RoundGallery from "./RoundGallery";
import RoundTimeline, { formatDelta, roundScore } from "./RoundTimeline";
import RunStatus from "./RunStatus";
import TrainingProgress, { RewardCurve, liveState } from "./TrainingProgress";

const EVENT_KINDS: Record<string, string> = {
  queued: "queued",
  scene: "scene",
  check: "pass/fail",
  round: "round",
  gate: "code check",
  train: "training",
  halving: "early stop",
  done: "finished",
  error: "error",
};

// One turn of a conversation: the user's message and the run it started.
export default function RequestDetail({
  detail,
  health,
  onRetry,
  queue,
  renderReply,
  now,
}: {
  detail: Detail;
  health: Health | null;
  // Puts the prompt back in the composer so it can be edited and run again.
  onRetry?: (text: string) => void;
  // Where a queued run stands, from the request list.
  queue?: QueuePlace | null;
  // The reply control for the run and each candidate. Unset until follow-ups exist.
  renderReply?: RenderReply;
  now?: number;
}) {
  const [round, setRound] = useState<number | null>(null);
  const [candidateId, setCandidateId] = useState<number | null>(null);
  // Rounds and candidates are picked below the result; bring it back into view when it has scrolled away.
  const stageRef = useRef<HTMLElement>(null);
  const show = (n: number | null, id: number | null) => {
    setRound(n);
    setCandidateId(id);
    const stage = stageRef.current;
    if (id != null && stage && stage.getBoundingClientRect().top < 0) {
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
      .filter(isScored)
      .map((c) => ({ round: r, c, score: c.score as number })),
  );
  const top =
    scoredAll.find(({ c }) => c.id === detail.best_candidate_id) ??
    scoredAll.reduce<(typeof scoredAll)[number] | null>((a, b) => {
      if (!a) return b;
      const aPass = a.c.check_result?.pass !== false;
      const bPass = b.c.check_result?.pass !== false;
      return bPass !== aPass ? (bPass ? b : a) : b.score > a.score ? b : a;
    }, null);
  const running = detail.status === "running" || detail.status === "queued";
  const lastRound = detail.rounds.at(-1);
  // Mid-round with nothing scored yet, show a candidate still training (the furthest along).
  const liveCandidate = lastRound?.candidates
    .filter((c) => liveState(c)?.fraction != null)
    .sort(
      (a, b) =>
        Number(b.status === "training") - Number(a.status === "training") ||
        liveState(b)!.fraction! - liveState(a)!.fraction!,
    )[0];
  const picked = detail.rounds
    .flatMap((r) => r.candidates.map((c) => ({ round: r, c })))
    .find(({ c }) => c.id === candidateId);
  const shown =
    picked ??
    (top ? { round: top.round, c: top.c } : null) ??
    (lastRound && liveCandidate
      ? { round: lastRound, c: liveCandidate }
      : null) ??
    (lastRound?.candidates.length
      ? { round: lastRound, c: lastRound.candidates.at(-1)! }
      : null);
  const candidate = shown?.c;
  const viewingBest = Boolean(top && candidate?.id === top.c.id);
  const explored =
    detail.rounds.find((r) => r.n === round) ?? shown?.round ?? lastRound;
  const live = candidate ? liveState(candidate) : null;
  const firstScore =
    detail.rounds.map(roundScore).find((s): s is number => s != null) ?? null;
  const candidatesTotal = detail.rounds.reduce(
    (n, r) => n + r.candidates.length,
    0,
  );
  const statusText =
    detail.status === "queued"
      ? "Waiting"
      : running
        ? "In progress"
        : detail.status === "failed"
          ? "Failed"
          : scoredAll.length
            ? "Finished"
            : "Finished · no result";
  // Nothing to show and nothing coming: say so instead of an empty stage.
  const nothingWorked = !running && scoredAll.length === 0;
  const metrics = Object.entries(candidate?.metrics ?? {}).filter(
    (entry): entry is [string, number] =>
      typeof entry[1] === "number" && Number.isFinite(entry[1]),
  );
  const sentence = successSentence(detail.success_spec);
  const check = candidate?.check_result;
  const name = shown ? candidateName(shown.round.n, shown.c.idx) : null;
  const robustness = candidate?.robustness;
  // Before any candidate exists there is nothing to show but the status: give it the width.
  const showResult = Boolean(candidate) || nothingWorked;

  return (
    <article
      className="cmd turn"
      id={runAnchor(detail.id)}
      aria-labelledby={`exp-${detail.id}`}
    >
      <div className="exp-head">
        <span className="who">You</span>
        <h2 className="exp-title" id={`exp-${detail.id}`}>
          {detail.text}
        </h2>
        <span
          className={`badge ${detail.status} ${nothingWorked ? "empty" : ""}`}
        >
          {statusText}
        </span>
        <span className="exp-id num">
          Run #{detail.id}
          {Number.isFinite(detail.created_at) && (
            <>
              {" · "}
              <time dateTime={new Date(detail.created_at * 1000).toISOString()}>
                {new Date(detail.created_at * 1000).toLocaleTimeString([], {
                  hour: "numeric",
                  minute: "2-digit",
                })}
              </time>
            </>
          )}
        </span>
      </div>
      <div className="run-card">
        <div className={`run-grid ${showResult ? "" : "status-only"}`}>
          <div className="run-side">
            <RunStatus
              detail={detail}
              candidatesPerRound={health?.candidates_per_round}
              queue={queue}
              now={now}
            />
            <div className="success-spec">
              <span className="label">
                <Explain term="successCheck">Success check</Explain>
              </span>
              <div>
                <p>
                  {sentence ??
                    (running
                      ? "Claude writes a pass/fail test from your prompt before training starts."
                      : "This run has no pass/fail test.")}
                </p>
                {detail.success_check_error ? (
                  <p className="success-check" role="status">
                    Success could not be verified: the checker failed, so the
                    test fell back to “don't fall over”.
                  </p>
                ) : detail.status === "done" && detail.succeeded != null ? (
                  <p
                    className={`success-check ${detail.succeeded ? "pass" : ""}`}
                    role="status"
                  >
                    {detail.succeeded
                      ? "Success verified: the best robot passed the test with a high critic score."
                      : "Success criteria not met: no robot both passed the test and scored high enough."}
                  </p>
                ) : null}
              </div>
            </div>
          </div>

          {showResult && (
            <section
              className="result"
              aria-label="Robot rollout"
              ref={stageRef}
              data-candidate-id={candidate?.id}
            >
              <h3 className="cmd-title">
                <span>
                  {nothingWorked
                    ? "No result"
                    : viewingBest
                      ? running
                        ? "Best so far"
                        : "Best result"
                      : picked
                        ? "Selected candidate"
                        : "Live view"}
                </span>
                {name && !nothingWorked && (
                  <span className="result-who">
                    <span className="stage-title">{name}</span>
                    {candidate?.score != null && (
                      <span className="who-score num">
                        {candidate.score.toFixed(1)}
                        <small> / 10</small>
                      </span>
                    )}
                    {viewingBest && (
                      <span className="winner-tag">Best overall</span>
                    )}
                    <span
                      className={`tag ${candidate?.video_path ? "rollout" : live ? "live" : ""}`}
                    >
                      {candidate?.video_path
                        ? "VIDEO"
                        : live
                          ? "LIVE"
                          : "NO VIDEO"}
                    </span>
                  </span>
                )}
              </h3>
              {picked && top && !viewingBest && (
                <p className="viewing-note">
                  You picked this candidate below.{" "}
                  <button className="link" onClick={() => show(null, null)}>
                    Back to the best result
                  </button>
                </p>
              )}
              {nothingWorked ? (
                <RunProblem detail={detail} onRetry={onRetry} />
              ) : (
                <div className="result-grid">
                  <div className="stage">
                    <div className="well">
                      {candidate?.video_path ? (
                        <video
                          key={candidate.id}
                          src={artifactUrl(candidate.video_path)}
                          controls
                          muted
                          loop
                          autoPlay
                          playsInline
                        />
                      ) : candidate && live ? (
                        <TrainingProgress c={candidate} large />
                      ) : (
                        <div className="well-empty">
                          <strong>
                            {candidate?.status === "gate_failed" ||
                            candidate?.status === "failed" ||
                            candidate?.status === "stopped"
                              ? candidateState(candidate).text
                              : health?.fake_sim
                                ? "Fake simulator · no video generated"
                                : "No video for this candidate"}
                          </strong>
                          <span>
                            {(candidate && candidateState(candidate).why) ??
                              (running ? "This page updates by itself." : "")}
                          </span>
                        </div>
                      )}
                    </div>
                    <p className="stage-foot">
                      <span>
                        {candidate?.video_path
                          ? "A 10 s test run of the trained robot"
                          : live
                            ? "Training live · updates by itself"
                            : ""}
                      </span>
                      <span>MuJoCo Playground · Unitree Go1</span>
                    </p>
                  </div>
                  <div className="evaluation summary-best">
                    <p
                      className={`score ${candidate?.score == null ? "none" : ""}`}
                    >
                      <span className="num">
                        {candidate?.score?.toFixed(1) ?? "—"}
                      </span>
                      <span className="score-of num">/ 10</span>
                    </p>
                    <p className="score-label label">
                      <Explain term="critic">Critic score</Explain>
                      {candidate && candidate.score == null && (
                        <> · {candidateState(candidate, live?.label).text}</>
                      )}
                    </p>
                    {viewingBest &&
                      top &&
                      firstScore != null &&
                      top.round.n > 1 && (
                        <p className="improvement">
                          Since round 1{" "}
                          <span
                            className={`num delta ${top.score - firstScore > 0.05 ? "up" : ""}`}
                          >
                            {formatDelta(top.score - firstScore)}
                          </span>{" "}
                          <span className="subtle num">
                            ({firstScore.toFixed(1)} → {top.score.toFixed(1)})
                          </span>
                        </p>
                      )}
                    {check && (
                      <div
                        className={`check-result ${check.pass && check.verified !== false ? "pass" : ""}`}
                      >
                        <p>
                          Success check:{" "}
                          {check.verified === false
                            ? "unavailable"
                            : check.pass
                              ? "passed"
                              : "failed"}
                        </p>
                        {check.verified !== false &&
                          check.clauses?.length > 0 && (
                            <ul className="clauses">
                              {check.clauses.map((cl, i) => (
                                <li
                                  key={i}
                                  className={cl.pass ? "pass" : "fail"}
                                >
                                  <span aria-hidden="true">
                                    {cl.pass ? "✓" : "✕"}
                                  </span>
                                  <span className="sr-only">
                                    {cl.pass ? "Passed: " : "Failed: "}
                                  </span>
                                  {describeClause(cl)}
                                  {cl.actual != null && (
                                    <span className="subtle">
                                      {" "}
                                      (measured{" "}
                                      {metricInfo(
                                        cl.metric.replace(/^abs_/, ""),
                                      ).format(cl.actual)}
                                      )
                                    </span>
                                  )}
                                </li>
                              ))}
                            </ul>
                          )}
                      </div>
                    )}
                    {candidate?.critique ? (
                      <figure className="critic">
                        <blockquote>{candidate.critique}</blockquote>
                        <figcaption>— Claude, judging the video</figcaption>
                      </figure>
                    ) : (
                      <p className="critic-waiting subtle">
                        {candidate && running && candidate.score == null
                          ? "Claude's notes appear here after it watches the video."
                          : candidate
                            ? "Claude left no notes for this candidate."
                            : ""}
                      </p>
                    )}
                    {(candidate?.status === "trained" ||
                      candidate?.status === "scored") && (
                      <FlagControl key={candidate.id} candidate={candidate} />
                    )}
                    {renderReply && shown && (
                      <div className="reply-slot">
                        {renderReply({
                          kind: "candidate",
                          requestId: detail.id,
                          round: shown.round.n,
                          candidateId: shown.c.id,
                          idx: shown.c.idx,
                        })}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </section>
          )}
        </div>

        <div className="technical" aria-labelledby={`explore-${detail.id}`}>
          <h3 className="details-title" id={`explore-${detail.id}`}>
            Explore the run
            <span className="subtle">
              Optional detail · click a row to open it
            </span>
          </h3>
          <details>
            <summary>
              All candidates{" "}
              <span>
                {candidatesTotal} {candidatesTotal === 1 ? "reward" : "rewards"}{" "}
                in {detail.rounds.length}{" "}
                {detail.rounds.length === 1 ? "round" : "rounds"}
                {top ? ` · best: ${candidateName(top.round.n, top.c.idx)}` : ""}
              </span>
            </summary>
            <div>
              <p className="subtle">
                Each round, Claude writes several{" "}
                <Explain term="candidate">candidates</Explain> and trains them
                side by side. Pick a round, then a candidate to show it above.
              </p>
              <RoundTimeline
                rounds={detail.rounds}
                current={explored?.n}
                bestScore={top?.score ?? null}
                bestRound={detail.status === "done" ? top?.round.n : undefined}
                pending={null}
                onSelect={(n) => show(n, null)}
              />
              {explored && (
                <div
                  className="candidate-list"
                  aria-label={`Candidates in round ${explored.n}`}
                >
                  {explored.candidates.length === 0 && (
                    <p className="subtle">
                      Claude is still writing this round's rewards.
                    </p>
                  )}
                  {explored.candidates.map((c) => (
                    <CandidateCard
                      key={c.id}
                      c={c}
                      round={explored.n}
                      best={c.id === explored.best_candidate_id}
                      overall={c.id === top?.c.id}
                      selected={candidate?.id === c.id}
                      onSelect={() => show(explored.n, c.id)}
                      reply={renderReply?.({
                        kind: "candidate",
                        requestId: detail.id,
                        round: explored.n,
                        candidateId: c.id,
                        idx: c.idx,
                      })}
                    />
                  ))}
                </div>
              )}
            </div>
          </details>
          {candidate && (
            <details>
              <summary>
                {name}, in depth{" "}
                <span>learning curve · reward breakdown · measurements</span>
              </summary>
              <div>
                {candidate.error && (
                  <pre className="error" role="alert">
                    {candidate.error}
                  </pre>
                )}
                <RewardCurve
                  key={candidate.id}
                  curve={candidate.progress?.curve}
                  live={Boolean(live)}
                />
                <RewardTerms metrics={candidate.metrics} />
                <h3 className="sub-title">Measured in the test run</h3>
                {metrics.length ? (
                  <dl className="metrics">
                    {metrics
                      .filter(([key]) => isListedMetric(key))
                      .map(([key, value]) => {
                        const info = metricInfo(key);
                        return (
                          <div className="metric" key={key}>
                            <dt>
                              {info.label}
                              {info.help && (
                                <span className="subtle"> ({info.help})</span>
                              )}
                            </dt>
                            <dd className="num">{info.format(value)}</dd>
                          </div>
                        );
                      })}
                    {robustness &&
                      Object.entries(robustness).map(([group, r]) => (
                        <div className="metric" key={group}>
                          <dt>
                            Robustness: {group}
                            <span className="subtle"> (passed the check)</span>
                          </dt>
                          <dd className="num">
                            {r.passed} / {r.total}
                          </dd>
                        </div>
                      ))}
                  </dl>
                ) : (
                  <p className="subtle">
                    {running
                      ? "Measurements appear once it finishes training."
                      : "No measurements were recorded."}
                  </p>
                )}
              </div>
            </details>
          )}
          {candidate && (
            <details>
              <summary>
                Reward function <span>the Python Claude wrote</span>
              </summary>
              <pre className="code">{candidate.reward_src}</pre>
            </details>
          )}
          {candidate?.diff && (
            <details>
              <summary>
                What changed{" "}
                <span>Claude's edit to the reward it started from</span>
              </summary>
              <DiffView diff={candidate.diff} />
            </details>
          )}
          <RoundGallery
            rounds={detail.rounds}
            onPick={(n, id) => show(n, id)}
          />
          <details>
            <summary>
              Activity log{" "}
              <span>
                {detail.events.length}{" "}
                {detail.events.length === 1 ? "event" : "events"}, including{" "}
                code-check errors
              </span>
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
                    <span className="log-kind">
                      {EVENT_KINDS[event.kind] ?? event.kind}
                    </span>
                    <span>{event.message}</span>
                  </li>
                ))}
              </ol>
            ) : (
              <p>No events yet.</p>
            )}
          </details>
        </div>
      </div>
      {renderReply && (
        <div className="reply-slot">
          {renderReply({ kind: "run", requestId: detail.id })}
        </div>
      )}
    </article>
  );
}

// A finished run with nothing to show: what happened, and what to do next.
function RunProblem({
  detail,
  onRetry,
}: {
  detail: Detail;
  onRetry?: (text: string) => void;
}) {
  const all = detail.rounds.flatMap((r) => r.candidates);
  const counts = new Map<string, number>();
  for (const c of all) {
    const text = candidateState(c).text;
    counts.set(text, (counts.get(text) ?? 0) + 1);
  }
  const firstError = all.find((c) => c.error)?.error?.split("\n")[0];
  const failedRun = detail.status === "failed";
  const reason = [...detail.events]
    .reverse()
    .find((e) => e.kind === "error")?.message;
  return (
    <div className="run-problem">
      <p className="problem-title">
        {failedRun
          ? "The run stopped before it produced a robot."
          : `None of the ${all.length} rewards produced a robot Claude could judge.`}
      </p>
      {(failedRun ? reason : null) && <p>Reason: {reason}</p>}
      {all.length > 0 && (
        <p>
          What happened:{" "}
          {[...counts]
            .map(([text, n]) => `${n} × ${text.toLowerCase()}`)
            .join(", ")}
          .
          {firstError && (
            <>
              {" "}
              First error: <code>{firstError}</code>
            </>
          )}
        </p>
      )}
      <p>
        What to try: run it again (Claude writes different rewards each time),
        or describe the behavior more concretely. The candidates and the full
        errors are under “Explore the run”.
      </p>
      {onRetry && (
        <button className="primary" onClick={() => onRetry(detail.text)}>
          Edit and run again
        </button>
      )}
    </div>
  );
}
