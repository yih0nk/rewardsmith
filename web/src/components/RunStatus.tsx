import { useRef } from "react";
import { RequestDetail as Detail, serverClockOffset } from "../api";
import { Explain } from "../explain";
import {
  STEPS,
  candidateName,
  candidateState,
  elapsedSince,
  formatElapsed,
  runStage,
  stepIndex,
} from "../stages";
import { QueuePlace } from "../thread";
import { useNow } from "../useNow";
import { liveState, textBar } from "./TrainingProgress";

// What the run is doing now, how long it has taken, and which steps are left.
export default function RunStatus({
  detail,
  candidatesPerRound,
  queue,
  now: nowOverride,
}: {
  detail: Detail;
  candidatesPerRound?: number | null;
  queue?: QueuePlace | null;
  // Tests pass a fixed clock; the board ticks.
  now?: number;
}) {
  const stage = runStage(detail, candidatesPerRound, queue);
  const active = detail.status === "running" || detail.status === "queued";
  const ticking = useNow(active && nowOverride == null);
  const now = nowOverride ?? ticking;
  // When the API has no start time for a stage, time it from when this page first saw it.
  const seen = useRef(new Map<string, number>());
  const stageId = `${stage.key}:${stage.round ?? ""}`;
  if (!seen.current.has(stageId)) seen.current.set(stageId, now);
  const offset = serverClockOffset();
  const stageSeconds =
    stage.startedAt != null
      ? elapsedSince(stage.startedAt, now, offset)
      : (now - seen.current.get(stageId)!) / 1000;
  const end = detail.events.at(-1)?.created_at ?? detail.created_at;
  // The run's time counts from when the worker picked it up; time in line is shown apart,
  // so a task that waited behind others doesn't look slow.
  const starts = [
    detail.events.find((e) => e.kind !== "queued" && e.kind !== "control")?.created_at,
    detail.stages?.[0]?.started_at,
    detail.rounds[0]?.created_at,
  ].filter((t): t is number => t != null);
  const pickedUp = starts.length ? Math.min(...starts) : null;
  // Under half a minute in line is just the worker's poll: count it as part of the run.
  const waited = pickedUp != null && pickedUp - detail.created_at >= 30 ? pickedUp - detail.created_at : 0;
  const from = detail.created_at + waited;
  const totalSeconds = active ? elapsedSince(from, now, offset) : end - from;
  const step = stepIndex(stage.key);
  const finished = stage.key === "done";
  const broken = stage.key === "failed" || stage.key === "no_results";
  const last = detail.rounds.at(-1);
  const lanes =
    active &&
    last &&
    ["compiling", "training", "rendering", "judging"].includes(stage.key)
      ? last.candidates
      : [];

  return (
    <div
      className={`run-status-panel ${broken ? "broken" : ""} ${finished ? "finished" : ""}`}
    >
      <div className="now">
        <span className="now-tag">
          {active
            ? "NOW"
            : finished
              ? "DONE"
              : stage.key === "no_results"
                ? "NO RESULT"
                : stage.key === "paused"
                  ? "PAUSED"
                  : "STOPPED"}
        </span>
        <p className="now-label" role="status">
          {stage.label}
        </p>
        <dl className="timers">
          {active && (
            <div>
              <dt>This step</dt>
              <dd className="num" data-testid="stage-timer">
                {formatElapsed(stageSeconds)}
              </dd>
            </div>
          )}
          <div>
            <dt>{active ? "Whole run" : "Took"}</dt>
            <dd className="num" data-testid="run-timer">
              {formatElapsed(totalSeconds)}
            </dd>
          </div>
          {waited > 0 && (
            <div>
              <dt>In line first</dt>
              <dd className="num" data-testid="wait-timer">
                {formatElapsed(waited)}
              </dd>
            </div>
          )}
        </dl>
      </div>
      {stage.hint && <p className="now-hint">{stage.hint}</p>}
      {active && (
        <>
          <ol className="steps" aria-label="Pipeline steps">
            {stage.round != null && (
              <li className="steps-round" aria-hidden="true">
                Round {stage.round}
              </li>
            )}
            {STEPS.map((s, i) => {
              const state = i < step ? "done" : i === step ? "current" : "todo";
              return (
                <li
                  key={s.key}
                  className={`step ${state}`}
                  aria-current={state === "current" ? "step" : undefined}
                >
                  <span aria-hidden="true">
                    {state === "done" ? "✓" : state === "current" ? "▸" : "·"}
                  </span>{" "}
                  {s.key === "gate" ? (
                    <Explain term="gate">{s.short}</Explain>
                  ) : (
                    s.short
                  )}
                  <span className="sr-only">
                    {state === "done"
                      ? " (done)"
                      : state === "current"
                        ? " (in progress)"
                        : ""}
                  </span>
                </li>
              );
            })}
          </ol>
          <p className="steps-note">
            {stage.round != null && (
              <span className="sr-only">Round {stage.round}. </span>
            )}
            {detail.mode === "instant" ? (
              <>One round: Instant writes one reward and stops. Follow up to refine it.</>
            ) : (
              <>
                ↻ Steps 2–7 repeat each round:{" "}
                <Explain term="carry">Claude revises the best rewards</Explain>
              </>
            )}
          </p>
        </>
      )}
      {lanes.length > 0 && (
        <ul className="lanes" aria-label={`Candidates in round ${last!.n}`}>
          {lanes.map((c) => {
            const live = liveState(c);
            const state = candidateState(c, live?.label);
            return (
              <li key={c.id} className={`lane ${state.kind}`}>
                <span className="lane-name">
                  {candidateName(last!.n, c.idx)}
                </span>
                <span className="lane-bar" aria-hidden="true">
                  {live?.fraction != null ? textBar(live.fraction, 20) : ""}
                </span>
                <span className="lane-state">{state.text}</span>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
