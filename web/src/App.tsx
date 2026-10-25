import { useEffect, useRef, useState } from "react";
import {
  Health,
  MessageOptions as Options,
  RequestSummary,
  createFollowUp,
  createRequest,
  getHealth,
  listRequests,
} from "./api";
import Composer from "./components/Composer";
import ConversationList from "./components/ConversationList";
import ConversationView from "./components/ConversationView";
import HallOfFame from "./components/HallOfFame";
import MessageOptions from "./components/MessageOptions";
import NewTask from "./components/NewTask";
import Showcase, { showcasePicks } from "./components/Showcase";
import { FOLLOWUP_OPTIONS, TASK_OPTIONS, bodiesOf, modelsOf } from "./modes";
import { candidateName } from "./stages";
import { ReplyTarget, conversationsFromRequests, runAnchor } from "./thread";

// A message's options, and whether the user picked the model (then a mode switch keeps it).
type Picked = Options & { picked?: boolean };
// The candidate a follow-up will refine, as the composer names it.
interface Target {
  candidateId: number;
  label: string;
}

// Routes live in the hash: "#new", "#hall", or "#<task id>".
const readRoute = () => location.hash.slice(1);

export default function App() {
  const [runs, setRuns] = useState<RequestSummary[]>([]);
  const [route, setRoute] = useState(readRoute);
  const [health, setHealth] = useState<Health | null>(null);
  // False until the first health check answers or fails: "connecting", not "unreachable".
  const [healthChecked, setHealthChecked] = useState(false);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [listLoading, setListLoading] = useState(true);
  const [listError, setListError] = useState("");
  const [submitError, setSubmitError] = useState("");
  // Phones: the task list opens over the page from the header.
  const [navOpen, setNavOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const [taskOptions, setTaskOptions] = useState<Picked>(TASK_OPTIONS);
  // The follow-up composer at the bottom of the open task.
  const [followText, setFollowText] = useState("");
  const [followOptions, setFollowOptions] = useState<Picked>(FOLLOWUP_OPTIONS);
  const [target, setTarget] = useState<Target | null>(null);
  const [followBusy, setFollowBusy] = useState(false);
  const [followError, setFollowError] = useState("");
  const followRef = useRef<HTMLInputElement>(null);
  // Open the newest task on first load only; afterwards the route is the user's.
  const firstLoad = useRef(true);

  const go = (next: string) => {
    setRoute(next);
    setNavOpen(false);
    window.scrollTo({ top: 0 });
  };

  useEffect(() => {
    let active = true;
    async function refresh() {
      try {
        const data = await listRequests();
        if (!active) return;
        setRuns(data);
        setListError("");
        if (firstLoad.current) {
          firstLoad.current = false;
          if (!readRoute()) setRoute(data[0]?.id ?? "new");
        }
      } catch (error) {
        if (active) setListError(`Cannot load tasks: ${String(error)}`);
      } finally {
        if (active) setListLoading(false);
      }
    }
    refresh();
    const timer = setInterval(refresh, 2000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    let active = true;
    const refresh = () =>
      getHealth()
        .then((data) => {
          if (active) {
            setHealth(data);
            setHealthChecked(true);
          }
        })
        .catch(() => {
          if (active) {
            setHealth(null);
            setHealthChecked(true);
          }
        });
    refresh();
    const timer = setInterval(refresh, 10000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (route && readRoute() !== route) location.hash = route;
  }, [route]);
  useEffect(() => {
    const change = () => setRoute(readRoute());
    window.addEventListener("hashchange", change);
    return () => window.removeEventListener("hashchange", change);
  }, []);

  async function submit(prompt: string) {
    setBusy(true);
    setSubmitError("");
    try {
      const { picked: _, ...options } = taskOptions;
      const result = await createRequest(prompt, options);
      setText("");
      // Show it in the sidebar at once rather than on the next poll.
      setRuns((current) => [
        {
          id: result.id,
          text: prompt,
          status: "queued",
          best_score: null,
          created_at: Date.now() / 1000,
          conversation_id: result.id,
          turn: 1,
          ...options,
        },
        ...current.filter((r) => r.id !== result.id),
      ]);
      go(result.id);
    } catch (error) {
      setSubmitError(`Could not start the task: ${String(error)}`);
    } finally {
      setBusy(false);
    }
  }
  // A finished task's prompt, back in a new task's box to edit and run again.
  const retry = (prompt: string) => {
    setText(prompt);
    go("new");
    setTimeout(() => inputRef.current?.focus(), 0);
  };

  const conversations = conversationsFromRequests(runs);
  const hall = route === "hall";
  const current = conversations.find((c) => c.id === route) ?? null;

  // A new task starts with an empty follow-up box aimed at its best candidate. The
  // follow-up is Instant, on the model the task's latest turn used, so the box never
  // names a different model than the run above it.
  const lastModel = current
    ? runs
        .filter((r) => (r.conversation_id ?? r.id) === current.id)
        .sort((a, b) => (b.turn ?? 1) - (a.turn ?? 1))[0]?.model
    : null;
  useEffect(() => {
    setFollowText("");
    setTarget(null);
    setFollowError("");
    setFollowOptions(
      lastModel ? { ...FOLLOWUP_OPTIONS, model: lastModel, picked: true } : FOLLOWUP_OPTIONS,
    );
    // Only when the task changes: a model picked for this follow-up stays picked.
  }, [current?.id]);

  async function followUp(note: string) {
    if (!current) return;
    setFollowBusy(true);
    setFollowError("");
    try {
      const { picked: _, ...options } = followOptions;
      const result = await createFollowUp(current.id, note, options, target?.candidateId);
      const turn = Math.max(...current.turns.map((t) => t.turn)) + 1;
      setRuns((rs) => [
        {
          id: result.id,
          text: note || "Keep going",
          status: "queued",
          best_score: null,
          created_at: Date.now() / 1000,
          conversation_id: current.id,
          turn,
          target_candidate_id: target?.candidateId ?? null,
          ...options,
        },
        ...rs.filter((r) => r.id !== result.id),
      ]);
      setFollowText("");
      setTarget(null);
      setTimeout(
        () => document.getElementById(runAnchor(result.id))?.scrollIntoView?.({ block: "start" }),
        400,
      );
    } catch (error) {
      setFollowError(`Could not send the follow-up: ${String(error)}`);
    } finally {
      setFollowBusy(false);
    }
  }

  const turnNumber = (requestId: string) => runs.find((r) => r.id === requestId)?.turn ?? 1;
  const name = (requestId: string, round: number, idx: number) => {
    const base = candidateName(round, idx);
    return (current?.turns.length ?? 1) > 1 ? `Turn ${turnNumber(requestId)} · ${base}` : base;
  };
  const aim = (next: Target) => {
    setTarget(next);
    setTimeout(() => followRef.current?.focus(), 0);
  };
  // "Refine this" on each scored candidate, and on the run's best after the run.
  const renderReply = (reply: ReplyTarget) => {
    if (reply.kind === "candidate") {
      if (reply.scored === false) return null;
      const on = target?.candidateId === reply.candidateId;
      return (
        <button
          type="button"
          className={`refine ${on ? "on" : ""}`}
          aria-pressed={on}
          onClick={() =>
            aim({ candidateId: reply.candidateId, label: name(reply.requestId, reply.round, reply.idx) })
          }
        >
          {on ? "✓ Refining this" : "Refine this"}
        </button>
      );
    }
    const run = runs.find((r) => r.id === reply.requestId);
    if (!run?.best_candidate_id || run.status === "running" || run.status === "queued") return null;
    return (
      <button
        type="button"
        className="refine"
        onClick={() =>
          aim({ candidateId: run.best_candidate_id!, label: `the best of turn ${run.turn ?? 1}` })
        }
      >
        Follow up on this run's best
      </button>
    );
  };
  const followComposer = current && (
    <Composer
      label="Follow up"
      inputId="follow-up"
      inputRef={followRef}
      busy={followBusy}
      error={followError}
      text={followText}
      onText={(value) => {
        setFollowText(value);
        setFollowError("");
      }}
      onSubmit={followUp}
      examples={false}
      allowEmpty
      compact={!target}
      placeholder="Follow up, e.g. “go faster” (empty = keep going)"
      submitLabel={followText.trim() ? "Send" : "Keep going"}
      context={
        <p className="refine-target">
          <span className="label">Refines</span>{" "}
          {target ? (
            <>
              <strong>{target.label}</strong>{" "}
              <button type="button" className="link" onClick={() => setTarget(null)}>
                Use the best instead
              </button>
            </>
          ) : current.bestScore != null ? (
            <>the best candidate so far, or pick one with Refine this</>
          ) : (
            <>the best candidate once one is scored; before that, your note is added to the task</>
          )}
        </p>
      }
      controls={
        <MessageOptions
          value={followOptions}
          onChange={setFollowOptions}
          models={modelsOf(health)}
          idPrefix="follow-up"
        />
      }
    />
  );
  const running = runs.find((r) => r.status === "running")?.text ?? null;
  const mode = !health
    ? healthChecked
      ? "Server unreachable · retrying"
      : "Connecting…"
    : health.fake_sim
      ? `Fake simulator · ${health.fake_llm ? "Fake LLM" : "Live LLM"}`
      : `Real simulator · ${health.fake_llm ? "Fake LLM" : "Live LLM"}`;

  return (
    <div className="terminal app">
      <header className="session-bar">
        <a className="wordmark" href="#new" onClick={() => go("new")}>
          rewardsmith
          <span className="cursor" aria-hidden="true" />
        </a>
        <button
          className="nav-link nav-toggle"
          aria-expanded={navOpen}
          aria-controls="sidebar"
          onClick={() => setNavOpen((open) => !open)}
        >
          {navOpen ? "Close tasks" : "Tasks"}{" "}
          <span className="num">({conversations.length})</span>
        </button>
        <span className={`mode ${health || !healthChecked ? "" : "down"}`}>
          {mode}
        </span>
      </header>
      <div className="workspace">
        <aside
          id="sidebar"
          className={`sidebar ${navOpen ? "open" : ""}`}
          aria-label="Tasks and pages"
        >
          <button
            className={`primary new-task ${route === "new" ? "current" : ""}`}
            aria-current={route === "new" ? "page" : undefined}
            onClick={() => {
              go("new");
              setTimeout(() => inputRef.current?.focus(), 0);
            }}
          >
            + New task
          </button>
          <ConversationList
            conversations={conversations}
            selected={hall || route === "new" ? null : route}
            loading={listLoading}
            onSelect={go}
            health={health}
          />
          {listError && (
            <p className="error" role="alert">
              {listError} Retrying automatically.
            </p>
          )}
          <a
            className={`nav-link hall-link ${hall ? "active" : ""}`}
            aria-current={hall ? "page" : undefined}
            href="#hall"
            onClick={() => setNavOpen(false)}
          >
            Hall of fame
          </a>
        </aside>
        <main className="session main-pane">
          {hall ? (
            <HallOfFame onOpen={go} onBack={() => go(runs[0]?.id ?? "new")} />
          ) : current ? (
            <ConversationView
              key={current.id}
              conversation={current}
              requests={runs}
              health={health}
              onRetry={retry}
              renderReply={renderReply}
              composer={followComposer}
            />
          ) : listLoading && route !== "new" ? (
            <p className="cmd-note" role="status">
              Loading…
            </p>
          ) : (
            <>
              {route && route !== "new" && (
                <p className="error" role="alert">
                  There is no task #{route} on this server. Start a new one
                  below, or pick one from the list.
                </p>
              )}
              <NewTask
                busyWith={running}
                showcase={
                  <Showcase picks={showcasePicks(conversations, runs)} onOpen={go} />
                }
                composer={
                  <Composer
                    label="What should the robot learn?"
                    inputRef={inputRef}
                    busy={busy}
                    error={submitError}
                    text={text}
                    onText={(value) => {
                      setText(value);
                      setSubmitError("");
                    }}
                    onSubmit={submit}
                    controls={
                      <MessageOptions
                        value={taskOptions}
                        onChange={setTaskOptions}
                        models={modelsOf(health)}
                        bodies={bodiesOf(health)}
                        idPrefix="new-task"
                      />
                    }
                  />
                }
              />
            </>
          )}
        </main>
      </div>
    </div>
  );
}
