import { useEffect, useRef, useState } from "react";
import {
  Health,
  RequestSummary,
  createRequest,
  getHealth,
  listRequests,
} from "./api";
import Composer from "./components/Composer";
import ConversationList from "./components/ConversationList";
import ConversationView from "./components/ConversationView";
import HallOfFame from "./components/HallOfFame";
import NewTask from "./components/NewTask";
import { conversationsFromRequests } from "./thread";

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
      const result = await createRequest(prompt);
      setText("");
      // Show it in the sidebar at once rather than on the next poll.
      setRuns((current) => [
        {
          id: result.id,
          text: prompt,
          status: "queued",
          best_score: null,
          created_at: Date.now() / 1000,
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
