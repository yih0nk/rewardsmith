import { FormEvent, useEffect, useState } from "react";
import {
  RequestDetail as Detail,
  RequestSummary,
  Health,
  ApiError,
  createRequest,
  getRequest,
  listRequests,
  getHealth,
  MOCK_MODE,
} from "./api";
import RequestDetail from "./components/RequestDetail";
import RequestList from "./components/RequestList";

const suggestions = [
  "Walk backwards",
  "Stand on two legs",
  "Do a little dance",
];
export default function App() {
  const [runs, setRuns] = useState<RequestSummary[]>([]);
  const [selected, setSelected] = useState(location.hash.slice(1));
  const [detail, setDetail] = useState<Detail | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [listLoading, setListLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [listError, setListError] = useState("");
  const [detailError, setDetailError] = useState("");
  const [submitError, setSubmitError] = useState("");
  const hall = selected === "hall";

  useEffect(() => {
    let active = true;
    async function refresh() {
      try {
        const data = await listRequests();
        if (active) {
          setRuns(data);
          setListError("");
          if (!selected && data[0]) setSelected(data[0].id);
        }
      } catch (error) {
        if (active) setListError(`Cannot load experiments: ${String(error)}`);
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
  }, [selected]);

  useEffect(() => {
    let active = true;
    const refresh = () =>
      getHealth()
        .then((data) => {
          if (active) setHealth(data);
        })
        .catch(() => {
          if (active) setHealth(null);
        });
    refresh();
    const timer = setInterval(refresh, 10000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    setDetail(null);
    setDetailError("");
    setDetailLoading(Boolean(selected && !hall));
    if (!selected || hall) return;
    let active = true;
    async function refresh() {
      try {
        const data = await getRequest(selected);
        if (active) {
          setDetail(data);
          setDetailError("");
        }
      } catch (error) {
        if (active) {
          if (error instanceof ApiError && error.status === 404)
            setDetail(null);
          setDetailError(`Cannot load this experiment: ${String(error)}`);
        }
      } finally {
        if (active) setDetailLoading(false);
      }
    }
    refresh();
    const timer = setInterval(refresh, 1500);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [selected, hall]);

  useEffect(() => {
    location.hash = selected;
  }, [selected]);
  useEffect(() => {
    const change = () => setSelected(location.hash.slice(1));
    window.addEventListener("hashchange", change);
    return () => window.removeEventListener("hashchange", change);
  }, []);
  async function submit(event: FormEvent) {
    event.preventDefault();
    if (text.trim().length < 3 || busy) return;
    setBusy(true);
    setSubmitError("");
    try {
      const result = await createRequest(text.trim());
      setSelected(result.id);
      setText("");
    } catch (error) {
      setSubmitError(`Could not start experiment: ${String(error)}`);
    } finally {
      setBusy(false);
    }
  }
  const mode = MOCK_MODE
    ? "Demo · Mock API"
    : !health
      ? "API unavailable"
      : health.fake_sim
        ? `Fake simulator · ${health.fake_llm ? "Fake LLM" : "Live LLM"}`
        : `Real simulator · ${health.fake_llm ? "Fake LLM" : "Live LLM"}`;
  return (
    <div className="shell">
      <aside className="sidebar">
        <a className="brand" href="#" onClick={() => setSelected("")}>
          <span className="brand-icon">⌘</span> rewardsmith
          <span className="brand-dot">.</span>
        </a>
        <div className="workspace-label">WORKSPACE</div>
        <button
          className="nav-item"
          onClick={() => setSelected(runs[0]?.id ?? "")}
        >
          <span>▦</span> Experiments{" "}
          <span className="count">{runs.length}</span>
        </button>
        <a className="hall-link" href="#hall">
          ↗ Hall of fame
        </a>
        <div className="sidebar-heading">RECENT RUNS</div>
        <RequestList
          requests={runs}
          selected={selected}
          onSelect={setSelected}
        />
        <div className="sidebar-footer">
          <span className="tiny-logo">R</span>
          <div>
            Cal Hacks 13.0<small>Built to learn. Built to iterate.</small>
          </div>
        </div>
      </aside>
      <main>
        <header className="topbar">
          <span>
            Workspace <span className="slash">/</span>{" "}
            <strong>{hall ? "Hall of fame" : "Experiments"}</strong>
          </span>
          <span className="mode">
            <span className="status-dot" />
            {mode}
          </span>
        </header>
        <div className="content">
          <select
            className="mobile-runs"
            aria-label="Select experiment"
            value={selected}
            onChange={(event) => setSelected(event.target.value)}
          >
            <option value="">Experiments</option>
            <option value="hall">Hall of fame</option>
            {runs.map((run) => (
              <option key={run.id} value={run.id}>
                {run.text} · {run.status}
              </option>
            ))}
          </select>
          {hall ? (
            <section className="empty-state">
              <div className="eyebrow">HALL OF FAME</div>
              <h1>The best is yet to come.</h1>
              <p>
                This route is reserved for the upcoming hall of fame. The
                backend does not support flagged candidates yet.
              </p>
              <button
                className="primary"
                onClick={() => setSelected(runs[0]?.id ?? "")}
              >
                Back to experiments
              </button>
            </section>
          ) : (
            <>
              <div className="page-heading">
                <div>
                  <div className="eyebrow">
                    THE REINFORCEMENT LEARNING PLAYGROUND
                  </div>
                  <h1>A little prompt. A new behavior.</h1>
                  <p>Describe a behavior. Watch your robot figure it out.</p>
                </div>
                <span className="version">DRAFT 01</span>
              </div>
              <form className="prompt-card" onSubmit={submit}>
                <label htmlFor="behavior">What should the robot learn?</label>
                <div className="prompt-row">
                  <span className="spark">✧</span>
                  <input
                    id="behavior"
                    value={text}
                    onChange={(event) => setText(event.target.value)}
                    maxLength={300}
                    placeholder="Make the robot dog walk backwards…"
                    required
                    minLength={3}
                  />
                  <button
                    className="primary"
                    disabled={busy || text.trim().length < 3}
                  >
                    {busy ? "Starting…" : "Start experiment"} <span>↗</span>
                  </button>
                </div>
                <div className="suggestions">
                  <span>Try a behavior</span>
                  {suggestions.map((suggestion) => (
                    <button
                      type="button"
                      key={suggestion}
                      onClick={() =>
                        setText(
                          `Make the robot dog ${suggestion.toLowerCase()}`,
                        )
                      }
                    >
                      {suggestion} ↗
                    </button>
                  ))}
                </div>
              </form>
              {submitError && (
                <div className="error" role="alert">
                  {submitError}
                </div>
              )}
              {listError && (
                <div className="error" role="alert">
                  {listError} Retrying automatically.
                </div>
              )}
              {detailError && (
                <div className="error" role="alert">
                  {detailError}{" "}
                  <button onClick={() => setSelected(runs[0]?.id ?? "")}>
                    Back to experiments
                  </button>
                </div>
              )}
              {detail ? (
                <RequestDetail
                  key={detail.id}
                  detail={detail}
                  health={health}
                />
              ) : detailLoading || listLoading ? (
                <div className="loading" role="status">
                  Loading experiment…
                </div>
              ) : (
                !detailError && (
                  <section className="empty-state">
                    <h2>Your first experiment starts here.</h2>
                    <p>
                      Describe a behavior above to queue a request. Run the
                      worker to train and evaluate candidates.
                    </p>
                  </section>
                )
              )}
            </>
          )}
          <footer>
            REWARDSMITH <span>Language → reward → motion.</span>
            <span>Made at Cal Hacks 13.0</span>
          </footer>
        </div>
      </main>
    </div>
  );
}
