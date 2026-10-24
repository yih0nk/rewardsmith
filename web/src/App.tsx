import { FormEvent, useEffect, useRef, useState } from "react";
import {
  RequestDetail as Detail,
  RequestSummary,
  Health,
  ApiError,
  createRequest,
  getRequest,
  listRequests,
  getHealth,
} from "./api";
import HallOfFame from "./components/HallOfFame";
import QrPanel from "./components/QrPanel";
import RequestDetail from "./components/RequestDetail";
import RequestList from "./components/RequestList";

const suggestions = ["Walk backwards", "Spin in place", "Walk to the red ball"];
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
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLElement>(null);
  const hall = selected === "hall";
  // Open the newest run on first load only; afterwards the logo goes home and stays there.
  const firstLoad = useRef(true);

  useEffect(() => {
    let active = true;
    async function refresh() {
      try {
        const data = await listRequests();
        if (active) {
          setRuns(data);
          setListError("");
          if (firstLoad.current) {
            firstLoad.current = false;
            if (!location.hash.slice(1) && data[0]) setSelected(data[0].id);
          }
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
  }, []);

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
    if (busy) return;
    if (text.trim().length < 3) {
      setSubmitError(
        "Type a behavior first (at least 3 characters), or click an example.",
      );
      inputRef.current?.focus();
      return;
    }
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
  const mode = !health
    ? "Server unreachable · retrying"
    : health.fake_sim
      ? `Fake simulator · ${health.fake_llm ? "Fake LLM" : "Live LLM"}`
      : `Real simulator · ${health.fake_llm ? "Fake LLM" : "Live LLM"}`;
  const goHome = () => setSelected(runs[0]?.id ?? "");
  const showList = () => {
    if (hall) goHome();
    // After the hall of fame unmounts, the list is back on the page.
    setTimeout(() => listRef.current?.scrollIntoView({ block: "start" }), 0);
  };
  // A finished run's prompt, back in the composer to edit and run again.
  const retry = (prompt: string) => {
    setText(prompt);
    window.scrollTo({ top: 0 });
    inputRef.current?.focus();
  };
  return (
    <div className="terminal">
      <header className="session-bar">
        <a className="wordmark" href="#" onClick={() => setSelected("")}>
          rewardsmith
          <span className="cursor" aria-hidden="true" />
        </a>
        <nav aria-label="Board">
          <button
            className={`nav-link ${hall ? "" : "active"}`}
            aria-current={hall ? undefined : "page"}
            onClick={showList}
          >
            Experiments <span className="num">({runs.length})</span>
          </button>
          <a
            className={`nav-link ${hall ? "active" : ""}`}
            aria-current={hall ? "page" : undefined}
            href="#hall"
          >
            Hall of fame
          </a>
        </nav>
        <span className={`mode ${health ? "" : "down"}`}>{mode}</span>
      </header>
      <main className="session">
        {hall ? (
          <HallOfFame onOpen={setSelected} onBack={goHome} />
        ) : (
          <>
            <section className="cmd hero" aria-labelledby="hero-title">
              <div className="hero-main">
                <h1 id="hero-title">
                  <label htmlFor="behavior">
                    Teach a robot dog a new behavior.{" "}
                    <span className="em">What should it learn?</span>
                  </label>
                </h1>
                <form className="composer" onSubmit={submit}>
                  <div className="composer-row">
                    <span className="prompt" aria-hidden="true">
                      &gt;
                    </span>
                    <input
                      id="behavior"
                      ref={inputRef}
                      value={text}
                      onChange={(event) => {
                        setText(event.target.value);
                        setSubmitError("");
                      }}
                      maxLength={300}
                      placeholder="Make the robot dog walk backwards…"
                      aria-describedby="behavior-help"
                    />
                    <button className="primary" disabled={busy}>
                      {busy ? "Starting…" : "Start training"}
                    </button>
                  </div>
                  <p
                    className="composer-help error-text"
                    id="behavior-help"
                    role="alert"
                    hidden={!submitError}
                  >
                    {submitError}
                  </p>
                  <div className="suggestions">
                    <span className="label">Examples, click to use</span>
                    {suggestions.map((suggestion) => (
                      <button
                        type="button"
                        className="link"
                        key={suggestion}
                        onClick={() => {
                          setText(
                            `Make the robot dog ${suggestion.toLowerCase()}`,
                          );
                          inputRef.current?.focus();
                        }}
                      >
                        {suggestion}
                      </button>
                    ))}
                  </div>
                </form>
                <ol className="how" aria-label="How it works">
                  <li>You describe a behavior</li>
                  <li>
                    Claude writes reward functions (code that scores motion)
                  </li>
                  <li>Robots train in a physics simulator</li>
                  <li>Claude judges the videos and improves the best reward</li>
                  <li className="how-time">about 5–15 min per run</li>
                </ol>
              </div>
            </section>
            {listError && (
              <div className="error" role="alert">
                {listError} Retrying automatically.
              </div>
            )}
            {detailError && (
              <div className="error" role="alert">
                {detailError}{" "}
                <button className="link" onClick={goHome}>
                  Back to experiments
                </button>
              </div>
            )}
            {detail ? (
              <RequestDetail
                key={detail.id}
                detail={detail}
                health={health}
                onRetry={retry}
              />
            ) : detailLoading || listLoading ? (
              <p className="cmd-note" role="status">
                Loading experiment…
              </p>
            ) : (
              !detailError && (
                <section className="cmd empty-state">
                  {runs.length ? (
                    <>
                      <h2 className="cmd-title">Pick up where you left off.</h2>
                      <p>
                        Choose an experiment from the list below, or describe a
                        new behavior.
                      </p>
                    </>
                  ) : (
                    <>
                      <h2 className="cmd-title">
                        Your first experiment starts here.
                      </h2>
                      <p>
                        Type a behavior in the box above, or click an example,
                        then press Start training.
                      </p>
                    </>
                  )}
                </section>
              )
            )}
            <section className="cmd" aria-labelledby="runs-title" ref={listRef}>
              <h2 className="cmd-title" id="runs-title">
                All experiments
                <span className="subtle">
                  newest first · select one to open it
                </span>
              </h2>
              <RequestList
                requests={runs}
                loading={listLoading}
                selected={selected}
                onSelect={(id) => {
                  setSelected(id);
                  window.scrollTo({ top: 0 });
                }}
              />
            </section>
          </>
        )}
        <footer className="directory">
          <div>
            <h2>Rewardsmith</h2>
            <p>Language → reward → motion.</p>
          </div>
          <div>
            <h2>Board</h2>
            <button className="link" onClick={goHome}>
              Experiments
            </button>
            <a className="link" href="#hall">
              Hall of fame
            </a>
          </div>
          <div>
            <h2>Made at</h2>
            <p>Cal Hacks 13.0</p>
          </div>
          <div>
            <h2>On your phone</h2>
            <QrPanel />
          </div>
        </footer>
      </main>
    </div>
  );
}
