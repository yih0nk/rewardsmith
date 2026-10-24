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
  const mode = !health
    ? "API unavailable"
    : health.fake_sim
      ? `Fake simulator · ${health.fake_llm ? "Fake LLM" : "Live LLM"}`
      : `Real simulator · ${health.fake_llm ? "Fake LLM" : "Live LLM"}`;
  const goHome = () => setSelected(runs[0]?.id ?? "");
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
            onClick={goHome}
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
                <p className="eyebrow">The reinforcement learning playground</p>
                <h1 id="hero-title">
                  A little prompt. A new <span className="em">behavior.</span>
                </h1>
                <p className="hero-sub">
                  Describe a behavior. Watch your robot figure it out.
                </p>
                <form className="composer" onSubmit={submit}>
                  <label htmlFor="behavior">What should the robot learn?</label>
                  <div className="composer-row">
                    <span className="prompt" aria-hidden="true">
                      &gt;
                    </span>
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
                      {busy ? "Starting…" : "Start experiment"}
                    </button>
                  </div>
                  <div className="suggestions">
                    <span className="label">Try</span>
                    {suggestions.map((suggestion) => (
                      <button
                        type="button"
                        className="link"
                        key={suggestion}
                        onClick={() =>
                          setText(
                            `Make the robot dog ${suggestion.toLowerCase()}`,
                          )
                        }
                      >
                        {suggestion}
                      </button>
                    ))}
                  </div>
                </form>
              </div>
              <dl className="facts">
                <div>
                  <dt>Mode</dt>
                  <dd>{mode}</dd>
                </div>
                <div>
                  <dt>Experiments</dt>
                  <dd className="num">{runs.length}</dd>
                </div>
                <div className="facts-qr">
                  <dt>On your phone</dt>
                  <dd>
                    <QrPanel />
                  </dd>
                </div>
              </dl>
            </section>
            <section className="cmd" aria-labelledby="runs-title">
              <h2 className="cmd-title" id="runs-title">
                Experiments
              </h2>
              <RequestList
                requests={runs}
                selected={selected}
                onSelect={setSelected}
              />
            </section>
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
                <button className="link" onClick={goHome}>
                  Back to experiments
                </button>
              </div>
            )}
            {detail ? (
              <RequestDetail key={detail.id} detail={detail} health={health} />
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
                        Choose an experiment from the list above, or describe a
                        new behavior.
                      </p>
                    </>
                  ) : (
                    <>
                      <h2 className="cmd-title">
                        Your first experiment starts here.
                      </h2>
                      <p>
                        Describe a behavior above to queue a request. Run the
                        worker to train and evaluate candidates.
                      </p>
                    </>
                  )}
                </section>
              )
            )}
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
        </footer>
      </main>
    </div>
  );
}
