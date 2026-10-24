import { FormEvent, useEffect, useState } from "react";
import {
  RequestDetail,
  RequestSummary,
  createRequest,
  getRequest,
  listRequests,
  MOCK_MODE,
} from "./api";

const suggestions = [
  "Walk backwards",
  "Stand on two legs",
  "Do a little dance",
];
export default function App() {
  const [runs, setRuns] = useState<RequestSummary[]>([]);
  const [selected, setSelected] = useState(location.hash.slice(1));
  const [detail, setDetail] = useState<RequestDetail | null>(null);
  const [round, setRound] = useState<number | null>(null);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [playing, setPlaying] = useState(true);
  useEffect(() => {
    let active = true;
    async function refresh() {
      try {
        const data = await listRequests();
        if (active) {
          setRuns(data);
          if (!selected && data[0]) setSelected(data[0].id);
        }
      } catch (e) {
        if (active) setError(String(e));
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
    if (!selected) return;
    let active = true;
    location.hash = selected;
    setDetail(null);
    setRound(null);
    async function refresh() {
      try {
        const data = await getRequest(selected);
        if (active) setDetail(data);
      } catch (e) {
        if (active) setError(String(e));
      }
    }
    refresh();
    const timer = setInterval(refresh, 1500);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [selected]);
  useEffect(() => {
    const change = () => setSelected(location.hash.slice(1));
    window.addEventListener("hashchange", change);
    return () => window.removeEventListener("hashchange", change);
  }, []);
  async function submit(e: FormEvent) {
    e.preventDefault();
    if (text.trim().length < 3 || busy) return;
    setBusy(true);
    setError("");
    try {
      const result = await createRequest(text.trim());
      setSelected(result.id);
      setText("");
    } catch (e) {
      setError(`Could not start run: ${String(e)}`);
    } finally {
      setBusy(false);
    }
  }
  const current =
    detail?.rounds.find((r) => r.n === round) ?? detail?.rounds.at(-1);
  const candidate =
    current?.candidates.find((c) => c.id === current.best_candidate_id) ??
    current?.candidates[0];
  const running = detail?.status === "running" || detail?.status === "queued";
  const stage = detail?.events.at(-1)?.message ?? "Preparing experiment";
  return (
    <div className="shell">
      <aside className="sidebar">
        <a className="brand" href="#" onClick={() => setSelected("")}>
          <span className="brand-icon">⌘</span> rewardsmith
          <span className="brand-dot">.</span>
        </a>
        <div className="workspace-label">WORKSPACE</div>
        <div className="nav-item">
          <span>▦</span> Experiments{" "}
          <span className="count">{runs.length}</span>
        </div>
        <div className="sidebar-heading">RECENT RUNS</div>
        <nav aria-label="Recent experiments">
          {runs.map((run) => (
            <button
              key={run.id}
              aria-pressed={selected === run.id}
              onClick={() => {
                setSelected(run.id);
                setError("");
              }}
              className={`run-link ${selected === run.id ? "selected" : ""}`}
            >
              <span
                className={`status-dot ${run.status === "done" ? "green" : ""}`}
              />
              <span>{run.text}</span>
            </button>
          ))}
        </nav>
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
            <strong>Experiments</strong>
          </span>
          <span className="mode">
            <span className="status-dot" />
            {MOCK_MODE ? "Demo mode · Mock API" : "Live API"}
          </span>
        </header>
        <div className="content">
          <select
            className="mobile-runs"
            aria-label="Select experiment"
            value={selected}
            onChange={(event) => { setSelected(event.target.value); setError(""); }}
          >
            {!selected && <option value="">Select an experiment</option>}
            {runs.map((run) => <option key={run.id} value={run.id}>{run.text} · {run.status}</option>)}
          </select>
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
                onChange={(e) => setText(e.target.value)}
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
              {suggestions.map((s) => (
                <button
                  type="button"
                  key={s}
                  onClick={() =>
                    setText(`Make the robot dog ${s.toLowerCase()}`)
                  }
                >
                  {s} ↗
                </button>
              ))}
            </div>
          </form>
          {error && (
            <div className="error" role="alert">
              {error}
            </div>
          )}
          {!detail ? (
            <div className="loading" role="status">
              Loading experiment…
            </div>
          ) : (
            <>
              <div className="experiment-title">
                <div>
                  <span className="eyebrow">CURRENT EXPERIMENT</span>
                  <h2>{detail.text}</h2>
                </div>
                <span className={`badge ${running ? "working" : ""}`}>
                  <span className="status-dot green" />
                  {running
                    ? "In progress"
                    : detail.status === "failed"
                      ? "Failed"
                      : "Completed"}
                </span>
              </div>
              <div className="pipeline">
                <span className="pipeline-label">
                  {running ? stage : "Training complete"}
                </span>
                <div className="steps">
                  {["Write", "Check", "Train", "Render", "Critique"].map(
                    (s, i) => (
                      <span key={s}>
                        <b>{running ? String(i + 1).padStart(2, "0") : "✓"}</b>
                        {s}
                      </span>
                    ),
                  )}
                </div>
                <span className="round-count">
                  {detail.rounds.length} / 3 rounds
                </span>
              </div>
              <div className="result-grid">
                <section className="preview-card" aria-label="Robot rollout">
                  <div className="panel-heading">
                    <span>
                      Policy preview{" "}
                      <span className="subtle">
                        / Round {current?.n ?? "—"}
                      </span>
                    </span>
                    <span className="mini-badge">
                      {MOCK_MODE ? "ILLUSTRATIVE SIMULATION" : "ROLLOUT"}
                    </span>
                  </div>
                  {candidate?.video_path ? (
                    <video
                      src={`/artifacts/${candidate.video_path}`}
                      controls
                      muted
                      loop
                      playsInline
                    />
                  ) : (
                    <RobotScene playing={playing} round={current?.n ?? 0} />
                  )}
                  <div className="playback">
                    <button
                      aria-label={
                        playing ? "Pause animation" : "Play animation"
                      }
                      onClick={() => setPlaying(!playing)}
                    >
                      {playing ? "Ⅱ" : "▶"}
                    </button>
                    <span>
                      {MOCK_MODE
                        ? "Animated placeholder · no training footage"
                        : "Awaiting rollout footage"}
                    </span>
                    <span className="playback-end">MuJoCo Playground</span>
                  </div>
                </section>
                <section className="evaluation">
                  <div className="eyebrow">
                    {candidate
                      ? "CANDIDATE EVALUATION"
                      : "EXPERIMENT IN PROGRESS"}
                  </div>
                  <div className="score">
                    {candidate?.score?.toFixed(1) ?? "—"}
                    <span>/ 10</span>
                    <span className="score-label">Critic score</span>
                  </div>
                  <div className="metrics">
                    {[
                      ["Backward speed", "backward_speed", "m/s"],
                      ["Stability", "stability", "%"],
                      ["Motion smoothness", "smoothness", "%"],
                    ].map(([label, key, unit]) => (
                      <div className="metric" key={key}>
                        <div>
                          <span>{label}</span>
                          <strong>
                            {candidate?.metrics?.[key] ?? "—"}{" "}
                            <small>{unit}</small>
                          </strong>
                        </div>
                        <div className="meter">
                          <i
                            style={{
                              width: `${candidate?.metrics ? Math.min(100, candidate.metrics[key] * (unit === "m/s" ? 100 : 1)) : 0}%`,
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="critic">
                    <span>✧ CRITIC NOTES</span>
                    <p>
                      {candidate?.critique ??
                        "The worker is exploring reward functions. Results will appear as each round finishes."}
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
                  const c = r.candidates[0];
                  return (
                    <button
                      key={r.id}
                      aria-pressed={current?.n === r.n}
                      className={`attempt ${current?.n === r.n ? "active" : ""}`}
                      onClick={() => setRound(r.n)}
                    >
                      <div className="attempt-top">
                        <span>ROUND {String(r.n).padStart(2, "0")}</span>
                        {c.score === detail.best_score && (
                          <span className="best">↗ BEST RESULT</span>
                        )}
                      </div>
                      <div className="attempt-score">
                        {c.score?.toFixed(1)}
                        <small>/ 10</small>
                        <span>
                          {r.n === 1
                            ? "First steps"
                            : r.n === 2
                              ? "Finding balance"
                              : "A smoother stride"}
                        </span>
                      </div>
                      <div className="attempt-bars">
                        {[25, 40, 34, 58, 44, 63, 55, 80, 69, 87, 78, 96].map(
                          (v, i) => (
                            <i
                              key={i}
                              style={{ height: `${v * (0.6 + r.n * 0.13)}%` }}
                            />
                          ),
                        )}
                      </div>
                    </button>
                  );
                })}
                {running && (
                  <div className="attempt pending">
                    <span className="spinner" />
                    <strong>{stage}</strong>
                    <span>Next candidate is on its way</span>
                  </div>
                )}
              </div>
              {candidate && (
                <div className="technical">
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
                  <details>
                    <summary>
                      Activity log <span>{detail.events.length} events ↗</span>
                    </summary>
                    {detail.events.map((event) => (
                      <p key={event.id}>{event.message}</p>
                    ))}
                  </details>
                </div>
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
function RobotScene({ playing, round }: { playing: boolean; round: number }) {
  return (
    <div className={`scene ${playing ? "" : "paused"}`}>
      <span className="scene-label">
        <i /> {round ? "POLICY ROLLOUT" : "PREPARING POLICY"}
      </span>
      <div className="floor" />
      <svg
        viewBox="0 0 720 330"
        role="img"
        aria-label="Illustrative animated quadruped on a simulation grid"
      >
        <defs>
          <linearGradient id="body" x2="0.8" y2="1">
            <stop stopColor="#f8faf8" />
            <stop offset="1" stopColor="#9daaa4" />
          </linearGradient>
        </defs>
        <ellipse
          cx="365"
          cy="267"
          rx="115"
          ry="17"
          fill="#324a3a"
          opacity=".12"
        />
        <g className="robot">
          <g
            stroke="#414e48"
            strokeWidth="13"
            strokeLinecap="round"
            strokeLinejoin="round"
            fill="none"
          >
            <path className="leg back" d="M310 172 L290 218 L315 256" />
            <path className="leg front" d="M406 172 L430 212 L413 253" />
          </g>
          <path
            d="M278 141 L388 125 L445 151 L335 174 Z"
            fill="#e3eae5"
            stroke="#697b70"
          />
          <path d="M278 141 L335 165 L335 194 L278 168 Z" fill="#8c9d92" />
          <path
            d="M335 165 L445 143 L445 175 L335 195 Z"
            fill="url(#body)"
            stroke="#819489"
          />
          <path d="M290 139 L382 126 L419 141 L327 157 Z" fill="#f7faf7" />
          <path d="M315 133 L365 126 L387 136 L336 145 Z" fill="#f3bd65" />
          <rect x="426" y="140" width="35" height="27" rx="8" fill="#344c40" />
          <circle cx="451" cy="151" r="4" fill="#b9eec8" />
          <g
            stroke="#35463d"
            strokeWidth="14"
            strokeLinecap="round"
            strokeLinejoin="round"
            fill="none"
          >
            <path className="leg front" d="M301 171 L326 216 L299 259" />
            <path className="leg back" d="M413 182 L388 221 L414 263" />
          </g>
          <g fill="#e4eae5" stroke="#718579" strokeWidth="3">
            <circle cx="301" cy="171" r="12" />
            <circle cx="413" cy="182" r="12" />
          </g>
        </g>
        <g transform="translate(72 255)" strokeWidth="2">
          <path d="M0 0 L35 0" stroke="#ad6c59" />
          <path d="M0 0 L0 -35" stroke="#6e9177" />
          <path d="M0 0 L-18 15" stroke="#7a91a4" />
        </g>
        <text x="114" y="259" fill="#9b7a6d" fontSize="10">
          X
        </text>
        <text x="69" y="212" fill="#6e9177" fontSize="10">
          Z
        </text>
      </svg>
      <span className="scene-bottom">
        QUADRUPED / {round ? `ITERATION 0${round}` : "STANDBY"}
      </span>
      <span className="scene-coords">x −0.64 &nbsp; y 0.02 &nbsp; z 0.32</span>
    </div>
  );
}
