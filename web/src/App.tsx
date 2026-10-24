import { FormEvent, useEffect, useState } from "react";
import { Candidate, RequestDetail, RequestSummary, createRequest, getRequest, listRequests } from "./api";

const POLL_MS = 2000;

// The board. A request id in the URL hash (#abc123) opens that request: that's the phone link.
export default function App() {
  const [requests, setRequests] = useState<RequestSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(() => location.hash.slice(1) || null);
  const [detail, setDetail] = useState<RequestDetail | null>(null);
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const tick = () => listRequests().then(setRequests).catch((e) => setError(String(e)));
    tick();
    const t = setInterval(tick, POLL_MS);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    if (!selected) return setDetail(null);
    location.hash = selected;
    const tick = () => getRequest(selected).then(setDetail).catch((e) => setError(String(e)));
    tick();
    const t = setInterval(tick, POLL_MS);
    return () => clearInterval(t);
  }, [selected]);

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (text.trim().length < 3) return;
    try {
      const { id } = await createRequest(text.trim());
      setText("");
      setSelected(id);
    } catch (err) {
      setError(`Couldn't queue that request: ${err}`);
    }
  }

  return (
    <main>
      <header>
        <h1>Rewardsmith</h1>
        <p className="muted">Type a behavior. Watch the robot learn it.</p>
      </header>

      <form onSubmit={submit} className="ask">
        <label htmlFor="behavior" className="label">Behavior</label>
        <div className="row">
          <input
            id="behavior"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="make the robot dog walk backwards"
            maxLength={300}
          />
          <button type="submit">Teach it</button>
        </div>
      </form>

      {error && <p className="error" role="alert">{error}</p>}

      <div className="layout">
        <section aria-label="Requests">
          <h2 className="label">Requests</h2>
          {requests.length === 0 && <p className="muted">Nothing yet. Type a behavior above to start.</p>}
          <ul className="requests">
            {requests.map((r) => (
              <li key={r.id}>
                <button className={r.id === selected ? "req active" : "req"} onClick={() => setSelected(r.id)}>
                  <span>{r.text}</span>
                  <span className="muted">
                    {r.status}
                    {r.best_score != null && ` · ${r.best_score.toFixed(1)}`}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </section>

        <section aria-label="Rounds">{detail ? <Detail detail={detail} /> : <p className="muted">Pick a request.</p>}</section>
      </div>
    </main>
  );
}

function Detail({ detail }: { detail: RequestDetail }) {
  return (
    <div>
      <h2>{detail.text}</h2>
      <p className="muted">
        {detail.status} · best score {detail.best_score?.toFixed(1) ?? "n/a"} · link: {location.origin}/#{detail.id}
      </p>
      {detail.rounds.map((round) => (
        <div key={round.id} className="round">
          <h3 className="label">Round {round.n}</h3>
          <div className="grid">
            {round.candidates.map((c) => (
              <CandidateView key={c.id} c={c} best={c.id === round.best_candidate_id} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function CandidateView({ c, best }: { c: Candidate; best: boolean }) {
  return (
    <article className={best ? "cand best" : "cand"}>
      <div className="cand-head">
        <span>#{c.idx}</span>
        <span>{c.score != null ? c.score.toFixed(1) : c.status}</span>
      </div>
      {c.video_path ? (
        <video src={`/artifacts/${c.video_path}`} controls muted loop playsInline />
      ) : (
        <div className="novideo muted">{c.status === "gate_failed" ? "Didn't compile" : "No video yet"}</div>
      )}
      {c.critique && <p className="critique">{c.critique}</p>}
      {c.error && <pre className="err">{c.error}</pre>}
      <details>
        <summary>{c.diff ? "Diff" : "Reward"}</summary>
        <pre>{c.diff || c.reward_src}</pre>
      </details>
    </article>
  );
}
