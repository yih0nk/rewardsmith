import { FormEvent, useEffect, useState } from "react";
import { RequestDetail as Detail, RequestSummary, createRequest, getRequest, listRequests } from "./api";
import RequestDetail from "./components/RequestDetail";
import RequestList from "./components/RequestList";

const POLL_MS = 2000;
const HALL = "hall";

const fromHash = () => location.hash.slice(1) || null;

// The board. A request id in the URL hash (#abc123) opens that request: that's the phone link.
// #hall is the hall of fame.
export default function App() {
  const [requests, setRequests] = useState<RequestSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(fromHash);
  const [detail, setDetail] = useState<Detail | null>(null);
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const tick = () => listRequests().then(setRequests).catch((e) => setError(String(e)));
    tick();
    const t = setInterval(tick, POLL_MS);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    const onHash = () => setSelected(fromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    if (!selected) return setDetail(null);
    location.hash = selected;
    if (selected === HALL) return setDetail(null);
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
          <RequestList requests={requests} selected={selected} onSelect={setSelected} />
        </section>

        <section aria-label="Rounds">
          {selected === HALL ? (
            <p className="muted">Hall of fame: nothing flagged yet.</p>
          ) : detail ? (
            <RequestDetail detail={detail} />
          ) : (
            <p className="muted">Pick a request.</p>
          )}
        </section>
      </div>
    </main>
  );
}
