import { FormEvent, useEffect, useState } from "react";
import { Candidate, flagCandidate, unflagCandidate } from "../api";

// Flag a candidate as reward hacking, which adds it to the hall of fame.
export default function FlagControl({ candidate }: { candidate: Candidate }) {
  const [editing, setEditing] = useState(false);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  // Shown until the next detail poll brings the server's copy back.
  const [pending, setPending] = useState<{ note: string | null } | null>(null);
  useEffect(() => setPending(null), [candidate.flagged_at]);

  const flagged = pending
    ? pending.note
    : candidate.flagged_at != null
      ? candidate.flag_note
      : null;

  async function save(event: FormEvent) {
    event.preventDefault();
    if (!note.trim() || busy) return;
    setBusy(true);
    setError("");
    try {
      await flagCandidate(candidate.id, note.trim());
      setPending({ note: note.trim() });
      setEditing(false);
      setNote("");
    } catch (err) {
      setError(`Could not flag: ${String(err)}`);
    } finally {
      setBusy(false);
    }
  }

  async function remove() {
    setBusy(true);
    setError("");
    try {
      await unflagCandidate(candidate.id);
      setPending({ note: null });
    } catch (err) {
      setError(`Could not remove the flag: ${String(err)}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flag">
      {flagged != null ? (
        <>
          <span className="flag-title">⚑ IN THE HALL OF FAME</span>
          <p>{flagged}</p>
          <div className="flag-actions">
            <a href="#hall">View hall of fame</a>
            <button className="link" onClick={remove} disabled={busy}>
              Remove flag
            </button>
          </div>
        </>
      ) : editing ? (
        <form onSubmit={save}>
          <label htmlFor={`flag-${candidate.id}`} className="label">
            How did it game the reward?
          </label>
          <input
            id={`flag-${candidate.id}`}
            value={note}
            onChange={(event) => setNote(event.target.value)}
            maxLength={200}
            placeholder="Scores on velocity by vibrating in place"
            autoFocus
          />
          <div className="flag-actions">
            <button className="primary" disabled={busy || !note.trim()}>
              Add to hall of fame
            </button>
            <button
              type="button"
              className="link"
              onClick={() => setEditing(false)}
            >
              Cancel
            </button>
          </div>
        </form>
      ) : (
        <button className="link flag-open" onClick={() => setEditing(true)}>
          ⚑ Flag as reward hacking
        </button>
      )}
      {error && (
        <p className="candidate-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
