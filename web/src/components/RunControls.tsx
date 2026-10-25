import { useEffect, useState } from "react";
import { RequestSummary, RunAction, controlRequest } from "../api";

// Pause, resume and cancel for one run. A running run pauses after its current round and
// cancels at the next checkpoint, so those show as pending until the worker gets there.
export default function RunControls({ run }: { run: RequestSummary }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  // Clear a stale error once the run moves on (the 1.5 s poll brings the new state).
  useEffect(() => setError(""), [run.status, run.control]);

  const act = async (action: RunAction) => {
    setBusy(true);
    setError("");
    try {
      await controlRequest(run.id, action);
    } catch (err) {
      setError(`Could not ${action}: ${String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  const button = (action: RunAction, label: string) => (
    <button
      key={action}
      className={`control ${action}`}
      disabled={busy}
      onClick={() => act(action)}
    >
      {label}
    </button>
  );

  let note = "";
  let actions: JSX.Element[] = [];
  if (run.status === "queued") {
    actions = [button("pause", "Pause"), button("cancel", "Cancel")];
  } else if (run.status === "paused") {
    note = "Paused. Resume to continue from the next round.";
    actions = [button("resume", "Resume"), button("cancel", "Cancel")];
  } else if (run.status === "running" && run.control === "cancel") {
    note = "Cancelling at the next checkpoint…";
  } else if (run.status === "running" && run.control === "pause") {
    note = "Pausing after this round…";
    actions = [button("resume", "Keep running"), button("cancel", "Cancel")];
  } else if (run.status === "running") {
    actions = [button("pause", "Pause after this round"), button("cancel", "Cancel")];
  } else {
    return null; // done, failed and cancelled runs have nothing to control
  }

  return (
    <div className="run-controls">
      {note && (
        <span className="run-controls-note" role="status">
          {note}
        </span>
      )}
      {actions.length > 0 && <span className="run-controls-actions">{actions}</span>}
      {error && (
        <span className="candidate-error" role="alert">
          {error}
        </span>
      )}
    </div>
  );
}
