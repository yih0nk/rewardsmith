import { FormEvent, ReactNode, RefObject, useState } from "react";

const EXAMPLES = ["Walk backwards", "Spin in place", "Walk to the red ball"];

// The message box, pinned to the bottom of the pane. Today it starts a new task.
// `controls` is where per-message options (Instant/Effort, model) go once the
// backend reads them; nothing renders there until then.
export default function Composer({
  label,
  inputRef,
  busy,
  error,
  text,
  onText,
  onSubmit,
  examples = true,
  controls,
}: {
  label: string;
  inputRef: RefObject<HTMLInputElement>;
  busy: boolean;
  error: string;
  text: string;
  onText: (text: string) => void;
  onSubmit: (text: string) => void;
  examples?: boolean;
  controls?: ReactNode;
}) {
  const [hint, setHint] = useState("");
  function submit(event: FormEvent) {
    event.preventDefault();
    if (busy) return;
    if (text.trim().length < 3) {
      setHint(
        "Type a behavior first (at least 3 characters), or click an example.",
      );
      inputRef.current?.focus();
      return;
    }
    setHint("");
    onSubmit(text.trim());
  }
  const message = error || hint;
  return (
    <form className="composer composer-dock" onSubmit={submit}>
      <label htmlFor="behavior" className="composer-label">
        {label}
      </label>
      <div className="composer-row">
        <span className="prompt" aria-hidden="true">
          &gt;
        </span>
        <input
          id="behavior"
          ref={inputRef}
          value={text}
          onChange={(event) => {
            onText(event.target.value);
            setHint("");
          }}
          maxLength={300}
          placeholder="Make the robot dog walk backwards…"
          aria-describedby="behavior-help"
          autoComplete="off"
        />
        <button className="primary" disabled={busy}>
          {busy ? "Starting…" : "Start training"}
        </button>
      </div>
      {controls && <div className="composer-controls">{controls}</div>}
      <p
        className="composer-help error-text"
        id="behavior-help"
        role="alert"
        hidden={!message}
      >
        {message}
      </p>
      {examples && (
        <div className="suggestions">
          <span className="label">Examples, click to use</span>
          {EXAMPLES.map((example) => (
            <button
              type="button"
              className="link"
              key={example}
              onClick={() => {
                onText(`Make the robot dog ${example.toLowerCase()}`);
                setHint("");
                inputRef.current?.focus();
              }}
            >
              {example}
            </button>
          ))}
        </div>
      )}
    </form>
  );
}
