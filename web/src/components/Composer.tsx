import { FormEvent, ReactNode, RefObject, useState } from "react";

const EXAMPLES = ["Walk backwards", "Spin in place", "Walk to the red ball"];

// The message box, pinned to the bottom of the pane. It starts a new task, or, at the
// bottom of a task, sends a follow-up. `controls` holds the per-message options
// (Instant/Effort, model); `context` says what a follow-up refines.
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
  context,
  inputId = "behavior",
  placeholder = "Make the robot dog walk backwards…",
  submitLabel = "Start training",
  allowEmpty = false,
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
  context?: ReactNode;
  inputId?: string;
  placeholder?: string;
  submitLabel?: string;
  // A follow-up may be empty: it means "keep going".
  allowEmpty?: boolean;
}) {
  const [hint, setHint] = useState("");
  function submit(event: FormEvent) {
    event.preventDefault();
    if (busy) return;
    if (!(allowEmpty && !text.trim()) && text.trim().length < 3) {
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
      <label htmlFor={inputId} className="composer-label">
        {label}
      </label>
      {context && <div className="composer-context">{context}</div>}
      <div className="composer-row">
        <span className="prompt" aria-hidden="true">
          &gt;
        </span>
        <input
          id={inputId}
          ref={inputRef}
          value={text}
          onChange={(event) => {
            onText(event.target.value);
            setHint("");
          }}
          maxLength={300}
          placeholder={placeholder}
          aria-describedby={`${inputId}-help`}
          autoComplete="off"
        />
        <button className="primary" disabled={busy}>
          {busy ? "Starting…" : submitLabel}
        </button>
      </div>
      {controls && <div className="composer-controls">{controls}</div>}
      <p
        className="composer-help error-text"
        id={`${inputId}-help`}
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
