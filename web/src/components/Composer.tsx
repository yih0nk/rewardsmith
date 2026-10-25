import { FormEvent, ReactNode, RefObject, useState } from "react";

// Prompts that have trained well on the pod (8/10 runs), so a live click doesn't flop.
const EXAMPLES = [
  { label: "Walk sideways", prompt: "walk sideways to the left" },
  { label: "Trot fast", prompt: "trot forward as fast as you can without falling" },
  {
    label: "Hop",
    prompt: "hop with all four feet leaving the ground at the same time",
  },
];

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
  placeholder = "What should the robot learn?",
  submitLabel = "Start training",
  allowEmpty = false,
  compact = false,
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
  // One line until focused: the label, context and options open with the box.
  compact?: boolean;
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
    <form
      className={`composer composer-dock ${compact && !message ? "compact" : ""}`}
      onSubmit={submit}
    >
      {/* The placeholder asks the question; the label stays for screen readers. */}
      <label htmlFor={inputId} className="composer-label sr-only">
        {label}
      </label>
      <div className="composer-card">
        {context && <div className="composer-context">{context}</div>}
        <div className="composer-row">
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
      </div>
      <p
        className="composer-help error-text"
        id={`${inputId}-help`}
        role="alert"
        hidden={!message}
      >
        {message}
      </p>
      {examples && (
        <p className="suggestions">
          <span className="label">Try</span>{" "}
          {EXAMPLES.map((example, i) => (
            <span key={example.label}>
              {i > 0 && <span aria-hidden="true"> · </span>}
              <button
                type="button"
                className="example"
                onClick={() => {
                  onText(example.prompt);
                  setHint("");
                  inputRef.current?.focus();
                }}
              >
                {example.label.toLowerCase()}
              </button>
            </span>
          ))}
        </p>
      )}
    </form>
  );
}
