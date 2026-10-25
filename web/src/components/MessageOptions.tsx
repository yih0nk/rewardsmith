import { BodyInfo, MessageOptions as Options } from "../api";
import { MODES, MODE_MODEL, modelLabel } from "../modes";

// The composer's per-message controls: Instant / Effort, the Claude model that
// writes, revises and judges the rewards, and (new tasks only: bodies given) the robot.
// Switching mode moves the model to that mode's default unless the user picked one
// for this message.
export default function MessageOptions({
  value,
  onChange,
  models,
  bodies,
  idPrefix,
}: {
  value: Options & { picked?: boolean };
  onChange: (next: Options & { picked?: boolean }) => void;
  models: string[];
  // The registered robots; leave it out for a follow-up, which keeps its task's robot.
  bodies?: BodyInfo[];
  idPrefix: string;
}) {
  const hint = MODES.find((m) => m.id === value.mode)?.hint;
  return (
    <div className="message-options">
      <div className="mode-toggle" role="radiogroup" aria-label="Mode" aria-describedby={`${idPrefix}-mode-hint`}>
        {MODES.map((m) => (
          <button
            type="button"
            key={m.id}
            role="radio"
            aria-checked={value.mode === m.id}
            className={`mode-option ${value.mode === m.id ? "on" : ""}`}
            onClick={() =>
              onChange({
                ...value,
                mode: m.id,
                model: value.picked ? value.model : MODE_MODEL[m.id],
              })
            }
          >
            {m.label}
          </button>
        ))}
      </div>
      <label className="model-pick">
        <span className="label">Model</span>
        <select
          value={value.model}
          onChange={(event) => onChange({ ...value, model: event.target.value, picked: true })}
        >
          {models.map((id) => (
            <option key={id} value={id}>
              {modelLabel(id)}
            </option>
          ))}
        </select>
      </label>
      {bodies && (
        <label className="model-pick body-pick">
          <span className="label">Robot</span>
          <select
            value={value.body ?? "auto"}
            title={bodies.find((b) => b.name === value.body)?.summary ?? "Claude picks the robot that fits the task"}
            onChange={(event) => onChange({ ...value, body: event.target.value })}
          >
            <option value="auto">Auto</option>
            {bodies.map((b) => (
              <option key={b.name} value={b.name}>
                {b.title}
              </option>
            ))}
          </select>
        </label>
      )}
      <span className="mode-hint subtle" id={`${idPrefix}-mode-hint`}>
        {hint}
      </span>
    </div>
  );
}
