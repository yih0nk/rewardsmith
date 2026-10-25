import { MessageOptions as Options } from "../api";
import { MODES, MODE_MODEL, modelLabel } from "../modes";

// The composer's per-message controls: an Effort switch (off = Instant), and the
// Claude model that writes, revises and judges the rewards. Switching mode moves the
// model to that mode's default unless the user picked one for this message.
export default function MessageOptions({
  value,
  onChange,
  models,
  idPrefix,
}: {
  value: Options & { picked?: boolean };
  onChange: (next: Options & { picked?: boolean }) => void;
  models: string[];
  idPrefix: string;
}) {
  const effort = value.mode === "effort";
  const hint = MODES.find((m) => m.id === value.mode)?.hint;
  const next = effort ? "instant" : "effort";
  return (
    <div className="message-options">
      <button
        type="button"
        role="switch"
        aria-checked={effort}
        aria-describedby={`${idPrefix}-mode-hint`}
        className="toggle"
        onClick={() =>
          onChange({
            ...value,
            mode: next,
            model: value.picked ? value.model : MODE_MODEL[next],
          })
        }
      >
        <span className="toggle-track" aria-hidden="true" />
        Effort
      </button>
      <span className="mode-hint subtle" id={`${idPrefix}-mode-hint`}>
        {hint}
      </span>
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
    </div>
  );
}
