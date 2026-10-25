import { BodyInfo, MessageOptions as Options } from "../api";
import { MODES, MODE_MODEL, modelLabel } from "../modes";

// The composer's per-message controls: an Effort switch (off = Instant), the Claude
// model that writes, revises and judges the rewards, and (new tasks only: bodies
// given) the robot. Switching mode moves the model to that mode's default unless the
// user picked one for this message.
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
      <details className="training-budget">
        <summary>Training budget <span className="subtle">{value.training_steps == null ? "Auto" : `${value.training_steps / 1_000_000}M`}</span></summary>
        <div className="training-budget-panel">
          <label className="steps-pick">
            <span className="label">Training steps (millions)</span>
            <input
              type="number"
              min={1}
              max={100}
              step={1}
              value={value.training_steps == null ? "" : value.training_steps / 1_000_000}
              placeholder="Auto"
              aria-describedby={`${idPrefix}-steps-hint`}
              onChange={(event) => onChange({
                ...value,
                training_steps: event.target.value === "" ? undefined : Number(event.target.value) * 1_000_000,
              })}
            />
          </label>
          <p className="mode-hint subtle" id={`${idPrefix}-steps-hint`}>
            Per candidate, 1–100M. Blank uses the mode default. More steps take longer; PPO may round up.
            {effort ? " Effort trains four candidates, so 20M means about 80M total steps." : " Instant trains one candidate."}
          </p>
        </div>
      </details>
    </div>
  );
}
