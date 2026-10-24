import { Candidate } from "../api";

// The term_* metrics a candidate reported, one bar per named reward term.
export default function RewardTerms({
  metrics,
}: {
  metrics: Candidate["metrics"];
}) {
  const terms = Object.entries(metrics ?? {}).filter(
    (entry): entry is [string, number] =>
      entry[0].startsWith("term_") &&
      typeof entry[1] === "number" &&
      Number.isFinite(entry[1]),
  );
  if (terms.length === 0) return null;
  const largest = Math.max(...terms.map(([, v]) => Math.abs(v)), 0.001);
  return (
    <section className="reward-terms">
      <h2>Reward breakdown</h2>
      {terms.map(([key, value]) => (
        <div className="metric" key={key}>
          <div>
            <span>{key.slice(5).replaceAll("_", " ")}</span>
            <strong>{Number(value.toFixed(4))}</strong>
          </div>
          <div className="meter">
            <i
              style={{
                width: `${(Math.abs(value) / largest) * 100}%`,
                background: value < 0 ? "#b17c67" : undefined,
              }}
            />
          </div>
        </div>
      ))}
      <p className="subtle">
        Bar lengths show relative magnitude within this candidate; negative
        terms are shown in brown.
      </p>
    </section>
  );
}
