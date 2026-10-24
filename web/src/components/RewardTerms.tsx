import { Candidate } from "../api";

// The term_* metrics a candidate reported, one bar per named reward term.
export default function RewardTerms({
  metrics,
  className = "",
}: {
  metrics: Candidate["metrics"];
  className?: string;
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
    <section className={`reward-terms ${className}`}>
      <h2 className="cmd-title">Reward breakdown</h2>
      {terms.map(([key, value]) => (
        <div className={`term ${value < 0 ? "negative" : ""}`} key={key}>
          <span className="term-name">{key.slice(5).replaceAll("_", " ")}</span>
          {/* Diverging from zero: positive terms grow right, negative ones left. */}
          <span className="term-bar" aria-hidden="true">
            <i
              style={{
                width: `${(Math.abs(value) / largest) * 50}%`,
                [value < 0 ? "right" : "left"]: "50%",
              }}
            />
          </span>
          <strong className="num">{Number(value.toFixed(4))}</strong>
        </div>
      ))}
      <p className="subtle">
        Bars are scaled to the largest term in this candidate. Positive terms
        extend right of zero, negative terms extend left.
      </p>
    </section>
  );
}
