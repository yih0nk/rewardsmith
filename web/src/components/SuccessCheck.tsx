import { CheckResult } from "../api";
import { metricInfo } from "../explain";

const names: Record<string, string> = {
  target_reached: "Reached target",
  target_fell: "Episodes with a fall",
  target_final_dist: "Final target distance",
  scene_fell: "Episodes with a fall",
};

function label(metric: string): string {
  if (names[metric]) return names[metric];
  if (metric.startsWith("sequence:"))
    return metric.slice(9).split(">").map(label).join(", then ");
  if (metric.startsWith("all:"))
    return metric.slice(4).split("&").map(label).join(" and ");
  const [event, duration] = metric.split("@");
  const [kind, object, zone] = event.split(":");
  const objectName = object?.replaceAll("_", " ");
  const descriptions: Record<string, string> = {
    reach: `Reach ${objectName}`,
    push: `Push ${objectName} into ${zone?.replaceAll("_", " ")}`,
    climb: `Climb onto ${objectName}`,
    stand_on: `Stand still on ${objectName}`,
    avoid: `Avoid ${objectName} with progress`,
    stand_still: "Stand still",
    move: "Move from the start",
    forward: "Move forward",
    backward: "Move backward",
    spin: "Complete a turn",
  };
  return (
    (descriptions[kind] ?? metricInfo(metric).label) +
    (duration ? ` for ${duration} s` : "")
  );
}

function isRate(metric: string): boolean {
  return (
    [
      "target_reached",
      "target_fell",
      "scene_fell",
      "stand_still",
      "move",
      "forward",
      "backward",
      "spin",
    ].includes(metric.split("@")[0]) ||
    /^(?:all|sequence|reach|push|climb|stand_on|avoid):/.test(metric)
  );
}

function value(metric: string, actual: number | null): string {
  if (actual == null || !Number.isFinite(actual)) return "Not measured";
  if (isRate(metric)) {
    return `${Number((actual * 100).toFixed(1))}%`;
  }
  return metricInfo(metric).format(actual);
}

// "Fell over" is yes or no; "< 0.5" would mean nothing to a reader.
function need(metric: string, op: string, threshold: number): string {
  if (metric === "fell") return op.startsWith("<") ? "no" : "yes";
  return `${op} ${value(metric, threshold)}`;
}

// Short headers that never break mid-word; plain metric names, with the raw name on hover.
export default function SuccessCheck({ check }: { check: CheckResult }) {
  const verified = check.verified !== false;
  const passed = check.clauses?.filter((c) => c.pass).length ?? 0;
  return (
    <div className="success-check">
      <p className={`check-result ${check.pass && verified ? "pass" : ""}`}>
        Success check:{" "}
        {verified ? (check.pass ? "passed" : "failed") : "unavailable"}
        {verified && check.clauses?.length > 0 && (
          <span className="subtle num">
            {" "}
            · {passed} of {check.clauses.length} criteria met
          </span>
        )}
      </p>
      {check.error && <p className="subtle">{check.error}</p>}
      {check.clauses?.length > 0 && (
        <div className="table-scroll">
          <table className="check-criteria" aria-label="Success check criteria">
            <thead>
              <tr>
                <th scope="col">Metric</th>
                <th scope="col">Got</th>
                <th scope="col">Need</th>
                <th scope="col">Result</th>
              </tr>
            </thead>
            <tbody>
              {check.clauses.map((clause, i) => (
                <tr key={i}>
                  <th scope="row" title={clause.metric}>
                    {label(clause.metric)}
                  </th>
                  <td className="num">{value(clause.metric, clause.actual)}</td>
                  <td className="num">
                    {need(clause.metric, clause.op, clause.value)}
                  </td>
                  <td
                    className={verified ? (clause.pass ? "pass" : "fail") : ""}
                  >
                    {verified && (
                      <span aria-hidden="true">
                        {clause.pass ? "✓ " : "✕ "}
                      </span>
                    )}
                    <span className="result-word">
                      {verified
                        ? clause.pass
                          ? "Pass"
                          : "Fail"
                        : "Unverified"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
