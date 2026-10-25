import { CheckResult } from "../api";
import { checkLabel, checkValue, visibleClauses } from "../explain";

// "Fell over" is yes or no; "< 0.5" would mean nothing to a reader.
function need(metric: string, op: string, threshold: number): string {
  if (metric === "fell") return op.startsWith("<") ? "no" : "yes";
  return `${op} ${checkValue(metric, threshold)}`;
}

// Short headers that never break mid-word; plain metric names, with the raw name on hover.
export default function SuccessCheck({ check }: { check: CheckResult }) {
  const verified = check.verified !== false;
  // One fall row: the per-episode rate, not also the single test run's "fell".
  const { shown, hidden } = visibleClauses(check.clauses ?? []);
  const passed = shown.filter((c) => c.pass).length;
  const hiddenFail = verified && hidden.some((c) => !c.pass);
  return (
    <div className="success-check">
      <p className={`check-result ${check.pass && verified ? "pass" : ""}`}>
        Success check:{" "}
        {verified ? (check.pass ? "passed" : "failed") : "unavailable"}
        {verified && shown.length > 0 && (
          <span className="subtle num">
            {" "}
            · {passed} of {shown.length} criteria met
          </span>
        )}
      </p>
      {check.error && <p className="subtle">{check.error}</p>}
      {shown.length > 0 && (
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
              {shown.map((clause, i) => (
                <tr key={i}>
                  <th scope="row" title={clause.metric}>
                    {checkLabel(clause.metric)}
                  </th>
                  <td className="num">
                    {checkValue(clause.metric, clause.actual)}
                  </td>
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
      {hiddenFail && (
        <p className="check-note">
          <span aria-hidden="true">✕ </span>
          The robot also fell in the recorded test run, and the check counts
          that run too.
        </p>
      )}
    </div>
  );
}
