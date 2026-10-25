import { Candidate } from "../api";

// Download a trained candidate as a zip: its policy, its task and run_policy.py, which
// rolls it out in MuJoCo (rewardsmith/export.py). primary marks the run's best result.
export default function ExportPolicy({
  c,
  primary = false,
}: {
  c: Candidate;
  primary?: boolean;
}) {
  if (!c.exportable) return null;
  return (
    <a
      className={`control export-policy ${primary ? "primary" : ""}`}
      href={`/api/candidates/${c.id}/export`}
      download
    >
      Export policy
    </a>
  );
}
