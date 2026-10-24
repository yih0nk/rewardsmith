// A unified diff with added, removed and hunk lines colored.
const lineClass = (line: string) =>
  line.startsWith("+++") || line.startsWith("---")
    ? "diff-file"
    : line.startsWith("+")
      ? "diff-add"
      : line.startsWith("-")
        ? "diff-del"
        : line.startsWith("@@")
          ? "diff-hunk"
          : undefined;

export default function DiffView({ diff }: { diff: string }) {
  return (
    <pre className="diff">
      {diff.replace(/\n$/, "").split("\n").map((line, i) => (
        <span key={i} className={lineClass(line)}>
          {line || " "}
        </span>
      ))}
    </pre>
  );
}
