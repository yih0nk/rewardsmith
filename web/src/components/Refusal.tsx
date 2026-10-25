import { Feasibility } from "../api";

// The scene writer's verdict, when there is something to say: a refusal (nothing was
// trained, with the reason and the closest request that would work) or the simplification
// an approximate request was trained with. Renders nothing for doable requests.
export default function Refusal({ feasibility }: { feasibility?: Feasibility | null }) {
  if (!feasibility || feasibility.verdict === "doable") return null;
  const refused = feasibility.verdict === "refuse";
  return (
    <div
      role={refused ? "alert" : "note"}
      className={`refusal ${refused ? "refused" : "approximate"}`}
      style={{
        marginTop: "var(--dp-spacing-row-compact)",
        padding: "var(--dp-spacing-row-compact) var(--dp-spacing-gutter)",
        border: `var(--dp-border-hairline) ${refused ? "solid" : "dashed"} var(--dp-color-emphasis)`,
      }}
    >
      <p style={{ margin: 0, color: "var(--dp-color-emphasis)" }}>
        {refused ? "Can't train this" : "Trained with a simplification"}
      </p>
      {refused ? (
        <>
          {feasibility.reason && <p style={{ margin: 0 }}>{feasibility.reason}</p>}
          {feasibility.suggestion && (
            <p style={{ margin: 0 }} className="subtle">
              Closest thing that works: <q>{feasibility.suggestion}</q>
            </p>
          )}
        </>
      ) : (
        <p style={{ margin: 0 }}>{feasibility.simplification || feasibility.reason}</p>
      )}
    </div>
  );
}
