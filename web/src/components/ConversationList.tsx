import { useEffect, useRef, useState } from "react";
import { Conversation, titleCase } from "../thread";

const STATUS: Record<Conversation["status"], string> = {
  queued: "queued",
  running: "running",
  paused: "paused",
  done: "done",
  failed: "failed",
  cancelled: "cancelled",
};

// The sidebar: one row per task, newest first. Status, best score, then the task.
export default function ConversationList({
  conversations,
  selected,
  loading,
  onSelect,
}: {
  conversations: Conversation[];
  selected: string | null;
  loading: boolean;
  onSelect: (id: string) => void;
}) {
  // Keep the open task visible in a long list (a link straight to an old task).
  const current = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    current.current?.scrollIntoView?.({ block: "nearest" });
  }, [selected]);
  // Cancelled and failed tasks fold away (the open one always shows), so the list
  // leads with robots that trained.
  const [showAll, setShowAll] = useState(false);
  const stopped = (c: Conversation) =>
    (c.status === "cancelled" || c.status === "failed") && c.id !== selected;
  const hidden = conversations.filter(stopped).length;
  const listed = showAll ? conversations : conversations.filter((c) => !stopped(c));
  return (
    <nav aria-label="Tasks" className="conversations">
      {conversations.length === 0 && (
        <p className="subtle">
          {loading
            ? "Loading tasks…"
            : "No tasks yet. Start one with New task."}
        </p>
      )}
      <ul>
        {listed.map((c) => {
          // A follow-up still working on a finished task: say it is the follow-up, not the task.
          const followUp =
            c.turns.length > 1 &&
            (c.status === "running" || c.status === "queued");
          const status = followUp
            ? `follow-up ${STATUS[c.status]}`
            : c.status === "done" && c.bestScore == null
              ? "no result"
              : STATUS[c.status];
          return (
            <li key={c.id}>
              <button
                ref={c.id === selected ? current : undefined}
                className={`conversation ${c.id === selected ? "selected" : ""}`}
                aria-current={c.id === selected ? "page" : undefined}
                onClick={() => onSelect(c.id)}
              >
                <span className="conversation-title">{titleCase(c.title)}</span>
                <span className="conversation-meta">
                  <span className={`run-status ${c.status}`}>{status}</span>
                  <span className="num">
                    {c.bestScore != null ? (
                      <>
                        <span className="sr-only">best </span>
                        {c.bestScore.toFixed(1)}
                        <span aria-hidden="true">/10</span>
                        <span className="sr-only"> out of 10</span>
                      </>
                    ) : null}
                  </span>
                </span>
              </button>
            </li>
          );
        })}
      </ul>
      {hidden > 0 && (
        <button
          type="button"
          className="link show-stopped"
          aria-expanded={showAll}
          onClick={() => setShowAll((v) => !v)}
        >
          {showAll
            ? "Hide cancelled and failed"
            : `Show ${hidden} cancelled or failed`}
        </button>
      )}
    </nav>
  );
}
