import { useEffect, useRef } from "react";
import { Conversation } from "../thread";

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
        {conversations.map((c) => {
          const status =
            c.status === "done" && c.bestScore == null
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
                <span className="conversation-title">{c.title}</span>
                <span className="conversation-meta">
                  <span className={`run-status ${c.status}`}>{status}</span>
                  <span className="num">
                    {c.bestScore != null ? (
                      <>
                        best {c.bestScore.toFixed(1)}
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
    </nav>
  );
}
