import { ReactNode, useEffect, useState } from "react";
import {
  ApiError,
  Health,
  RequestDetail as Detail,
  RequestSummary,
  getRequest,
} from "../api";
import { Conversation, RenderReply, queuePlace } from "../thread";
import RequestDetail, { turnLabel } from "./RequestDetail";

// One request's details, polled fast while it runs and slowly after.
function useRequest(id: string, live: boolean) {
  const [detail, setDetail] = useState<Detail | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    const refresh = () =>
      getRequest(id)
        .then((data) => {
          if (active) {
            setDetail(data);
            setError("");
          }
        })
        .catch((err) => {
          if (!active) return;
          setError(
            err instanceof ApiError && err.status === 404
              ? "This run no longer exists on the server."
              : `Can't load this run (${String(err)}). Retrying.`,
          );
        });
    refresh();
    const timer = setInterval(refresh, live ? 1500 : 15000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [id, live]);
  return { detail, error };
}

function TurnView({
  requestId,
  text,
  turn,
  summary,
  requests,
  health,
  onRetry,
  renderReply,
}: {
  requestId: string;
  text: string;
  turn: number;
  summary: RequestSummary | undefined;
  requests: RequestSummary[];
  health: Health | null;
  onRetry: (text: string) => void;
  renderReply?: RenderReply;
}) {
  const live = summary?.status === "running" || summary?.status === "queued";
  const { detail, error } = useRequest(requestId, live);
  // Until the run loads, its message already says which turn this is.
  if (!detail)
    return (
      <article className="cmd turn" data-n={turnLabel(turn)}>
        <div className="exp-head">
          <span className="who">You</span>
          <h2 className="exp-title">{text}</h2>
        </div>
        {error ? (
          <div className="error" role="alert">
            {error}
          </div>
        ) : (
          <p className="turn-loading subtle" role="status">
            Loading…
          </p>
        )}
      </article>
    );
  return (
    <RequestDetail
      detail={detail}
      health={health}
      onRetry={onRetry}
      queue={queuePlace(requests, requestId)}
      renderReply={renderReply}
    />
  );
}

// The selected task as a thread of turns. `composer` is the reply box; it is only
// passed once the backend accepts follow-ups, so today the thread ends at the run.
export default function ConversationView({
  conversation,
  requests,
  health,
  onRetry,
  renderReply,
  composer,
}: {
  conversation: Conversation;
  requests: RequestSummary[];
  health: Health | null;
  onRetry: (text: string) => void;
  renderReply?: RenderReply;
  composer?: ReactNode;
}) {
  return (
    <div className="thread">
      <h1 className="sr-only">Task: {conversation.title}</h1>
      {conversation.turns.map((turn) => (
        <TurnView
          key={turn.requestId}
          requestId={turn.requestId}
          text={turn.text}
          turn={turn.turn}
          summary={requests.find((r) => r.id === turn.requestId)}
          requests={requests}
          health={health}
          onRetry={onRetry}
          renderReply={renderReply}
        />
      ))}
      {composer}
    </div>
  );
}
