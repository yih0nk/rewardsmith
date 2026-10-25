import { useEffect, useState } from "react";
import { RequestSummary, artifactUrl, getRequest } from "../api";
import { Conversation, titleCase } from "../thread";

// How many finished tasks the new-task page shows, and the lowest score that counts.
const SHOWN = 4;
const MIN_SCORE = 7;

interface Pick {
  conversationId: string;
  // The turn that holds the task's best score: its best candidate's video is the clip.
  requestId: string;
  title: string;
  score: number;
}

// The best finished tasks, one per distinct prompt, highest score then newest first.
export function showcasePicks(
  conversations: Conversation[],
  requests: RequestSummary[],
): Pick[] {
  const seen = new Set<string>();
  const picks: Pick[] = [];
  const ranked = conversations
    .filter((c) => c.bestScore != null && c.bestScore >= MIN_SCORE)
    .sort((a, b) => b.bestScore! - a.bestScore! || b.createdAt - a.createdAt);
  for (const c of ranked) {
    const key = c.title.trim().toLowerCase();
    if (seen.has(key)) continue;
    const best = requests
      .filter((r) => (r.conversation_id ?? r.id) === c.id && r.best_score === c.bestScore)
      .sort((a, b) => b.created_at - a.created_at)[0];
    if (!best) continue;
    seen.add(key);
    picks.push({ conversationId: c.id, requestId: best.id, title: c.title, score: c.bestScore! });
    if (picks.length === SHOWN) break;
  }
  return picks;
}

// Finished robots, looping, so the page shows what the board does before anyone types.
export default function Showcase({
  picks,
  onOpen,
}: {
  picks: Pick[];
  onOpen: (conversationId: string) => void;
}) {
  // Request id → its best candidate's video path; fetched once per pick.
  const [videos, setVideos] = useState<Record<string, string | null>>({});
  const wanted = picks.map((p) => p.requestId).join(",");
  useEffect(() => {
    let active = true;
    for (const id of wanted.split(",").filter(Boolean)) {
      if (id in videos) continue;
      getRequest(id)
        .then((d) => {
          const best = d.rounds
            .flatMap((r) => r.candidates)
            .find((c) => c.id === d.best_candidate_id);
          if (active) setVideos((v) => ({ ...v, [id]: best?.video_path ?? null }));
        })
        .catch(() => active && setVideos((v) => ({ ...v, [id]: null })));
    }
    return () => {
      active = false;
    };
    // `videos` is a cache filled here: refetching on its change would loop.
  }, [wanted]);
  const shown = picks.filter((p) => videos[p.requestId] !== null);
  if (!shown.length) return null;
  const still =
    typeof window !== "undefined" &&
    window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  return (
    <section className="showcase" aria-labelledby="showcase-title">
      <h2 id="showcase-title" className="showcase-title">
        Trained here, from one sentence each
      </h2>
      <ul className="showcase-grid">
        {shown.map((p) => {
          const video = videos[p.requestId];
          return (
            <li key={p.conversationId}>
              <button
                type="button"
                className="showcase-card"
                onClick={() => onOpen(p.conversationId)}
              >
                <span className="showcase-media">
                  {video ? (
                    <video
                      src={artifactUrl(video)}
                      autoPlay={!still}
                      muted
                      loop
                      playsInline
                      preload="metadata"
                      aria-hidden="true"
                    />
                  ) : (
                    <span className="subtle">Loading…</span>
                  )}
                </span>
                <span className="showcase-caption">
                  <span className="showcase-prompt">{titleCase(p.title)}</span>
                  <span className="showcase-score num">
                    {p.score.toFixed(1)}
                    <small>/10</small>
                  </span>
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
