import { Candidate } from "../api";

export default function CandidateCard({ c, best }: { c: Candidate; best: boolean }) {
  return (
    <article className={best ? "cand best" : "cand"}>
      <div className="cand-head">
        <span>#{c.idx}</span>
        <span>{c.score != null ? c.score.toFixed(1) : c.status}</span>
      </div>
      {c.video_path ? (
        <video src={`/artifacts/${c.video_path}`} controls muted loop playsInline />
      ) : (
        <div className="novideo muted">{c.status === "gate_failed" ? "Didn't compile" : "No video yet"}</div>
      )}
      {c.critique && <p className="critique">{c.critique}</p>}
      {c.error && <pre className="err">{c.error}</pre>}
      <details>
        <summary>{c.diff ? "Diff" : "Reward"}</summary>
        <pre>{c.diff || c.reward_src}</pre>
      </details>
    </article>
  );
}
