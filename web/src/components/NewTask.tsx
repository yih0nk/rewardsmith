import { ReactNode } from "react";
import QrPanel from "./QrPanel";

// The empty conversation a new task starts from: what the board does, what it has
// already trained, then the composer.
export default function NewTask({
  busyWith,
  showcase,
  composer,
}: {
  // The task on the GPU now, if any: a new one waits for it.
  busyWith: string | null;
  // Finished robots to show off; empty on a fresh server.
  showcase?: ReactNode;
  composer: ReactNode;
}) {
  return (
    <div className="thread">
      <article className="cmd turn intro" aria-labelledby="new-task-title">
        <div className="intro-head">
          <div>
            <h1 id="new-task-title" className="intro-title">
              Teach a robot dog a new behavior,{" "}
              <span className="em">in plain English.</span>
            </h1>
            <ol className="how" aria-label="How it works">
              <li>You describe a behavior</li>
              <li>Claude writes reward functions (code that scores motion)</li>
              <li>Robots train in a physics simulator</li>
              <li>Claude judges the videos; follow up to improve the best one</li>
              <li className="how-time">about 2 min (Instant) or 5 min (Effort)</li>
            </ol>
            {busyWith && (
              <p className="intro-note">
                The GPU is training “{busyWith}” right now. A new task waits in
                line and starts when it finishes.
              </p>
            )}
          </div>
          <div className="intro-qr">
            <QrPanel />
          </div>
        </div>
      </article>
      {/* Outside the intro, so a phone can put it under the composer (responsive.css). */}
      {showcase}
      {composer}
    </div>
  );
}
