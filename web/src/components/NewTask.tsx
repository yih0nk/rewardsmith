import { ReactNode } from "react";
import QrPanel from "./QrPanel";

// The empty conversation a new task starts from: what the board does, then the composer.
export default function NewTask({
  busyWith,
  composer,
}: {
  // The task on the GPU now, if any: a new one waits for it.
  busyWith: string | null;
  composer: ReactNode;
}) {
  return (
    <div className="thread">
      <article className="cmd turn intro" aria-labelledby="new-task-title">
        <h1 id="new-task-title" className="intro-title">
          Teach a robot dog a new behavior,{" "}
          <span className="em">in plain English.</span>
        </h1>
        <ol className="how" aria-label="How it works">
          <li>You describe a behavior</li>
          <li>Claude writes reward functions (code that scores motion)</li>
          <li>Robots train in a physics simulator</li>
          <li>Claude judges the videos and improves the best reward</li>
          <li className="how-time">about 5–15 min per task</li>
        </ol>
        {busyWith && (
          <p className="intro-note">
            The GPU is training “{busyWith}” right now. A new task waits in line
            and starts when it finishes.
          </p>
        )}
        <div className="intro-qr">
          <QrPanel />
        </div>
      </article>
      {composer}
    </div>
  );
}
