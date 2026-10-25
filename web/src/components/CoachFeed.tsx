import { RequestDetail } from "../api";

type Event = RequestDetail["events"][number];

// How many of the newest notes stay in view; older ones fold away.
const SHOWN = 6;

// The live coach's running commentary: one line per look at a training candidate
// (kind "coach" events), newest last. Renders nothing until the coach has spoken.
export default function CoachFeed({ events }: { events: Event[] }) {
  const notes = events.filter((e) => e.kind === "coach");
  if (!notes.length) return null;
  const older = notes.slice(0, -SHOWN);
  const recent = notes.slice(-SHOWN);
  return (
    <section className="coach-feed" aria-label="Coach commentary">
      <span className="label">Coach, watching training</span>
      {older.length > 0 && (
        <details>
          <summary>
            {older.length} earlier {older.length === 1 ? "note" : "notes"}
          </summary>
          <CoachNotes notes={older} />
        </details>
      )}
      <CoachNotes notes={recent} live />
    </section>
  );
}

function CoachNotes({ notes, live }: { notes: Event[]; live?: boolean }) {
  return (
    <ol aria-live={live ? "polite" : undefined}>
      {notes.map((e) => (
        <li key={e.id} className={e.candidate == null ? "coach-summary" : ""}>
          <span className="coach-who">{who(e)}</span>
          <span>{text(e)}</span>
        </li>
      ))}
    </ol>
  );
}

// "R1 C3" for a note about one candidate, "R1" for a round's summary line.
export function who(e: Event): string {
  const round = e.round_n != null ? `R${e.round_n}` : "";
  return e.candidate != null ? `${round} C${e.candidate + 1}`.trim() : round;
}

// The tag already names the candidate, so drop the message's own "Candidate N at ".
export function text(e: Event): string {
  return e.candidate != null
    ? e.message.replace(/^Candidate \d+ at /, "")
    : e.message;
}
