import { Fragment, ReactNode } from "react";
import { metricInfo } from "../explain";

// Claude's notes on a video: a short lead, then the full text, formatted.
//
// The critic writes plain text: the 2-3 numbers its score rests on, what the frames
// show, then what to change, often as "- " or "1. " lists. apply_limits may put
// "Capped at 3: the robot fell." and "Reward hacking suspected." in front.

export type CritiqueBlock =
  { kind: "p"; text: string } | { kind: "ul" | "ol"; items: string[] };

export interface ParsedCritique {
  flags: string[];
  lead: string;
  blocks: CritiqueBlock[];
  words: number;
  // False when the lead already says everything: no "full notes" to open.
  more: boolean;
}

const FLAGS = [
  /^Capped at 3: the robot fell\.\s*/,
  /^Reward hacking suspected\.\s*/,
];
const BULLET = /^\s*[-•*]\s+/;
const NUMBERED = /^\s*\d+[.)]\s+/;
const LEAD_MAX = 320;
const NUMBERS = /^(?:key numbers|(?:the )?score rests on|numbers)\b/i;
// "What the frames show: the robot walks…" → "The robot walks…"
const withoutLabel = (sentence: string) => {
  const m = sentence.match(/^([A-Z][^.:!?]{0,40}):\s+(\S)/);
  if (!m || m[1].split(/\s+/).length > 5) return sentence;
  return m[2].toUpperCase() + sentence.slice(m[0].length);
};

// Sentence ends: punctuation, then space, then a capital, an opening mark or a
// metric name (target_reached). "0.662", "e.g. body" and "5×." stay whole.
export const sentences = (text: string) =>
  text
    .split(/(?<=[.!?])\s+(?=[A-Z“"(]|[a-z]+_[a-z])/)
    .map((s) => s.trim())
    .filter(Boolean);

export function parseCritique(raw: string): ParsedCritique {
  let text = raw.trim();
  const flags: string[] = [];
  for (let changed = true; changed;) {
    changed = false;
    for (const flag of FLAGS) {
      const m = text.match(flag);
      if (m) {
        flags.push(m[0].trim().replace(/\.$/, ""));
        text = text.slice(m[0].length);
        changed = true;
      }
    }
  }

  const blocks: CritiqueBlock[] = [];
  for (const chunk of text.split(/\n\s*\n/)) {
    // The paragraph and the list being built from this chunk's lines.
    const cur: {
      para: string[];
      list: { kind: "ul" | "ol"; items: string[] } | null;
    } = { para: [], list: null };
    const flushPara = () => {
      if (cur.para.length) blocks.push({ kind: "p", text: cur.para.join(" ") });
      cur.para = [];
    };
    const flushList = () => {
      if (cur.list) blocks.push(cur.list);
      cur.list = null;
    };
    for (const line of chunk.split("\n")) {
      const kind = BULLET.test(line) ? "ul" : NUMBERED.test(line) ? "ol" : null;
      if (kind) {
        flushPara();
        if (cur.list?.kind !== kind) {
          flushList();
          cur.list = { kind, items: [] };
        }
        cur.list.items.push(
          line.replace(kind === "ul" ? BULLET : NUMBERED, "").trim(),
        );
      } else if (line.trim()) {
        if (cur.list && /^\s{2,}/.test(line)) {
          // An indented line continues the list item above it.
          cur.list.items[cur.list.items.length - 1] += ` ${line.trim()}`;
        } else {
          flushList();
          cur.para.push(line.trim());
        }
      }
    }
    flushPara();
    flushList();
  }

  // The lead: two sentences of prose, skipping list items and the numbers the critic
  // is told to open with ("Key numbers: …"), which the score row already sums up.
  const paras = blocks.filter((b) => b.kind === "p").map((b) => b.text);
  // A sentence that introduces a list ("…should change:") leads nowhere on its own.
  const usable = (list: string[]) => list.filter((x) => !x.endsWith(":"));
  const first = paras.length ? sentences(paras[0]) : [];
  const rest = usable(paras.slice(1).flatMap(sentences));
  // Skip the numbers paragraph when the rest says enough; else only its first sentence.
  const prose = (
    paras.length && NUMBERS.test(paras[0])
      ? rest.length >= 2
        ? rest
        : [...usable(first.slice(1)), ...rest]
      : [...usable(first), ...rest]
  ).map(withoutLabel);
  let lead = prose.slice(0, 2).join(" ");
  if (lead.length > LEAD_MAX && prose.length) lead = prose[0];
  if (lead.length > LEAD_MAX)
    lead = `${lead.slice(0, LEAD_MAX - 1).replace(/\s+\S*$/, "")}…`;
  const words = text.split(/\s+/).filter(Boolean).length;
  const more =
    blocks.some((b) => b.kind !== "p") ||
    prose.length > 2 ||
    lead.endsWith("…");
  return { flags, lead, blocks, words, more };
}

// Metric and reward-term names (term_bipedal_hold, base_height) as code, with the
// plain name on hover when the board knows it. A short "Key numbers:" opener is bold.
const NAME = /\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b/g;

export function inline(text: string, opener = false): ReactNode {
  const out: ReactNode[] = [];
  let rest = text;
  const head = opener ? rest.match(/^([A-Z][^.:!?]{0,40}):\s/) : null;
  if (head && head[1].split(/\s+/).length <= 5) {
    out.push(<strong key="head">{head[1]}:</strong>, " ");
    rest = rest.slice(head[0].length);
  }
  let last = 0;
  for (const m of rest.matchAll(NAME)) {
    out.push(rest.slice(last, m.index));
    const info = metricInfo(m[0]);
    const plain =
      info.label !== m[0].replaceAll("_", " ") ? info.label : undefined;
    out.push(
      <code key={`${m.index}-${m[0]}`} title={plain}>
        {m[0]}
      </code>,
    );
    last = m.index! + m[0].length;
  }
  out.push(rest.slice(last));
  return out.map((part, i) => <Fragment key={i}>{part}</Fragment>);
}

export default function Critique({ text }: { text: string }) {
  const parsed = parseCritique(text);
  return (
    <section className="critique" aria-label="Claude's notes on this video">
      <p className="critique-who">
        Claude, judging the video
        {parsed.flags.map((flag) => (
          <span className="critique-flag" key={flag}>
            {flag}
          </span>
        ))}
      </p>
      <p className="critique-lead">{inline(parsed.lead)}</p>
      {parsed.more && (
        <details className="critique-full">
          <summary>
            Read Claude's full notes{" "}
            <span className="num">{parsed.words} words</span>
          </summary>
          {parsed.blocks.map((block, i) =>
            block.kind === "p" ? (
              <p key={i}>{inline(block.text, true)}</p>
            ) : block.kind === "ul" ? (
              <ul key={i}>
                {block.items.map((item, j) => (
                  <li key={j}>{inline(item)}</li>
                ))}
              </ul>
            ) : (
              <ol key={i}>
                {block.items.map((item, j) => (
                  <li key={j}>{inline(item)}</li>
                ))}
              </ol>
            ),
          )}
        </details>
      )}
    </section>
  );
}
