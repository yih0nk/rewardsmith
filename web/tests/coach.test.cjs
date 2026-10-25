const fs = require("node:fs");
const assert = require("node:assert/strict");
const { test } = require("node:test");
const ts = require("typescript");
const React = require("react");
const { renderToStaticMarkup } = require("react-dom/server");

for (const extension of [".ts", ".tsx"]) {
  require.extensions[extension] = (module, filename) => {
    const source = fs.readFileSync(filename, "utf8");
    const result = ts.transpileModule(source, {
      compilerOptions: {
        module: ts.ModuleKind.CommonJS,
        target: ts.ScriptTarget.ES2022,
        jsx: ts.JsxEmit.ReactJSX,
        esModuleInterop: true,
      },
    });
    module._compile(result.outputText, filename);
  };
}
const CoachFeed = require("../src/components/CoachFeed.tsx");

const event = (id, message, round_n = 1, candidate = null, kind = "coach") => ({
  id,
  kind,
  message,
  created_at: 1000 + id,
  round_n,
  candidate,
});

test("the coach feed renders nothing before the coach speaks", () => {
  const html = renderToStaticMarkup(
    React.createElement(CoachFeed.default, {
      events: [event(1, "Round 1 started", null, null, "round")],
    }),
  );
  assert.equal(html, "");
});

test("coach notes carry a round and candidate tag instead of the prefix", () => {
  const html = renderToStaticMarkup(
    React.createElement(CoachFeed.default, {
      events: [
        event(1, "Training 4 candidate(s)", null, null, "train"),
        event(
          2,
          "Candidate 3 at 5.2M steps: posture is 500x the task term. Stopping it.",
          1,
          2,
        ),
        event(3, "Coach: 9 looks, median 2.1 s, 1 stopped, $0.0040"),
      ],
    }),
  );
  assert.match(html, /R1 C3/);
  assert.match(html, /5\.2M steps: posture is 500x the task term\. Stopping it\./);
  assert.doesNotMatch(html, /Candidate 3 at/);
  assert.match(html, /class="coach-summary"/);
  assert.doesNotMatch(html, /Training 4/);
});

test("older notes fold away behind a count", () => {
  const events = Array.from({ length: 9 }, (_, i) =>
    event(i + 1, `Candidate 1 at ${i}.0M steps: note ${i}`, 1, 0),
  );
  const html = renderToStaticMarkup(
    React.createElement(CoachFeed.default, { events }),
  );
  assert.match(html, /<summary>3 earlier notes<\/summary>/);
  assert.equal(html.match(/<li/g).length, 9);
});

test("tags without a round still name the candidate", () => {
  assert.equal(CoachFeed.who(event(1, "x", null, 0)), "C1");
  assert.equal(CoachFeed.who(event(1, "x", 2, null)), "R2");
});
