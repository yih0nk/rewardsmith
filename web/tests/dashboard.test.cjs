const fs = require("node:fs");
const assert = require("node:assert/strict");
const { test } = require("node:test");
const ts = require("typescript");
const React = require("react");
const { renderToStaticMarkup } = require("react-dom/server");

// Compile the real components with the existing TypeScript dependency.
// Vite supplies import.meta.env; an empty environment represents the live default.
for (const extension of [".ts", ".tsx"]) {
  require.extensions[extension] = (module, filename) => {
    const source = fs
      .readFileSync(filename, "utf8")
      .replaceAll("import.meta", "({ env: {} })");
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
const Detail = require("../src/components/RequestDetail.tsx").default;
const api = require("../src/api.ts");
const health = { ok: true, fake_sim: true, fake_llm: true };
const candidate = {
  id: 1,
  idx: 0,
  reward_src: "def reward(state, action, prev_action): pass",
  diff: null,
  status: "scored",
  error: null,
  metrics: {
    forward_velocity: -0.43,
    term_task: 1.2,
    term_energy: -0.1,
    metadata: { ignored: true },
  },
  video_path: null,
  score: 8.2,
  critique: "Stable backward gait.",
};
const base = {
  id: "abcd1234",
  text: "Walk backwards",
  status: "queued",
  best_score: null,
  created_at: 1,
  rounds: [],
  events: [{ id: 1, kind: "queued", message: "Request queued", created_at: 1 }],
};
const render = (detail) =>
  renderToStaticMarkup(React.createElement(Detail, { detail, health }));

test("queued requests and running rounds with no candidates render safely", () => {
  assert.match(render(base), /Queued · waiting for worker/);
  const html = render({
    ...base,
    status: "running",
    rounds: [
      {
        id: 1,
        n: 1,
        status: "running",
        best_candidate_id: null,
        candidates: [],
      },
    ],
  });
  assert.match(html, /0 candidates/);
  assert.match(html, /No metrics reported yet/);
  assert.doesNotMatch(html, /NaN/);
});

test("all candidates, backend metrics, reward terms and failures are inspectable", () => {
  const failed = {
    ...candidate,
    id: 2,
    idx: 1,
    score: null,
    status: "gate_failed",
    error: "SyntaxError in reward.py",
  };
  const html = render({
    ...base,
    status: "done",
    best_score: 8.2,
    rounds: [
      {
        id: 1,
        n: 1,
        status: "done",
        best_candidate_id: 1,
        candidates: [candidate, failed],
      },
    ],
  });
  assert.match(html, /Candidate 1/);
  assert.match(html, /Candidate 2/);
  assert.match(html, /Validation failed/);
  assert.match(html, /forward velocity/);
  assert.match(html, /-0.43/);
  assert.match(html, /Reward breakdown/);
  assert.match(html, /energy/);
  assert.match(html, /Fake simulator · no video generated/);
  assert.doesNotMatch(
    html,
    /Animated placeholder|backward speed|NaN|\[object Object\]/,
  );
});

test("completed requests default to the overall best round, even if a later round is worse", () => {
  const html = render({
    ...base,
    status: "done",
    best_score: 8.2,
    rounds: [
      {
        id: 1,
        n: 1,
        status: "done",
        best_candidate_id: 1,
        candidates: [candidate],
      },
      {
        id: 2,
        n: 2,
        status: "done",
        best_candidate_id: 3,
        candidates: [
          { ...candidate, id: 3, score: 5, critique: "Worse candidate." },
        ],
      },
    ],
  });
  assert.match(html, /Stable backward gait/);
  assert.doesNotMatch(html, /Worse candidate/);
});

test("failed requests retain worker events and expose gate errors", () => {
  const html = render({
    ...base,
    status: "failed",
    rounds: [
      {
        id: 1,
        n: 1,
        status: "done",
        best_candidate_id: null,
        candidates: [
          {
            ...candidate,
            status: "gate_failed",
            score: null,
            error: "SyntaxError in reward.py",
          },
        ],
      },
    ],
  });
  assert.match(html, /Experiment failed/);
  assert.match(html, /SyntaxError in reward.py/);
  assert.match(html, /Request queued/);
  assert.doesNotMatch(html, /Experiment complete/);
});

test("video URLs normalize worker paths and escape reserved characters", () => {
  assert.equal(
    api.artifactUrl("abc\\r1c0\\roll out#1.mp4"),
    "/artifacts/abc/r1c0/roll%20out%231.mp4",
  );
  const html = render({
    ...base,
    rounds: [
      {
        id: 1,
        n: 1,
        status: "done",
        best_candidate_id: 1,
        candidates: [{ ...candidate, video_path: "abc/r1c0/rollout.mp4" }],
      },
    ],
  });
  assert.match(html, /src="\/artifacts\/abc\/r1c0\/rollout.mp4"/);
});

test("live API is the default and preserves the server request contract", async () => {
  assert.equal(api.MOCK_MODE, false);
  const originalFetch = global.fetch;
  const calls = [];
  global.fetch = async (url, options) => {
    calls.push({ url, options });
    const response =
      url === "/api/health"
        ? health
        : options?.method === "POST"
          ? { id: "new123" }
          : url === "/api/requests"
            ? [base]
            : base;
    return new Response(JSON.stringify(response), { status: 200 });
  };
  try {
    assert.deepEqual(await api.getHealth(), health);
    assert.deepEqual(await api.createRequest("Walk backwards"), {
      id: "new123",
    });
    assert.equal(calls[1].url, "/api/requests");
    assert.deepEqual(JSON.parse(calls[1].options.body), {
      text: "Walk backwards",
    });
    assert.deepEqual(await api.listRequests(), [base]);
    assert.deepEqual(await api.getRequest(base.id), base);
    assert.equal(calls[3].url, `/api/requests/${base.id}`);
    global.fetch = async () =>
      new Response(JSON.stringify({ detail: "No request with that id" }), {
        status: 404,
      });
    await assert.rejects(
      api.getRequest("missing"),
      (error) =>
        error instanceof api.ApiError &&
        error.status === 404 &&
        /No request/.test(error.message),
    );
  } finally {
    global.fetch = originalFetch;
  }
});
