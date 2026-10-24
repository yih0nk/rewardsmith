const fs = require("node:fs");
const assert = require("node:assert/strict");
const { test } = require("node:test");
const ts = require("typescript");
const React = require("react");
const { renderToStaticMarkup } = require("react-dom/server");

// Compile the real components with the existing TypeScript dependency.
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

test("API calls preserve the server request contract", async () => {
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

const { HallCard } = require("../src/components/HallOfFame.tsx");
const scoredRound = (id, n, c) => ({
  id,
  n,
  status: "done",
  best_candidate_id: c.id,
  candidates: [c],
});

test("revision diffs color added, removed and hunk lines", () => {
  const diff =
    "--- reward.py (before)\n+++ reward.py (after)\n@@ -1 +1 @@\n-old = 1\n+new = 2\n";
  const html = render({
    ...base,
    status: "done",
    best_score: 8.2,
    rounds: [scoredRound(1, 1, { ...candidate, diff })],
  });
  assert.match(html, /class="diff-add">\+new = 2/);
  assert.match(html, /class="diff-del">-old = 1/);
  assert.match(html, /class="diff-hunk">@@/);
  assert.match(html, /class="diff-file">\+\+\+ reward.py/);
});

test("scored candidates can be flagged; flagged ones show their note", () => {
  const detail = (c) => ({
    ...base,
    status: "done",
    best_score: 8.2,
    rounds: [scoredRound(1, 1, c)],
  });
  assert.match(render(detail(candidate)), /Flag as reward hacking/);
  const flagged = render(
    detail({ ...candidate, flagged_at: 5, flag_note: "Vibrates in place" }),
  );
  assert.match(flagged, /IN THE HALL OF FAME/);
  assert.match(flagged, /Vibrates in place/);
  assert.doesNotMatch(flagged, /Flag as reward hacking/);
  const gateFailed = render(
    detail({ ...candidate, status: "gate_failed", score: null }),
  );
  assert.doesNotMatch(gateFailed, /Flag as reward hacking/);
});

test("the round gallery shows each round's winning video once two exist", () => {
  const withVideo = (id, n) =>
    scoredRound(id, n, { ...candidate, id, video_path: `abc/r${n}c0/rollout.mp4` });
  const one = render({ ...base, status: "done", best_score: 8.2, rounds: [withVideo(1, 1)] });
  assert.doesNotMatch(one, /Every round, side by side/);
  const two = render({
    ...base,
    status: "done",
    best_score: 8.2,
    rounds: [withVideo(1, 1), withVideo(2, 2)],
  });
  assert.match(two, /Every round, side by side/);
  assert.match(two, /src="\/artifacts\/abc\/r2c0\/rollout.mp4"/);
});

test("hall of fame entries show the run, the note, the video and the terms", () => {
  const entry = {
    ...candidate,
    video_path: "abc/r2c1/rollout.mp4",
    flagged_at: 5,
    flag_note: "Scores by vibrating",
    round_n: 2,
    request_id: "abcd1234",
    request_text: "Spin in place",
  };
  const html = renderToStaticMarkup(
    React.createElement(HallCard, { entry, onOpen: () => {} }),
  );
  assert.match(html, /Spin in place/);
  assert.match(html, /ROUND 2 · CANDIDATE 1/);
  assert.match(html, /Scores by vibrating/);
  assert.match(html, /src="\/artifacts\/abc\/r2c1\/rollout.mp4"/);
  assert.match(html, /Reward breakdown/);
});

test("hall of fame API calls match the server routes", async () => {
  const originalFetch = global.fetch;
  const calls = [];
  global.fetch = async (url, options) => {
    calls.push({ url, method: options?.method ?? "GET", body: options?.body });
    return new Response(JSON.stringify(url === "/api/hall-of-fame" ? [] : { ok: true }), {
      status: 200,
    });
  };
  try {
    assert.deepEqual(await api.listHallOfFame(), []);
    assert.deepEqual(await api.flagCandidate(7, "Games the velocity term"), { ok: true });
    assert.deepEqual(await api.unflagCandidate(7), { ok: true });
    assert.deepEqual(
      calls.map((c) => `${c.method} ${c.url}`),
      ["GET /api/hall-of-fame", "POST /api/candidates/7/flag", "DELETE /api/candidates/7/flag"],
    );
    assert.deepEqual(JSON.parse(calls[1].body), { note: "Games the velocity term" });
  } finally {
    global.fetch = originalFetch;
  }
});
