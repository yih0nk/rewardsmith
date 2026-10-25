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
  assert.match(render(base), /Waiting in line for the GPU worker/);
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
  assert.match(html, /Claude is writing reward functions/);
  assert.match(html, /0 rewards in 1 round/);
  // Nothing to show yet: the status takes the whole width, with no empty video box.
  assert.match(html, /class="run-grid status-only"/);
  assert.doesNotMatch(html, /aria-label="Robot rollout"/);
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
  assert.match(html, /Round 1 · Candidate 1/);
  assert.match(html, /Round 1 · Candidate 2/);
  assert.match(html, /Code never ran/);
  assert.match(html, /failed the code check 3 times/);
  assert.match(html, /Forward speed/);
  assert.match(html, /-0.43 m\/s/);
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
  assert.match(html, /The run stopped with an error/);
  assert.match(html, /The run stopped before it produced a robot/);
  assert.match(html, /SyntaxError in reward.py/);
  assert.match(html, /Request queued/);
  assert.doesNotMatch(html, /Finished/);
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

const progress = require("../src/components/TrainingProgress.tsx");
const point = (steps, episode_reward) => ({
  steps,
  wall_s: steps / 1000,
  episode_reward,
  terms: { task: episode_reward == null ? null : episode_reward / 10 },
});
const live = (stage, fraction, curve = []) => ({
  stage,
  steps: Math.round(fraction * 1000),
  total_steps: 1000,
  fraction,
  wall_s: 30,
  curve,
});
const training = (overrides) => ({
  ...candidate,
  status: "training",
  score: null,
  critique: null,
  metrics: null,
  progress: null,
  ...overrides,
});

test("stage labels follow the candidate status and its progress file", () => {
  const label = (c) => progress.liveState(c)?.label ?? null;
  assert.equal(label(training({ status: "written" })), "Waiting to train");
  assert.equal(label(training()), "Starting…");
  assert.equal(label(training({ progress: live("compiling", 0) })), "Compiling");
  assert.equal(label(training({ progress: live("training", 0) })), "Training 0%");
  assert.equal(label(training({ progress: live("training", 0.428) })), "Training 42%");
  // Floor: 99.6% done is not 100% done.
  assert.equal(label(training({ progress: live("training", 0.996) })), "Training 99%");
  assert.equal(label(training({ progress: live("rendering", 1) })), "Recording video");
  assert.equal(label(training({ progress: live("done", 1) })), "Finishing up");
  assert.equal(label(training({ status: "trained", progress: null })), "Judging");
  assert.equal(label(training({ status: "trained", progress: live("done", 1) })), "Judging");
  // Finished candidates have no live state, whatever a stale progress file says.
  assert.equal(label({ ...candidate, progress: live("training", 0.5) }), null);
  assert.equal(label(training({ status: "failed", progress: live("training", 0.5) })), null);
  assert.equal(label(training({ status: "gate_failed" })), null);
});

test("progress bar fractions are clamped and never NaN", () => {
  const frac = (p, status = "training") => progress.liveState(training({ status, progress: p })).fraction;
  assert.equal(frac(null), 0);
  assert.equal(frac(null, "written"), null);
  assert.equal(frac(live("training", 1.7)), 1);
  assert.equal(frac(live("training", -0.2)), 0);
  assert.equal(frac({ ...live("training", 0), fraction: null }), 0);
  assert.equal(frac(live("rendering", 0.9)), 1);
});

test("the sparkline skips null rewards and survives empty or flat curves", () => {
  assert.equal(progress.sparkline(null, 100, 20), null);
  assert.equal(progress.sparkline([], 100, 20), null);
  assert.equal(progress.sparkline([point(0, null), point(10, null)], 100, 20), null);
  const shape = progress.sparkline(
    [point(0, 1), point(500, null), point(1000, 3), point(2000, 2)],
    100,
    20,
    0,
  );
  assert.equal(shape.points.length, 3);
  assert.equal(shape.latest, 2);
  assert.deepEqual(shape.points[0], { x: 0, y: 20 });
  assert.deepEqual(shape.points[1], { x: 50, y: 0 });
  assert.deepEqual(shape.points[2], { x: 100, y: 10 });
  // One point, or all-equal rewards, sit in the middle instead of dividing by zero.
  assert.deepEqual(progress.sparkline([point(0, 5)], 100, 20, 0).points, [{ x: 50, y: 10 }]);
  const flat = progress.sparkline([point(0, 5), point(10, 5)], 100, 20, 0);
  assert.ok(flat.points.every((p) => p.y === 10 && Number.isFinite(p.x)));
});

test("a training candidate shows its stage, an accessible bar and the reward curve", () => {
  const curve = [point(0, 0.5), point(400, null), point(800, 12.345)];
  const detail = (c) => ({
    ...base,
    status: "running",
    rounds: [{ id: 1, n: 1, status: "running", best_candidate_id: null, candidates: [c] }],
  });
  const starting = render(detail(training()));
  assert.match(starting, /Starting…/);
  assert.match(starting, /role="progressbar"[^>]*aria-valuenow="0"/);
  assert.doesNotMatch(starting, /<polyline/);

  const html = render(detail(training({ progress: live("training", 0.42, curve) })));
  assert.match(html, /Training 42%/);
  assert.match(html, /LIVE/);
  assert.match(html, /role="progressbar"/);
  assert.match(html, /aria-valuenow="42"/);
  assert.match(html, /aria-valuetext="Training 42%"/);
  assert.match(html, /<polyline points="/);
  assert.match(html, /latest 12.35/);
  assert.match(html, /420 \/ 1,000 training steps/);
  // The training process's own clock stands still while it compiles: not shown.
  assert.doesNotMatch(html, /steps · \d+ s/);
  assert.doesNotMatch(html, /NaN|undefined/);

  const finishing = render(detail(training({ progress: live("done", 1, curve) })));
  assert.match(finishing, /Finishing up/);
  assert.match(finishing, /aria-valuenow="100"/);
});

test("scored candidates keep a small learning curve and drop the bar", () => {
  const curve = [point(0, 0.5), point(1000, 9)];
  const html = render({
    ...base,
    status: "done",
    best_score: 8.2,
    rounds: [scoredRound(1, 1, { ...candidate, progress: live("done", 1, curve) })],
  });
  // The score and its "/ 10" are styled apart, so markup sits between them.
  assert.match(html, />8\.2<small> \/ 10</);
  assert.match(html, /final reward/);
  assert.match(html, /<polyline/);
  assert.doesNotMatch(html, /role="progressbar"|Judging|LIVE/);
});

const timeline = require("../src/components/RoundTimeline.tsx");

test("rounds report their winner's score and the change from the last scored round", () => {
  const round = (id, n, scores, bestIdx) => ({
    id,
    n,
    status: "done",
    best_candidate_id: bestIdx == null ? null : id * 10 + bestIdx,
    candidates: scores.map((score, i) => ({ ...candidate, id: id * 10 + i, idx: i, score })),
  });
  const rounds = [
    round(1, 1, [4.2, 3.1], 0),
    round(2, 2, [null, null], null),
    round(3, 3, [6.1, 7.4], 1),
    round(4, 4, [5, 6.5], 0),
  ];
  // No winner yet: fall back to the best scored candidate; none at all: null.
  assert.equal(timeline.roundScore(round(9, 9, [2, 5], null)), 5);
  assert.equal(timeline.roundScore(rounds[1]), null);
  const deltas = timeline.roundDeltas(rounds).map((r) => r.delta);
  assert.equal(deltas[0], null);
  assert.equal(deltas[1], null);
  // Round 3 compares with round 1, skipping the unscored round 2.
  assert.ok(Math.abs(deltas[2] - 3.2) < 1e-9);
  assert.ok(Math.abs(deltas[3] + 2.4) < 1e-9);
  assert.equal(timeline.formatDelta(3.2), "+3.2");
  assert.equal(timeline.formatDelta(-2.4), "−2.4");
  assert.equal(timeline.formatDelta(0), "±0.0");
});

test("the run summary and round timeline show the improvement since round 1", () => {
  const html = render({
    ...base,
    status: "done",
    best_score: 8.2,
    rounds: [
      scoredRound(1, 1, { ...candidate, id: 1, score: 5 }),
      scoredRound(2, 2, { ...candidate, id: 2, score: 8.2 }),
    ],
  });
  assert.match(html, /Since round 1/);
  assert.match(html, /\+3\.2/);
  assert.match(html, /5\.0 → 8\.2/);
  assert.match(html, /Round 2 · Candidate 1/);
  assert.match(html, /class="round-best">Best</);
  assert.doesNotMatch(html, /NaN|undefined/);
});

test("the stage draws the full reward curve, or says why there is none", () => {
  const curve = [point(0, 1), point(500000, null), point(1000000, 4)];
  const full = renderToStaticMarkup(
    React.createElement(progress.RewardCurve, { curve }),
  );
  assert.match(full, /<path class="curve-line" d="M/);
  assert.match(full, /latest 4/);
  assert.match(full, />1M</);
  assert.match(full, /final/);
  const empty = (live) =>
    renderToStaticMarkup(React.createElement(progress.RewardCurve, { curve: [], live }));
  assert.match(empty(true), /First evaluation in a few seconds/);
  assert.match(empty(false), /No learning curve recorded/);
  assert.equal(progress.formatSteps(26214400), "26.2M");
  assert.equal(progress.formatSteps(500), "500");
});

test("mid-round, the stage shows the candidate furthest into training", () => {
  const html = render({
    ...base,
    status: "running",
    rounds: [
      {
        id: 1,
        n: 1,
        status: "running",
        best_candidate_id: null,
        candidates: [
          training({ id: 1, idx: 0, progress: live("training", 0.3) }),
          training({ id: 2, idx: 1, progress: live("training", 0.6) }),
          training({ id: 3, idx: 2, status: "written" }),
          // Judging is further along, but a candidate still training is more to watch.
          training({ id: 4, idx: 3, status: "trained", progress: live("done", 1) }),
        ],
      },
    ],
  });
  assert.match(html, /class="stage-title">Round 1 · Candidate 2</);
  assert.match(html, /class="training-stage">Training 60%/);
});

test("bars and strips are drawn in characters from the real numbers", () => {
  assert.equal(progress.textBar(0.42, 10), "[####......]");
  assert.equal(progress.textBar(1.7, 4), "[####]");
  assert.equal(progress.textBar(Number.NaN, 4), "[....]");
  const r = {
    id: 1,
    n: 1,
    status: "done",
    best_candidate_id: 2,
    candidates: [
      { ...candidate, id: 1, score: 0 },
      { ...candidate, id: 2, score: 10 },
      { ...candidate, id: 3, score: 5 },
      { ...candidate, id: 4, score: null },
    ],
  };
  // 0–10 in half-point cells: candidates are "o", the winner "O", unscored ones are skipped.
  assert.equal(timeline.scoreStrip(r, 21).join(""), "o·········o·········O");
  const html = render({
    ...base,
    status: "running",
    rounds: [
      {
        id: 1,
        n: 1,
        status: "running",
        best_candidate_id: null,
        candidates: [training({ progress: live("training", 0.5) })],
      },
    ],
  });
  assert.match(html, /role="progressbar"[^>]*><span aria-hidden="true">\[#{8}\.{8}\]/);
});

const stages = require("../src/stages.ts");
const explain = require("../src/explain.tsx");
const RunStatus = require("../src/components/RunStatus.tsx").default;

// A running request at server time T0, with helpers to build each stage.
const T0 = 1_800_000_000;
const runningAt = (rounds, events = []) => ({
  ...base,
  status: "running",
  created_at: T0,
  events: [
    { id: 1, kind: "queued", message: "Request queued", created_at: T0 },
    ...events,
  ],
  rounds,
});
const roundAt = (n, created_at, candidates, status = "running") => ({
  id: n,
  n,
  status,
  best_candidate_id: null,
  created_at,
  candidates,
});
const cand = (idx, overrides) => ({
  ...training({ id: 100 + idx, idx, created_at: T0 + 70 + idx }),
  ...overrides,
});
const curveAt = (wall) => [{ steps: 0, wall_s: wall, episode_reward: 0.1, terms: {} }];
const trainEvt = { id: 9, kind: "train", message: "Training 4 candidate(s)", created_at: T0 + 100 };
const sceneEvt = { id: 2, kind: "scene", message: "flat floor", created_at: T0 + 5 };

test("each pipeline stage gets a plain-language label and a start time", () => {
  const stage = (detail, k) => stages.runStage(detail, k);
  const queued = stage({ ...base, created_at: T0 });
  assert.equal(queued.label, "Waiting in line for the GPU worker");
  assert.equal(queued.startedAt, T0);

  assert.equal(stage(runningAt([])).label, "Claude is reading your prompt and setting up the scene");
  assert.equal(stage(runningAt([], [sceneEvt])).label, "Claude is writing the pass/fail check");

  const writing = stage(runningAt([roundAt(1, T0 + 10, [])]), 4);
  assert.equal(writing.label, "Claude is writing 4 reward functions");
  assert.equal(writing.startedAt, T0 + 10);
  // Without the per-round count from the API, it doesn't make one up.
  assert.equal(stage(runningAt([roundAt(1, T0 + 10, [])])).label, "Claude is writing reward functions");
  const done1 = roundAt(1, T0 + 10, [0, 1, 2, 3].map((i) => cand(i, { status: "scored", score: 5 })), "done");
  assert.equal(
    stage(runningAt([done1, roundAt(2, T0 + 300, [])])).label,
    "Claude is revising the best rewards into 4 new ones",
  );

  const gate = stage(runningAt([roundAt(1, T0 + 10, [cand(0, { status: "written" }), cand(1, { status: "gate_failed" })])]));
  assert.equal(gate.key, "gate");
  assert.equal(gate.label, "Test-running the code of 2 rewards");
  assert.equal(gate.startedAt, T0 + 70);

  const four = (progress) => [0, 1, 2, 3].map((i) => cand(i, { progress }));
  const compiling = stage(runningAt([roundAt(1, T0 + 10, four(null))], [trainEvt]));
  assert.equal(compiling.label, "Compiling the simulator for 4 robots");
  assert.match(compiling.hint, /1–2 min/);
  assert.equal(compiling.startedAt, T0 + 100);

  const trainingStage = stage(
    runningAt([roundAt(1, T0 + 10, four({ ...live("training", 0.4), curve: curveAt(55) }))], [trainEvt]),
  );
  assert.equal(trainingStage.label, "Training 4 robots in parallel");
  // The train event plus the training process's own clock at its first evaluation.
  assert.equal(trainingStage.startedAt, T0 + 155);

  const rendering = stage(runningAt([roundAt(1, T0 + 10, four(live("rendering", 1)))], [trainEvt]));
  assert.equal(rendering.label, "Recording a video of each trained robot");

  const judging = stage(
    runningAt(
      [roundAt(1, T0 + 10, [cand(0, { status: "trained", progress: { ...live("done", 1), wall_s: 120 } }), cand(1, { status: "trained" })])],
      [trainEvt],
    ),
  );
  assert.equal(judging.label, "Claude is watching 2 videos and scoring each one");
  assert.equal(judging.startedAt, T0 + 220);
});

test("finished runs say whether anything worked; failures never look in progress", () => {
  const scored = { ...base, status: "done", best_score: 6.8, rounds: [scoredRound(1, 1, candidate)] };
  assert.equal(stages.runStage(scored).label, "Finished after 1 round · best score 6.8 / 10");

  const crashed = (i) => ({ ...candidate, id: i, idx: i, status: "failed", score: null, error: "TypeError: mul got incompatible shapes" });
  const nothing = {
    ...base,
    status: "done",
    rounds: [{ id: 1, n: 1, status: "done", best_candidate_id: null, candidates: [crashed(1), crashed(2)] }],
  };
  assert.equal(stages.runStage(nothing).key, "no_results");
  const html = render(nothing);
  assert.match(html, /None of the 2 rewards produced a robot Claude could judge/);
  assert.match(html, /2 × crashed while training/);
  assert.match(html, /What to try/);
  assert.match(html, /run-status-panel broken/);
  assert.doesNotMatch(html, />NOW</);
  assert.doesNotMatch(html, /class="steps"/);

  const running = render(runningAt([roundAt(1, T0 + 10, [])]));
  assert.match(running, />NOW</);
  assert.doesNotMatch(running, /broken|None of the/);
});

test("candidates are named the same way everywhere, with failures in plain words", () => {
  assert.equal(stages.candidateName(2, 2), "Round 2 · Candidate 3");
  const state = (c) => stages.candidateState({ ...candidate, score: null, ...c }).text;
  assert.equal(state({ status: "gate_failed" }), "Code never ran");
  assert.equal(state({ status: "stopped" }), "Stopped early");
  assert.equal(state({ status: "failed", error: "The critic could not score this run: x" }), "Couldn't be judged");
  assert.equal(state({ status: "failed", error: "Stopped: the run failed before this candidate finished." }), "Cut short");
  assert.equal(state({ status: "trained" }), "Being judged by Claude");
  assert.equal(stages.candidateState(candidate).text, "Scored 8.2 / 10");
});

test("elapsed time is formatted with units and never negative", () => {
  assert.equal(stages.formatElapsed(0), "0s");
  assert.equal(stages.formatElapsed(7.9), "7s");
  assert.equal(stages.formatElapsed(65), "1m 05s");
  assert.equal(stages.formatElapsed(3725), "1h 02m");
  assert.equal(stages.formatElapsed(-3), "0s");
  assert.equal(stages.formatElapsed(Number.NaN), "0s");
  assert.equal(stages.elapsedSince(T0, (T0 + 75) * 1000), 75);
  // The server clock runs 30 s ahead of this one.
  assert.equal(stages.elapsedSince(T0, (T0 + 75) * 1000, 30), 105);
  assert.equal(stages.elapsedSince(T0 + 10, T0 * 1000), 0);
});

test("the stage and run timers tick from their start times on the client clock", () => {
  const detail = runningAt(
    [roundAt(1, T0 + 10, [0, 1, 2, 3].map((i) => cand(i, { progress: null })))],
    [trainEvt],
  );
  const at = (seconds) =>
    renderToStaticMarkup(React.createElement(RunStatus, { detail, now: (T0 + seconds) * 1000 }));
  const timers = (html) => [
    html.match(/data-testid="stage-timer">([^<]+)</)[1],
    html.match(/data-testid="run-timer">([^<]+)</)[1],
  ];
  // Compiling started with the train event at T0 + 100.
  assert.deepEqual(timers(at(175)), ["1m 15s", "2m 55s"]);
  assert.deepEqual(timers(at(180)), ["1m 20s", "3m 00s"]);
  const html = at(175);
  assert.match(html, /role="status">Compiling the simulator for 4 robots</);
  assert.match(html, /class="step current" aria-current="step">[\s\S]*?Compile/);
  assert.equal((html.match(/class="step done"/g) ?? []).length, 3);
  // One lane per candidate, named like everywhere else.
  assert.match(html, /Round 1 · Candidate 4/);

  // The server's Date header corrects a drifting client clock.
  api.noteServerDate(new Date((T0 + 235) * 1000).toUTCString(), (T0 + 175) * 1000);
  try {
    assert.deepEqual(timers(at(175)), ["2m 15s", "3m 55s"]);
  } finally {
    api.noteServerDate(null);
    api.noteServerDate(new Date().toUTCString());
  }
  assert.equal(api.serverClockOffset(), 0);

  // A finished run shows how long it took, from its first to its last event.
  const finished = renderToStaticMarkup(
    React.createElement(RunStatus, {
      detail: { ...detail, status: "done", events: [...detail.events, { id: 20, kind: "done", message: "Finished", created_at: T0 + 600 }] },
      now: (T0 + 9999) * 1000,
    }),
  );
  assert.match(finished, /Took<\/dt><dd class="num" data-testid="run-timer">10m 00s/);
});

test("the success check reads as a sentence and jargon comes with an explanation", () => {
  assert.equal(
    explain.successSentence([
      { metric: "forward_velocity", op: "<", value: -0.3 },
      { metric: "fell", op: "<", value: 0.5 },
    ]),
    "Pass = the robot walks backwards faster than 0.3 m/s and doesn't fall over.",
  );
  assert.equal(
    explain.successSentence([
      { metric: "abs_yaw_rate", op: ">", value: 1 },
      { metric: "abs_forward_velocity", op: "<", value: 0.3 },
      { metric: "base_height", op: ">", value: 0.18 },
    ]),
    "Pass = the robot turns either way faster than 1 rad/s (57°/s), moves forwards or backwards slower than 0.3 m/s and keeps its body above 0.18 m.",
  );
  assert.equal(explain.successSentence([]), null);
  assert.equal(explain.metricInfo("fell").format(1), "yes");
  assert.equal(explain.metricInfo("target_reached").format(0.41), "41%");

  const html = render({
    ...base,
    status: "done",
    best_score: 8.2,
    success_spec: [{ metric: "forward_velocity", op: "<", value: -0.3 }],
    rounds: [scoredRound(1, 1, candidate)],
  });
  assert.match(html, /Pass = the robot walks backwards faster than 0.3 m\/s\./);
  for (const term of ["critic", "successCheck", "candidate"])
    assert.ok(html.includes(explain.GLOSSARY[term].replaceAll("'", "&#x27;")), term);
  assert.match(html, /role="tooltip"/);
});

const thread = require("../src/thread.ts");

test("each request is one conversation, newest first, with one turn", () => {
  const reqs = [
    { id: "a", text: "walk backwards", status: "done", best_score: 6.8, created_at: 10 },
    { id: "b", text: "spin in place", status: "running", best_score: null, created_at: 20 },
  ];
  const convos = thread.conversationsFromRequests(reqs);
  assert.deepEqual(convos.map((c) => c.id), ["b", "a"]);
  assert.deepEqual(convos[1].turns, [
    { requestId: "a", text: "walk backwards", createdAt: 10, turn: 1, status: "done" },
  ]);
  assert.equal(convos[1].bestScore, 6.8);
  assert.equal(convos[0].bestScore, null);
});

test("a queued task says which task it waits for", () => {
  const reqs = [
    { id: "r", text: "push the ball into the blue zone", status: "running", created_at: 1 },
    { id: "q1", text: "spin in place", status: "queued", created_at: 2 },
    { id: "q2", text: "walk backwards", status: "queued", created_at: 3 },
    { id: "d", text: "old", status: "done", created_at: 0 },
  ];
  const first = thread.queuePlace(reqs, "q1");
  assert.deepEqual(first, { running: "push the ball into the blue zone", ahead: [] });
  assert.deepEqual(thread.queuePlace(reqs, "q2").ahead, ["spin in place"]);
  assert.equal(thread.queuePlace(reqs, "r"), null);

  const queued = { ...base, status: "queued", created_at: T0 };
  assert.equal(
    stages.runStage(queued, null, first).label,
    "Queued, starts when “push the ball into the blue zone” finishes",
  );
  const second = stages.runStage(queued, null, thread.queuePlace(reqs, "q2"));
  assert.equal(second.label, "Queued behind 2 tasks");
  assert.match(second.hint, /“push the ball into the blue zone” is running, then “spin in place” go first/);
  assert.equal(
    stages.runStage(queued, null, { running: null, ahead: [] }).label,
    "Queued, starting in a few seconds",
  );
  // The turn shows it, so a waiting task never looks stuck.
  const html = renderToStaticMarkup(
    React.createElement(Detail, { detail: queued, health, queue: first }),
  );
  assert.match(html, /role="status">Queued, starts when “push the ball into the blue zone” finishes</);
});

test("runs and candidates have stable anchors, and reply slots stay empty until wired", () => {
  const detail = {
    ...base,
    status: "done",
    best_score: 8.2,
    rounds: [scoredRound(1, 1, { ...candidate, id: 7 })],
  };
  const plain = render(detail);
  assert.match(plain, /<article class="cmd turn(?: [^"]+)?" id="run-abcd1234"/);
  assert.match(plain, /id="candidate-7" data-round="1" data-candidate-id="7"/);
  assert.match(plain, /aria-label="Robot rollout" data-candidate-id="7"/);
  assert.doesNotMatch(plain, /reply-slot/);

  const targets = [];
  const wired = renderToStaticMarkup(
    React.createElement(Detail, {
      detail,
      health,
      renderReply: (target) => {
        targets.push(target);
        return React.createElement("span", null, "reply");
      },
    }),
  );
  assert.equal((wired.match(/class="reply-slot"/g) ?? []).length, 3);
  assert.deepEqual(targets, [
    { kind: "candidate", requestId: "abcd1234", round: 1, candidateId: 7, idx: 0, scored: true },
    { kind: "candidate", requestId: "abcd1234", round: 1, candidateId: 7, idx: 0, scored: true },
    { kind: "run", requestId: "abcd1234" },
  ]);
});

test("the task list shows status and best score, and marks the open task", () => {
  const List = require("../src/components/ConversationList.tsx").default;
  const convos = thread.conversationsFromRequests([
    { id: "a", text: "walk backwards", status: "done", best_score: 6.8, created_at: 1 },
    { id: "b", text: "kick the ball", status: "done", best_score: null, created_at: 2 },
    { id: "c", text: "spin", status: "queued", best_score: null, created_at: 3 },
  ]);
  const html = renderToStaticMarkup(
    React.createElement(List, { conversations: convos, selected: "a", loading: false, onSelect: () => {} }),
  );
  assert.match(html, /aria-current="page"[^>]*>[\s\S]*?Walk backwards[\s\S]*?best <\/span>6\.8/);
  assert.match(html, /Kick the ball[\s\S]*?no result/);
  assert.match(html, /Spin[\s\S]*?queued/);
  // Cancelled tasks fold behind a toggle unless they are the open one.
  const stopped = thread.conversationsFromRequests([
    { id: "d", text: "walk", status: "done", best_score: 8, created_at: 1 },
    { id: "e", text: "spin (cancel test)", status: "cancelled", best_score: null, created_at: 2 },
  ]);
  const list = (selected) =>
    renderToStaticMarkup(React.createElement(List, { conversations: stopped, selected, loading: false, onSelect: () => {} }));
  assert.doesNotMatch(list("d"), /cancel test/);
  assert.match(list("d"), /Show 1 cancelled or failed/);
  assert.match(list("e"), /cancel test/);
  const empty = (loading) =>
    renderToStaticMarkup(React.createElement(List, { conversations: [], selected: null, loading, onSelect: () => {} }));
  assert.match(empty(true), /Loading tasks…/);
  assert.match(empty(false), /No tasks yet/);
});

test("the composer has no option controls until they are passed in", () => {
  const Composer = require("../src/components/Composer.tsx").default;
  const props = { label: "What should the robot learn?", inputRef: { current: null }, busy: false, error: "", text: "", onText: () => {}, onSubmit: () => {} };
  const html = renderToStaticMarkup(React.createElement(Composer, props));
  assert.match(html, /<label for="behavior"[^>]*>What should the robot learn\?/);
  assert.match(html, /Start training/);
  assert.match(html, /walk sideways/);
  assert.doesNotMatch(html, /composer-controls/);
  const withControls = renderToStaticMarkup(
    React.createElement(Composer, { ...props, controls: React.createElement("span", null, "Model") }),
  );
  assert.match(withControls, /class="composer-controls"><span>Model/);
});

test("open server-timed stages set the start time; training keeps candidate progress", () => {
  const writingStage = { name: "writing", round_n: 1, started_at: T0 + 12, ended_at: null };
  const writing = { ...runningAt([roundAt(1, T0 + 10, [])]), stages: [writingStage] };
  assert.equal(stages.runStage(writing, 4).startedAt, T0 + 12);
  // A finished stage, or one from another round, doesn't count.
  const old = { ...writing, stages: [{ ...writingStage, round_n: 0 }, { ...writingStage, ended_at: T0 + 20 }] };
  assert.equal(stages.runStage(old, 4).startedAt, T0 + 10);
  const setup = { ...runningAt([], [sceneEvt]), stages: [{ name: "check", round_n: null, started_at: T0 + 6, ended_at: null }] };
  assert.equal(stages.runStage(setup).startedAt, T0 + 6);
});

test("run controls offer the actions each state allows", () => {
  const html = (fields) => render({ ...base, ...fields });
  const queued = html({ status: "queued" });
  assert.match(queued, />Pause</);
  assert.match(queued, />Cancel</);

  const running = html({ status: "running", control: null });
  assert.match(running, /Pause after this round/);
  assert.match(running, />Cancel</);

  const pausing = html({ status: "running", control: "pause" });
  assert.match(pausing, /Pausing after this round/);
  assert.match(pausing, /Keep running/);

  const cancelling = html({ status: "running", control: "cancel" });
  assert.match(cancelling, /Cancelling at the next checkpoint/);
  assert.doesNotMatch(cancelling, />Cancel</);

  const paused = html({ status: "paused" });
  assert.match(paused, />Resume</);
  assert.match(paused, /Paused before it started/);

  for (const status of ["done", "failed", "cancelled"]) {
    assert.doesNotMatch(html({ status }), /run-controls/);
  }
  assert.match(html({ status: "cancelled" }), /Cancelled before it started/);
});

test("run control calls post to the request's action route", async () => {
  const originalFetch = global.fetch;
  const calls = [];
  global.fetch = async (url, options) => {
    calls.push(`${options?.method} ${url}`);
    return new Response(JSON.stringify({ state: "pausing" }), { status: 200 });
  };
  try {
    assert.deepEqual(await api.controlRequest("abcd1234", "pause"), { state: "pausing" });
    await api.controlRequest("abcd1234", "resume");
    await api.controlRequest("abcd1234", "cancel");
    assert.deepEqual(calls, [
      "POST /api/requests/abcd1234/pause",
      "POST /api/requests/abcd1234/resume",
      "POST /api/requests/abcd1234/cancel",
    ]);
  } finally {
    global.fetch = originalFetch;
  }
});

// Real critiques from the live board: several hundred words with numbers, lists and
// term names. The fake LLM's one-line critiques hid how this text looked in a column.
const realCritiques = JSON.parse(fs.readFileSync(`${__dirname}/fixtures/real-critiques.json`, "utf8"));
const critique = require("../src/components/Critique.tsx");

test("a real critique leads with what happened and keeps its lists and numbers", () => {
  const parsed = critique.parseCritique(realCritiques.best.critique);
  assert.deepEqual(parsed.flags, []);
  // The numbers the critic opens with are skipped; the lead says what the video shows.
  assert.doesNotMatch(parsed.lead, /^Key numbers/);
  assert.match(parsed.lead, /^The robot does rear up onto its hind legs\./);
  assert.equal(critique.sentences(parsed.lead).length, 2);
  assert.ok(parsed.more);
  assert.ok(parsed.words > 250);
  const kinds = parsed.blocks.map((b) => b.kind);
  assert.ok(kinds.includes("ul") && kinds.includes("ol"));
  const fixes = parsed.blocks.find((b) => b.kind === "ol");
  assert.equal(fixes.items.length, 6);
  assert.match(fixes.items[0], /^Make the bipedal-hold term the main reward/);

  const flagged = critique.parseCritique(realCritiques.hacked.critique);
  assert.deepEqual(flagged.flags, ["Reward hacking suspected"]);
  assert.doesNotMatch(flagged.lead, /Reward hacking suspected/);
});

test("sentences split on real prose without breaking numbers or abbreviations", () => {
  assert.deepEqual(critique.sentences("upright 0.662 (e.g. body tilt). base_height drops to 0.196. Cut it 5×. Done"), [
    "upright 0.662 (e.g. body tilt).",
    "base_height drops to 0.196.",
    "Cut it 5×.",
    "Done",
  ]);
});

test("Claude's notes render as a lead, a collapsed full text, lists and code names", () => {
  const html = renderToStaticMarkup(React.createElement(critique.default, { text: realCritiques.hacked.critique }));
  assert.match(html, /class="critique-flag">Reward hacking suspected</);
  assert.match(html, /<p class="critique-lead">/);
  assert.match(html, /<details class="critique-full"><summary>Read Claude&#x27;s full notes/);
  assert.match(html, /<ol>/);
  assert.match(html, /<code[^>]*>term_vertical_hold<\/code>/);
  // Known metrics get their plain name on hover.
  assert.match(html, /<code title="Body height">base_height<\/code>/);
  assert.match(html, /<strong>Key numbers:<\/strong>/);
  // A one-sentence note has nothing more to open.
  const short = renderToStaticMarkup(React.createElement(critique.default, { text: "Fake critique: try a stronger task term." }));
  assert.doesNotMatch(short, /critique-full/);
});

test("the success check table uses short headers, plain metric names and units", () => {
  const SuccessCheck = require("../src/components/SuccessCheck.tsx").default;
  const html = renderToStaticMarkup(React.createElement(SuccessCheck, { check: realCritiques.best.check_result }));
  assert.match(html, /<th scope="col">Metric<\/th><th scope="col">Got<\/th><th scope="col">Need<\/th><th scope="col">Result<\/th>/);
  assert.doesNotMatch(html, /Criterion|Measured|Required/);
  assert.match(html, /<th scope="row" title="abs_forward_velocity">Speed forwards or backwards<\/th>/);
  assert.match(html, /<th scope="row" title="base_height">Body height<\/th><td class="num">0.28 m<\/td><td class="num">&gt; 0.2 m<\/td>/);
  // "fell < 0.5" reads as yes / no.
  assert.match(html, /title="fell">Fell over<\/th><td class="num">no<\/td><td class="num">no<\/td>/);
  assert.match(html, /3 of 3 criteria met/);
});

test("the score sits in a row over the video; the table and notes run full width under it", () => {
  const detail = {
    ...base,
    status: "done",
    best_score: realCritiques.best.score,
    success_spec: realCritiques.success_spec,
    rounds: [
      scoredRound(1, 1, {
        ...candidate,
        score: realCritiques.best.score,
        critique: realCritiques.best.critique,
        check_result: realCritiques.best.check_result,
        idea: "Classic velocity-tracking kernel on term_track_vel.",
      }),
    ],
  };
  const html = render(detail);
  assert.match(html, /class="verdict-row summary-best"><p class="score "><span class="num">4\.0<\/span>/);
  assert.match(html, /✓ Passed the success check/);
  // The evaluation follows the result/status grid instead of sitting in its result column.
  const grid = html.indexOf('class="run-grid');
  const rollout = html.indexOf('aria-label="Robot rollout"');
  const evaluation = html.indexOf('class="evaluation"');
  assert.ok(grid < rollout && rollout < evaluation);
  assert.doesNotMatch(html.slice(rollout, evaluation), /check-criteria|critique-lead/);
  assert.match(html.slice(evaluation), /check-criteria[\s\S]*critique-lead/);
  // What the candidate tries: two lines on its card, in full in its in-depth panel.
  assert.match(html, /class="candidate-idea idea">Classic velocity-tracking kernel on <code>term_track_vel<\/code>/);
  assert.match(html, /What this reward tries/);
});

test("the success sentence reads every op the right way round", () => {
  const say = (metric, op, value) => explain.describeClause({ metric, op, value });
  assert.equal(say("base_height", ">", 0.2), "keeps its body above 0.2 m");
  assert.equal(say("base_height", ">=", 0.2), "keeps its body at least 0.2 m");
  assert.equal(say("base_height", "<", 0.2), "keeps its body below 0.2 m");
  assert.equal(say("base_height", "<=", 0.2), "keeps its body at most 0.2 m");
  assert.equal(say("abs_forward_velocity", "<", 0.3), "moves forwards or backwards slower than 0.3 m/s");
  assert.equal(say("abs_forward_velocity", "<=", 0.3), "moves forwards or backwards at most 0.3 m/s");
  assert.equal(say("abs_forward_velocity", ">=", 0.3), "moves forwards or backwards at least 0.3 m/s");
  assert.equal(say("forward_velocity", "<", -0.3), "walks backwards faster than 0.3 m/s");
  assert.equal(say("forward_velocity", "<=", -0.3), "walks backwards at least 0.3 m/s");
  assert.equal(say("forward_velocity", ">=", 0.3), "walks forwards at least 0.3 m/s");
  assert.equal(say("abs_yaw_rate", ">=", 1), "turns either way at least 1 rad/s (57°/s)");
  assert.equal(say("fell", "<", 0.5), "doesn't fall over");
  assert.equal(say("fell", "<=", 0.5), "doesn't fall over");
  assert.equal(say("fell", ">=", 0.5), "falls over");
  assert.equal(say("target_final_dist", "<=", 0.5), "ends within 0.5 m of the target");
});

test("outcome metrics read as what the robot does, in a share of episodes", () => {
  const say = (metric, op, value) => explain.describeClause({ metric, op, value });
  assert.equal(say("all:reach:ball&push:ball:goal", ">=", 0.8),
    "reaches the ball and pushes the ball into the goal in at least 80% of episodes");
  assert.equal(say("scene_fell", "<=", 0.1), "falls in at most 10% of episodes");
  assert.equal(say("target_fell", "<", 0.1), "falls in fewer than 10% of episodes");
  assert.equal(say("target_reached", ">", 0.8), "reaches the target in more than 80% of episodes");
  assert.equal(say("sequence:climb:step>stand_on:step@2", ">=", 0.75),
    "climbs onto the step, then stands still on the step for 2 s in at least 75% of episodes");
  assert.equal(say("avoid:red_cone", ">=", 0.8), "avoids the red cone while making progress in at least 80% of episodes");
  // The table and the sentence come from the same parse.
  assert.equal(explain.checkLabel("all:reach:ball&push:ball:goal"), "Reach ball and Push ball into goal");
  assert.equal(explain.checkValue("scene_fell", 0.375), "37.5%");
  assert.equal(explain.checkValue("base_height", 0.28), "0.28 m");

  // The real spec from request 6d9cfe59 used to read "has all:reach:ball&push:ball:goal below 0.8".
  assert.equal(
    explain.successSentence(realCritiques.scene.success_spec),
    "Pass = the robot reaches the ball and pushes the ball into the goal in at least 80% of episodes and falls in at most 10% of episodes.",
  );
});

test("one fall row: the per-episode rate wins over the single run's fell", () => {
  const SuccessCheck = require("../src/components/SuccessCheck.tsx").default;
  const table = (check) => renderToStaticMarkup(React.createElement(SuccessCheck, { check }));
  // Real candidate: 37.5% of episodes fell (fails), the recorded run did not (passes).
  const html = table(realCritiques.scene.both_fail_rate);
  assert.match(html, /Episodes with a fall<\/th><td class="num">37.5%<\/td><td class="num">&lt;= 10%<\/td>/);
  assert.doesNotMatch(html, /title="fell"/);
  assert.match(html, /0 of 2 criteria met/);
  assert.doesNotMatch(html, /check-note/);
  // Real candidate: 3.1% of episodes fell, but the recorded run fell, which still fails the check.
  const runFell = table(realCritiques.scene.video_fell_only);
  assert.doesNotMatch(runFell, /title="fell"/);
  assert.match(runFell, /class="check-note">[\s\S]*fell in the recorded test run/);
  // Without a per-episode rate, "fell" stays.
  assert.match(table(realCritiques.best.check_result), /title="fell">Fell over/);
});

test("a Markdown critique gets headings, bold and a lead from what the frames show", () => {
  const text = realCritiques.scene.markdown_critique;
  const parsed = critique.parseCritique(text);
  assert.deepEqual(
    parsed.blocks.filter((b) => b.kind === "h").map((b) => b.text),
    ["What the frames show", "Reward terms", "What the next reward writer should change"],
  );
  assert.doesNotMatch(parsed.lead, /\*\*/);
  assert.match(parsed.lead, /The robot walks to the ball quickly/);
  const html = renderToStaticMarkup(React.createElement(critique.default, { text }));
  assert.match(html, /<h4 class="critique-heading">What the frames show<\/h4>/);
  assert.doesNotMatch(html, /\*\*/);
  assert.match(html, /<code[^>]*>term_ball_speed_track<\/code>/);
  const bold = renderToStaticMarkup(React.createElement("p", null, critique.inline("a **big** `x_y` win")));
  assert.match(bold, /a <strong>big<\/strong> <code>x_y<\/code> win/);
});

test("follow-ups join their task's conversation, in turn order", () => {
  const reqs = [
    { id: "t1", text: "push the ball", status: "done", best_score: 6.1, created_at: 1, conversation_id: "t1", turn: 1 },
    { id: "f3", text: "keep going", status: "queued", best_score: null, created_at: 9, conversation_id: "t1", turn: 3 },
    { id: "f2", text: "get behind it first", status: "running", best_score: 7.4, created_at: 5, conversation_id: "t1", turn: 2 },
    { id: "x", text: "spin", status: "done", best_score: 5, created_at: 3, conversation_id: "x", turn: 1 },
  ];
  const convos = thread.conversationsFromRequests(reqs);
  assert.deepEqual(convos.map((c) => c.id), ["x", "t1"]);
  const task = convos[1];
  assert.deepEqual(task.turns.map((t) => [t.requestId, t.turn]), [["t1", 1], ["f2", 2], ["f3", 3]]);
  assert.equal(task.title, "push the ball");
  assert.equal(task.status, "queued"); // the latest turn's
  assert.equal(task.bestScore, 7.4);
  // The queue names a follow-up by its task, not by its note alone.
  assert.deepEqual(thread.queuePlace(reqs, "f3"), {
    running: "push the ball (follow-up: get behind it first)",
    ahead: [],
  });
});

test("a follow-up turn says its mode, model and what it refines", () => {
  const detail = {
    ...base,
    text: "get behind the ball before pushing",
    status: "queued",
    turn: 2,
    conversation_id: "t1",
    mode: "instant",
    model: null,
    target: { id: 7, request_id: "t1", turn: 1, round_n: 2, idx: 2, score: 6.5 },
  };
  const html = render(detail);
  assert.match(html, /class="turn-meta"/);
  assert.match(html, /Follow-up/);
  assert.match(html, /Instant · Sonnet 5\.5/);
  assert.match(html, /refines[\s\S]*Turn 1 · Round 2 · Candidate 3[\s\S]*6\.5\/10/);
  const scene = render({ ...detail, target: null, status: "running", scene_changed: 1, mode: "effort", model: "claude-haiku-5-5" });
  assert.match(scene, /Effort · Haiku 5\.5/);
  assert.match(scene, /new scene, so it trains from scratch/);
  // A run from before modes shows no meta line.
  assert.doesNotMatch(render({ ...base, status: "done" }), /turn-meta/);
});

test("the mode toggle moves the model to the mode's default unless the user picked one", () => {
  const Options = require("../src/components/MessageOptions.tsx").default;
  const seen = [];
  const element = (value) =>
    Options({ value, onChange: (next) => seen.push(next), models: ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"], idPrefix: "t" });
  const html = renderToStaticMarkup(element({ mode: "instant", model: "claude-sonnet-5-5" }));
  // One switch: off is Instant.
  assert.match(html, /role="switch" aria-checked="false"[^>]*>[\s\S]*?Effort</);
  assert.match(html, /<option value="claude-haiku-5-5">Haiku 5\.5<\/option>/);
  // Flip the Effort switch.
  const find = (node, pred) => {
    if (!node || typeof node !== "object") return null;
    if (pred(node)) return node;
    const kids = [].concat(node.props?.children ?? []);
    for (const k of kids) {
      const hit = find(k, pred);
      if (hit) return hit;
    }
    return null;
  };
  const effort = (value) => find(element(value), (n) => n.props?.role === "switch");
  effort({ mode: "instant", model: "claude-sonnet-5-5" }).props.onClick();
  effort({ mode: "instant", model: "claude-haiku-5-5", picked: true }).props.onClick();
  assert.deepEqual(seen, [
    { mode: "effort", model: "claude-opus-5-5" },
    { mode: "effort", model: "claude-haiku-5-5", picked: true },
  ]);
});

test("the follow-up composer may send an empty note (keep going)", () => {
  const Composer = require("../src/components/Composer.tsx").default;
  const html = renderToStaticMarkup(
    React.createElement(Composer, {
      label: "Follow up",
      inputId: "follow-up",
      inputRef: { current: null },
      busy: false,
      error: "",
      text: "",
      onText: () => {},
      onSubmit: () => {},
      examples: false,
      allowEmpty: true,
      submitLabel: "Keep going",
      context: React.createElement("p", null, "Refines the best"),
    }),
  );
  assert.match(html, /for="follow-up"/);
  assert.match(html, /id="follow-up"/);
  assert.match(html, />Keep going</);
  assert.match(html, /Refines the best/);
  assert.doesNotMatch(html, /Examples, click to use/);
});

test("a run's time counts from when the worker picked it up, with time in line apart", () => {
  const detail = {
    ...base,
    status: "done",
    created_at: T0,
    events: [
      { id: 1, kind: "queued", message: "Request queued", created_at: T0 },
      { id: 2, kind: "mode", message: "Instant", created_at: T0 + 4500 },
      { id: 3, kind: "done", message: "Finished", created_at: T0 + 4643 },
    ],
    rounds: [],
  };
  const html = renderToStaticMarkup(React.createElement(RunStatus, { detail, now: (T0 + 5000) * 1000 }));
  assert.match(html, /<dt>Took<\/dt><dd class="num" data-testid="run-timer">2m 23s</);
  assert.match(html, /<dt>In line first<\/dt><dd class="num" data-testid="wait-timer">1h 15m</);
});

test("the new-task composer offers Auto and only the registered robots; a follow-up has no robot picker", () => {
  const Options = require("../src/components/MessageOptions.tsx").default;
  const { bodiesOf } = require("../src/modes.ts");
  const g1 = { name: "g1", title: "Unitree G1", kind: "humanoid", summary: "Unitree G1: humanoid" };
  const go1 = { name: "go1", title: "Unitree Go1", kind: "quadruped", summary: "Unitree Go1: small quadruped dog" };
  const seen = [];
  const element = (value, bodies) =>
    Options({ value, onChange: (next) => seen.push(next), models: ["claude-opus-5-5"], bodies, idPrefix: "t" });
  const html = renderToStaticMarkup(element({ mode: "effort", model: "claude-opus-5-5", body: "auto" }, bodiesOf({ ...health, bodies: [go1, g1] })));
  assert.match(html, />Robot</);
  assert.match(html, /<option value="auto" selected="">Auto<\/option><option value="go1">Unitree Go1<\/option><option value="g1">Unitree G1<\/option>/);
  assert.doesNotMatch(html, /panda/i);
  // An older server lists no bodies: the Go1 only.
  assert.deepEqual(bodiesOf(health).map((b) => b.name), ["go1"]);
  // Picking a robot keeps the mode and model.
  const find = (node, pred) => {
    if (!node || typeof node !== "object") return null;
    if (pred(node)) return node;
    for (const k of [].concat(node.props?.children ?? [])) {
      const hit = find(k, pred);
      if (hit) return hit;
    }
    return null;
  };
  const select = find(element({ mode: "effort", model: "claude-opus-5-5", body: "auto" }, [go1, g1]),
    (n) => n.type === "select" && n.props.title?.startsWith("Claude picks"));
  select.props.onChange({ target: { value: "g1" } });
  assert.deepEqual(seen, [{ mode: "effort", model: "claude-opus-5-5", body: "g1" }]);
  // A follow-up keeps its task's robot: no picker.
  assert.doesNotMatch(renderToStaticMarkup(element({ mode: "instant", model: "claude-opus-5-5" })), /Robot/);
});

test("each run names its robot, and why Claude picked it on Auto", () => {
  const withG1 = { ...health, bodies: [{ name: "go1", title: "Unitree Go1" }, { name: "g1", title: "Unitree G1" }] };
  const run = (detail) => renderToStaticMarkup(React.createElement(Detail, { detail, health: withG1 }));
  const picked = run({ ...base, status: "running", mode: "effort", body: "g1", body_reason: "the task needs two legs" });
  assert.match(picked, /class="tag body-tag"[^>]*>Unitree G1</);
  assert.match(picked, /Using the Unitree G1: the task needs two legs/);
  const auto = run({ ...base, mode: "effort", body: null });
  assert.match(auto, /class="tag body-tag"[^>]*>Auto</);
  assert.match(auto, /Claude picks the robot when it starts/);
  // A run from before bodies is the Go1, with no reason line.
  const old = run({ ...base, status: "done" });
  assert.match(old, /class="tag body-tag"[^>]*>Unitree Go1</);
  assert.doesNotMatch(old, /Using the/);
});

test("the sidebar labels each task with its robot", () => {
  const List = require("../src/components/ConversationList.tsx").default;
  const { conversationsFromRequests } = require("../src/thread.ts");
  const conversations = conversationsFromRequests([
    { id: "a", text: "wave hello", status: "done", best_score: 7, created_at: 2, conversation_id: "a", turn: 1, body: "g1" },
    { id: "b", text: "keep going", status: "queued", best_score: null, created_at: 3, conversation_id: "a", turn: 2, body: "g1" },
    { id: "c", text: "trot", status: "queued", best_score: null, created_at: 1, conversation_id: "c", turn: 1, body: null },
  ]);
  const html = renderToStaticMarkup(React.createElement(List, {
    conversations, selected: null, loading: false, onSelect: () => {},
    health: { ...health, bodies: [{ name: "go1", title: "Unitree Go1" }, { name: "g1", title: "Unitree G1" }] },
  }));
  assert.match(html, /Wave hello[\s\S]*class="tag body-tag">Unitree G1</);
  assert.match(html, /Trot[\s\S]*class="tag body-tag">Auto</);
});

test("an Instant run recording its video shows one status, not two full bars", () => {
  const html = render({
    ...base,
    status: "running",
    mode: "instant",
    rounds: [
      {
        id: 1,
        n: 1,
        status: "running",
        best_candidate_id: null,
        candidates: [
          { ...candidate, score: null, critique: null, status: "training", progress: live("rendering", 1) },
        ],
      },
    ],
  });
  assert.match(html, /Recording video/);
  assert.match(html, /Trained for 1k steps/);
  // The stage has no full bar, and one candidate gets no lane beside it.
  assert.match(html, /training-progress large"><strong class="training-stage">Recording video<\/strong><span/);
  assert.doesNotMatch(html, /class="lanes"/);
});
