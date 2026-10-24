import type { Candidate, RequestDetail } from "./api";

const reward = `def reward(state, action, prev_action):
    return {
        "backward_velocity": -state.velocity[0] * 2.0,
        "upright": state.up_vector[2] * 0.5,
        "smooth_motion": -jnp.sum((action - prev_action) ** 2) * 0.02,
    }`;
const stages = [
  "Writing reward",
  "Checking reward",
  "Training policy",
  "Rendering rollout",
  "Critiquing behavior",
];
const started = new Map<string, number>();
function candidate(n: number): Candidate {
  return {
    id: n,
    idx: 0,
    reward_src: reward,
    diff:
      n > 1
        ? "- backward_velocity: 1.0\n+ backward_velocity: 2.0\n+ smooth_motion: -0.02 * action_delta"
        : null,
    status: "scored",
    error: null,
    metrics: {
      backward_speed: Number((0.18 + n * 0.16).toFixed(2)),
      stability: 69 + n * 7,
      smoothness: 56 + n * 9,
    },
    video_path: null,
    score: [3.4, 6.2, 8.6][n - 1],
    critique: [
      "The robot takes a few backward steps, but loses balance. Increase the upright reward to encourage a stable gait.",
      "A consistent backward gait is emerging. Reduce abrupt action changes to make the motion smoother.",
      "The robot maintains a steady backward gait with good balance. A small lateral drift remains; penalize sideways velocity in the next revision.",
    ][n - 1],
  };
}
const demo: RequestDetail = {
  id: "demo-backwards",
  text: "Make the robot dog walk backwards",
  status: "done",
  best_score: 8.6,
  created_at: Date.now() / 1000,
  rounds: [1, 2, 3].map((n) => ({
    id: n,
    n,
    status: "done",
    best_candidate_id: n,
    candidates: [candidate(n)],
  })),
  events: [
    {
      id: 1,
      kind: "complete",
      message: "Run complete. Best candidate selected from round 3.",
      created_at: Date.now() / 1000,
    },
  ],
};
let runs: RequestDetail[] = [demo];
const delay = () => new Promise((resolve) => setTimeout(resolve, 350));
function update(run: RequestDetail): RequestDetail {
  const start = started.get(run.id);
  if (!start) return run;
  const step = Math.floor((Date.now() - start) / 2200);
  const completed = Math.min(3, Math.floor(step / 5));
  return {
    ...run,
    status: completed === 3 ? "done" : "running",
    best_score: completed ? candidate(completed).score : null,
    rounds: demo.rounds.slice(0, completed),
    events: [
      {
        id: step,
        kind: "progress",
        message:
          completed === 3
            ? "Run complete. All three rounds evaluated."
            : stages[step % 5],
        created_at: Date.now() / 1000,
      },
    ],
  };
}
export const mockApi = {
  async list() {
    await delay();
    return runs.map(update);
  },
  async get(id: string) {
    await delay();
    const run = runs.find((r) => r.id === id);
    if (!run) throw new Error("Run not found");
    return update(run);
  },
  async create(text: string) {
    await delay();
    const id = crypto.randomUUID();
    runs = [
      {
        ...demo,
        id,
        text,
        status: "queued",
        rounds: [],
        best_score: null,
        created_at: Date.now() / 1000,
      },
      ...runs,
    ];
    started.set(id, Date.now());
    return { id };
  },
};
