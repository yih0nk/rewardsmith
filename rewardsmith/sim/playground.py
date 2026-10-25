"""MuJoCo Playground backend: the real simulator behind the worker.

Settings (env vars):
  REWARDSMITH_BODY    the robot (default go1; a bodies.BODIES key)
  REWARDSMITH_PRESET  smoke | round | final   (default round; use smoke on a laptop)
  REWARDSMITH_IMPL    jax | warp              (default: warp on a CUDA GPU, jax elsewhere)
  REWARDSMITH_PARALLEL  candidates trained at once, one process each (default 4)
  REWARDSMITH_RANDOMIZE 1 = train on domain-randomized robots (default 1: about 5% slower,
                        same learning within a round, and spin in place falls less)
  REWARDSMITH_PUSH      1 = train with random pushes (default 0)
  REWARDSMITH_ROBUSTNESS  1 = score the policy on unseen robots, seeds and pushes (default 1)
  REWARDSMITH_XLA_AUTOTUNE  1 to keep XLA's GPU autotuner on (off by default, see below)
  REWARDSMITH_WARM_START  1 = build on the body's pretrained skill (default 1, when the file exists)
  REWARDSMITH_WALKER      the Go1 walker checkpoint (default /workspace/pretrained/go1_walker/params.pkl;
                          scripts/train_walker.py makes it); other bodies name theirs in skill_checkpoint
  REWARDSMITH_MODE        how a candidate uses the walker (default steer with a walker, off without):
                          steer = the policy outputs velocity commands to the frozen walker;
                          tune  = the policy is the walker's own network, fine-tuned on the reward;
                          off   = random weights, the reward alone (the old behavior)
  REWARDSMITH_STOCK       gait | pose | off: Playground's gait terms under each reward's own in tune
                          mode (default gait; see go1.STOCK_PROFILES). Always off in steer mode.
  REWARDSMITH_STOCK_WEIGHT  multiplier on the stock terms' scales (default go1.STOCK_WEIGHT)
  REWARDSMITH_RESTORE_VALUE 1 = warm starts keep the checkpoint's value network too (default 1)
  REWARDSMITH_TUNE_NOISE  tune mode resets the walker's exploration noise to this std (default 0.5;
                          0 keeps the walker's own, about 0.1, too little to leave its gait)
  REWARDSMITH_CONTINUE_NOISE  a run training on from a candidate's checkpoint (a follow-up, a revision)
                          resets its exploration noise to this std, in steer and off mode (default 0.6;
                          0 keeps the checkpoint's own). Tune mode uses REWARDSMITH_TUNE_NOISE.
  REWARDSMITH_STEPS       training steps per candidate (default: budget.py's per-request plan with a
                          warm start, else the preset's)
"""

import logging
import os
import threading
from pathlib import Path

# The worker gates rewards on the GPU itself, so it must not grab 75% of GPU memory
# before the training processes start. Has to be set before jax initializes.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# XLA's GPU autotuner times candidate kernels for every matmul while compiling.
# With four training processes compiling on one GPU that cost about a minute a
# round, and switching it off left steps/s and rewards unchanged on the 4090.
# The training processes inherit this. Set REWARDSMITH_XLA_AUTOTUNE=1 to keep it.
if os.environ.get("REWARDSMITH_XLA_AUTOTUNE") != "1" and "xla_gpu_autotune_level" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} --xla_gpu_autotune_level=0".strip()

from . import budget, evaluation, gate, go1, go1_spec, parallel, train  # noqa: E402
from .bodies import DEFAULT_BODY, get_body  # noqa: E402
from .base import Cull, Job, OnDone, RunResult, Stopped  # noqa: E402
from .progress import ProgressWriter  # noqa: E402
from .scene import EMPTY, SceneSpec  # noqa: E402

log = logging.getLogger("rewardsmith.sim")

MODES = ("steer", "tune", "off")
TUNE_NOISE = 0.5
# A steer policy's noise shrinks as it trains and keeps shrinking when it trains on: median std per
# command (vx, vy, yaw) on the ball task was 0.64/0.73/0.69 fresh, 0.43/0.58/0.62 after 3M steps,
# 0.35/0.43/0.51 two follow-ups later, with the forward command saturated (|tanh mean| ~0.9). The
# revised reward then barely moved it (eval reward flat over 2M steps, PR #52). 0.6 is about a fresh
# policy's noise, above every value measured, so a continued run explores again. The mean is kept,
# so it starts from the parent's behavior, and in steer mode noise only perturbs the walker's commands.
CONTINUE_NOISE = 0.6


def default_walker(body: str | None = None) -> str:
    """The body's skill checkpoint every candidate starts from (the Go1's walker), or "" for random weights."""
    if not _flag("REWARDSMITH_WARM_START", "1"):
        return ""
    b = get_body(body)
    path = b.skill_checkpoint if b.skill == "walker" else None
    if not path or not Path(path).is_file():
        log.warning("no %s walker checkpoint at %s: candidates start from random weights", b.name, path)
        return ""
    return path


class PlaygroundSimulator:
    def __init__(
        self,
        env_name: str | None = None,
        preset: str | None = None,
        impl: str | None = None,
        randomize: bool | None = None,
        push: bool | None = None,
        robustness: bool | None = None,
        behavior_metrics: bool = False,
        evaluation_spec: list[dict] | None = None,
        walker: str | None = None,
        mode: str | None = None,
        stock_profile: str | None = None,
        stock_weight: float | None = None,
        steps: int | None = None,
        restore_value: bool | None = None,
        tune_noise: float | None = None,
        continue_noise: float | None = None,
        body: str | None = None,
    ):
        self._body = get_body(body or os.environ.get("REWARDSMITH_BODY") or DEFAULT_BODY).name
        if env_name is not None and env_name != get_body(self._body).base_env:
            raise ValueError(f"the {self._body} body builds on {get_body(self._body).base_env}, not {env_name}")
        self.preset = preset or os.environ.get("REWARDSMITH_PRESET", "round")
        self.impl = impl or go1.default_impl()
        self.max_parallel = int(os.environ.get("REWARDSMITH_PARALLEL", "4"))
        self.randomize = _flag("REWARDSMITH_RANDOMIZE", "1") if randomize is None else randomize
        self.push = _flag("REWARDSMITH_PUSH") if push is None else push
        self.robustness = _flag("REWARDSMITH_ROBUSTNESS", "1") if robustness is None else robustness
        self.evaluation_spec = evaluation_spec
        self.behavior_metrics = behavior_metrics  # see go1.Go1Env.behavior
        # Warm start: a checkpoint path, "" for random weights (resolved here so children get the same).
        self._auto_walker, self._mode_arg = walker is None, mode
        self.walker = default_walker(self.body) if walker is None else walker
        self.mode = (mode or os.environ.get("REWARDSMITH_MODE") or "steer") if self._can_steer() else "off"
        if self.mode not in MODES:
            raise ValueError(f"unknown mode {self.mode!r}: use one of {MODES}")
        self.stock_profile = stock_profile or os.environ.get("REWARDSMITH_STOCK") or "gait"
        if self.stock_profile not in go1.STOCK_PROFILES:
            raise ValueError(f"unknown stock profile {self.stock_profile!r}: use one of {sorted(go1.STOCK_PROFILES)}")
        self.stock_weight = (
            float(os.environ.get("REWARDSMITH_STOCK_WEIGHT", go1.STOCK_WEIGHT)) if stock_weight is None else stock_weight
        )
        self.steps = steps or int(os.environ.get("REWARDSMITH_STEPS") or 0) or None  # None: the preset's
        # plan_request may pick these per request, unless they were set explicitly.
        self._auto_steps = self.steps is None
        self._auto_stock = stock_profile is None and not os.environ.get("REWARDSMITH_STOCK")
        self._auto_mode = mode is None and not os.environ.get("REWARDSMITH_MODE")
        self.restore_value = _flag("REWARDSMITH_RESTORE_VALUE", "1") if restore_value is None else restore_value
        self.tune_noise = float(os.environ.get("REWARDSMITH_TUNE_NOISE", TUNE_NOISE)) if tune_noise is None else tune_noise
        self.continue_noise = (
            float(os.environ.get("REWARDSMITH_CONTINUE_NOISE", CONTINUE_NOISE)) if continue_noise is None else continue_noise
        )
        self._pool: parallel.WarmPool | None = None
        # The worker gates drafts in threads, but two warp rollouts at once in one
        # process crash: warp captures a CUDA graph on its first call, and any other
        # GPU work during the capture fails it (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED).
        self._gate_lock = threading.Lock()

    def describe(self, scene: SceneSpec | None = None) -> str:
        return get_body(self.body).describe(
            scene or EMPTY, mode=self.mode, stock_profile=self.env_options.get("stock_profile") or "off",
            stock_weight=self.stock_weight,
        )

    def gate(self, source: str, scene: SceneSpec | None = None) -> str | None:
        """Gate in a warm training process when one is free, else here.

        In a warm process the gates of a round run side by side instead of one at a
        time, on a process whose imports and first compile are already paid, and the
        process that passed a source then trains it (see parallel.WarmPool).
        """
        pool = self._pool
        warm = pool.reserve() if pool else None
        if warm is not None:
            try:
                error = warm.gate(source, scene, self.impl, self.env_options, self.body)
            except parallel.WarmProcessError:
                log.exception("warm gate failed; gating in the worker instead")
                pool.discard(warm)
            else:
                pool.release(warm)
                return error
        with self._gate_lock:
            return gate.check_reward(source, impl=self.impl, scene=scene, env_options=self.env_options, body=self.body)

    @property
    def env_options(self) -> dict:
        """The body's make_env arguments for this mode: the walker to steer, or the stock terms to tune under."""
        if self.mode == "steer":
            return {"walker": self.walker}
        if self.mode == "tune":
            return {"stock_profile": self.stock_profile, "stock_weight": self.stock_weight}
        return {}

    def run(
        self, source: str, out_dir: Path, scene: SceneSpec | None = None,
        init_params: str | Path | None = None, steps: int | None = None,
    ) -> RunResult:
        """Train from init_params for steps (default: self.steps or the preset's).

        Without init_params, tune mode starts from the walker's weights; steer and off start from random ones.
        """
        steps = steps or self.steps or train.ppo_kwargs(self.preset, body=self.body)["num_timesteps"]
        progress = ProgressWriter(out_dir, steps)
        progress.stage("compiling")
        progress.start_heartbeat()  # nothing else writes during the minute of JIT compile
        try:
            start = init_params or (self.walker if self.mode == "tune" else None)
            return self._run(source, out_dir, scene, progress, start, steps)
        finally:
            progress.stop_heartbeat()

    def _run(
        self, source: str, out_dir: Path, scene: SceneSpec | None, progress: ProgressWriter,
        init_params: str | Path | None, steps: int,
    ) -> RunResult:
        body = get_body(self.body)
        env = body.make_env(source, impl=self.impl, scene=scene, behavior_metrics=self.behavior_metrics, **self.env_options)
        # Pushes are a training condition only: the video and metrics come from a push-free env.
        # The training env keeps the behavior metrics too: halving ranks on the training curve.
        push_overrides = getattr(body, "push_overrides", None)
        train_env = (
            body.make_env(
                source, impl=self.impl, overrides=push_overrides, scene=scene,
                behavior_metrics=self.behavior_metrics, **self.env_options,
            )
            if self.push and push_overrides
            else env
        )
        result = train.train(
            train_env, self.preset, {"num_timesteps": steps}, on_progress=progress.point, randomize=self.randomize,
            init_params=init_params, checkpoint_dir=out_dir, restore_value=self.restore_value,
            init_noise=self.start_noise(init_params),
            body=self.body,
        )
        progress.stage("rendering")
        # The video, target metrics and robustness rollouts, all in one batch (see evaluation.py).
        video, frames, rollout, robust = evaluation.evaluate(
            env, result, out_dir, robust=self.robustness, success_spec=self.evaluation_spec, body=self.body,
        )
        (out_dir / "reward.py").write_text(source)

        last = result.curve[-1]
        metrics = {
            "episode_reward": round(last.episode_reward, 3),
            **{f"term_{k}": round(v, 3) for k, v in last.terms.items()},
            **{k: round(v, 3) for k, v in rollout.items()},
            **robust,
            "train_s": round(result.train_s, 1),
            "compile_s": round(result.compile_s, 1),
            "train_randomized": int(self.randomize),
            "train_pushes": int(self.push),
            "train_steps": int(steps),
            "warm_started": int(init_params is not None),
            "steered": int(self.mode == "steer"),
        }
        progress.stage("done")
        return RunResult(metrics=metrics, frames=frames, video_path=video, reward_curve=result.reward_curve)

    def start_noise(self, init_params: str | Path | None) -> float | None:
        """The exploration std a run starting from init_params resets to, or None to keep the checkpoint's.

        Tune mode fine-tunes the walker's own network: any checkpoint gets tune_noise. In steer and off
        mode a candidate's checkpoint gets continue_noise (see CONTINUE_NOISE). 0 keeps the checkpoint's.
        """
        if init_params is None:
            return None
        return (self.tune_noise if self.mode == "tune" else self.continue_noise) or None

    def sim_options(self) -> dict[str, object]:
        """Everything a training child needs to train as this simulator would (PlaygroundSimulator(**these))."""
        return {
            "randomize": self.randomize, "push": self.push, "robustness": self.robustness,
            "evaluation_spec": self.evaluation_spec, "walker": self.walker, "mode": self.mode, "stock_profile": self.stock_profile,
            "stock_weight": self.stock_weight, "steps": self.steps, "restore_value": self.restore_value,
            "tune_noise": self.tune_noise, "continue_noise": self.continue_noise, "body": self.body,
        }

    def plan_request(self, text: str, scene: SceneSpec | None = None) -> str:
        """Pick this request's mode, steps per candidate and stock profile (budget.py); return the plan for the log.

        Poses fine-tune the walker (tune, pose stock terms); everything else steers it.
        Only with a walker: the budgets were measured with one. Explicit settings win.
        """
        if not self.walker:
            return "Training plan: preset steps, no warm start"
        plan = budget.plan(text, scene or EMPTY, get_body(self.body).capabilities.bipedal)
        if self._auto_mode:
            self.mode = "tune" if plan.family == "pose" else "steer"
        if self._auto_steps:
            self.steps = plan.steps
        if self._auto_stock:
            self.stock_profile = plan.family
        return f"{plan.summary()}, {self.mode} mode"

    def set_evaluation_spec(self, spec: list[dict] | None) -> None:
        self.evaluation_spec = spec

    @property
    def body(self) -> str:
        """The robot every call trains, gates and describes (a bodies.BODIES key)."""
        return self._body

    @body.setter
    def body(self, name: str | None) -> None:
        self.set_body(name)

    def set_body(self, name: str | None) -> None:
        """Train the next request on this body (None: the default). Its walker replaces the
        current one unless a walker was given explicitly; call before plan_request.
        `sim.body = name` does the same."""
        self._body = get_body(name or os.environ.get("REWARDSMITH_BODY") or DEFAULT_BODY).name
        if self._auto_walker:
            self.walker = default_walker(self.body)
        self.mode = (self._mode_arg or os.environ.get("REWARDSMITH_MODE") or "steer") if self._can_steer() else "off"

    def _can_steer(self) -> bool:
        """Steer mode needs the body's skill: the walker's checkpoint, or none for a skill that
        needs no training (the arm's end-effector IK)."""
        b = get_body(self.body)
        return bool(self.walker) or (b.skill is not None and b.skill_checkpoint is None)

    def warm_up(self, n: int) -> None:
        """Start n training processes now, so their imports overlap the writer's Claude call."""
        if self._pool is None:
            self._pool = parallel.WarmPool(n)

    def close_warm_pool(self) -> None:
        """Release unused trainers, including rounds that never reach run_many."""
        pool, self._pool = self._pool, None
        if pool:
            pool.close()

    def run_many(
        self, jobs: list[Job], scene: SceneSpec | None = None, cull: Cull | None = None,
        on_done: OnDone | None = None,
    ) -> list[RunResult | Stopped | str]:
        pool, self._pool = self._pool, None
        # Halving compares candidates at the same eval, so they all train at once.
        max_parallel = max(self.max_parallel, len(jobs)) if cull else self.max_parallel
        if cull is not None and getattr(cull, "keep", None) and _flag("REWARDSMITH_SURVIVOR_STEPS", "1"):
            jobs = self._stretch(jobs, cull.keep)
        try:
            return parallel.run_in_processes(
                jobs, preset=self.preset, impl=self.impl, max_parallel=max_parallel, scene=scene, pool=pool,
                cull=cull, behavior_metrics=cull is not None or self.behavior_metrics,
                sim_options=self.sim_options(),
                on_done=on_done,
            )
        finally:
            if pool:
                pool.close()


    def _stretch(self, jobs: list[Job], keep: list[int]) -> list[Job]:
        """Give the steps halving culls to the survivors (search.survivor_stretch)."""
        from ..search import survivor_stretch

        kwargs = train.ppo_kwargs(self.preset, body=self.body)
        stretch = survivor_stretch(len(jobs), keep, kwargs["num_evals"])
        out: list[Job] = []
        for source, out_dir, *rest in jobs:
            start = dict(rest[0]) if rest else {}
            start["steps"] = round((start.get("steps") or self.steps or kwargs["num_timesteps"]) * stretch)
            out.append((source, out_dir, start))
        return out


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip() == "1"
