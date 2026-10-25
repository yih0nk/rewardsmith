"""Instant and Effort: how hard one message makes the worker search.

Instant is for a first video fast: one reward, one round, a fast model, a small
training budget, no live coach. Effort searches wider: four rewards written by Opus,
trained side by side with the live coach watching. More rounds are follow-ups ("keep
going"), which revise the best and train on from its policy.

Why Effort is one round: on the 4090 every round pays a fixed ~90 s PPO compile (four
compiling at once on ~10 CPUs) plus ~50 s of Opus writing, so two rounds took 537 s on
the ball task (round 2 alone ~260 s, no better score). One round fits under 5 min.

Every per-mode number lives here. The worker turns a request's mode and model into
the Settings and agents for that run (orchestrator.main).
"""

from dataclasses import dataclass, replace
from typing import Any

from . import config

OPUS, SONNET, HAIKU = "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"
# The writer/editor/critic models a message may pick, strongest first.
MODELS = (OPUS, SONNET, HAIKU)

INSTANT, EFFORT = "instant", "effort"
# What a message gets when it names no mode: a new task searches, a follow-up is quick.
DEFAULT_TASK_MODE = EFFORT
DEFAULT_FOLLOWUP_MODE = INSTANT
MIN_TRAINING_STEPS = 1_000_000
MAX_TRAINING_STEPS = 100_000_000


@dataclass(frozen=True)
class Mode:
    name: str
    model: str  # writer, editor and critic, unless the message picks another
    effort: str  # the writer's, editor's and critic's thinking effort
    scene_model: str  # scene writer and checker
    k: int | None  # rewards per round; None = the worker's REWARDSMITH_K
    min_rounds: int | None  # None = the worker's REWARDSMITH_MIN_ROUNDS
    max_rounds: int | None  # None = the worker's REWARDSMITH_ROUNDS
    steps: int | None  # most training steps per candidate (caps the simulator's plan); None = the plan's
    continue_steps: int | None  # steps for a candidate continuing its parent's policy; None = the plan's
    coach: bool  # the live coach watches training
    robustness: bool  # score the policy on unseen robots, seeds and pushes too


# Targets, submit to final score on the 4090: Instant under 2 min, Effort under 5 min.
# Steps are on the pretrained walker (steer mode), which matched 10M-step cold runs at 5M.
MODES = {
    INSTANT: Mode(
        INSTANT, model=SONNET, effort="medium", scene_model=SONNET,
        k=1, min_rounds=1, max_rounds=1, steps=3_000_000, continue_steps=4_000_000, coach=False, robustness=False,
    ),
    EFFORT: Mode(
        EFFORT, model=OPUS, effort="medium", scene_model=SONNET,
        k=4, min_rounds=1, max_rounds=1, steps=5_000_000, continue_steps=4_000_000, coach=True, robustness=False,
    ),
}


def of(req: dict[str, Any]) -> Mode:
    """A request's mode. Rows from before modes ran the full search."""
    mode = MODES.get(req.get("mode") or EFFORT, MODES[EFFORT])
    if req.get("training_steps") is not None:
        mode = replace(mode, steps=req["training_steps"], continue_steps=req["training_steps"])
    return mode


def model_of(req: dict[str, Any]) -> str:
    """The writer/editor/critic model for a request: its pick, else its mode's."""
    return req.get("model") or of(req).model


def settings_for(settings: config.Settings, req: dict[str, Any]) -> config.Settings:
    """The worker's settings with the request's mode applied. A model the request picked wins."""
    mode = of(req)
    out = replace(settings, model=mode.model, max_steps=mode.steps, continue_steps=mode.continue_steps)
    if req.get("model"):
        out = replace(out, model=req["model"])
    if mode.k is not None:
        out = replace(out, candidates_per_round=mode.k, halving=())
    if mode.max_rounds is not None:
        out = replace(out, max_rounds=mode.max_rounds)
    if mode.min_rounds is not None:
        out = replace(out, min_rounds=min(mode.min_rounds, out.max_rounds))
    return out


def scene_model(settings: config.Settings, req: dict[str, Any]) -> str:
    """The scene writer's and checker's model (fast in both modes: they gate everything else)."""
    return of(req).scene_model


def largest_round(settings: config.Settings) -> int:
    """The most candidates any mode trains at once (the idle worker warms this many processes)."""
    return max(m.k or settings.candidates_per_round for m in MODES.values())


def describe() -> dict[str, Any]:
    """The modes and models, for /api/health (the board's toggle and picker)."""
    return {
        "models": list(MODELS),
        "modes": {
            name: {"model": m.model, "k": m.k, "max_rounds": m.max_rounds, "steps": m.steps,
                   "continue_steps": m.continue_steps, "coach": m.coach}
            for name, m in MODES.items()
        },
        "default_task_mode": DEFAULT_TASK_MODE,
        "default_followup_mode": DEFAULT_FOLLOWUP_MODE,
    }
