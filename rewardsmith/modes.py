"""Instant and Effort: how hard one message makes the worker search.

Instant is for a first video fast: one reward, one round, a fast model, a smaller
training budget, no live coach. Effort is the full search the board always ran:
K rewards a round, at least two rounds, the strongest model, the live coach on.

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


@dataclass(frozen=True)
class Mode:
    name: str
    model: str  # writer, editor and critic, unless the message picks another
    effort: str  # the writer's, editor's and critic's thinking effort
    scene_model: str  # scene writer and checker
    k: int | None  # rewards per round; None = the worker's REWARDSMITH_K
    min_rounds: int | None  # None = the worker's REWARDSMITH_MIN_ROUNDS
    max_rounds: int | None  # None = the worker's REWARDSMITH_ROUNDS
    steps: int | None  # training steps per candidate; None = the simulator's preset
    coach: bool  # the live coach watches training
    robustness: bool  # score the policy on unseen robots, seeds and pushes too


MODES = {
    INSTANT: Mode(
        INSTANT, model=SONNET, effort="medium", scene_model=SONNET,
        k=1, min_rounds=1, max_rounds=1, steps=6_000_000, coach=False, robustness=False,
    ),
    EFFORT: Mode(
        EFFORT, model=OPUS, effort="high", scene_model=OPUS,
        k=None, min_rounds=None, max_rounds=None, steps=None, coach=True, robustness=True,
    ),
}


def of(req: dict[str, Any]) -> Mode:
    """A request's mode. Rows from before modes ran the full search."""
    return MODES.get(req.get("mode") or EFFORT, MODES[EFFORT])


def model_of(req: dict[str, Any]) -> str:
    """The writer/editor/critic model for a request: its pick, else its mode's."""
    return req.get("model") or of(req).model


def settings_for(settings: config.Settings, req: dict[str, Any]) -> config.Settings:
    """The worker's settings with the request's mode applied.

    Effort keeps the worker's own (REWARDSMITH_MODEL, _K, _ROUNDS, ...), so a request
    from before modes runs exactly as it did. A model the request picked wins in both.
    """
    mode = of(req)
    out = settings if mode.name == EFFORT else replace(settings, model=mode.model)
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
    """The scene writer's and checker's model: Effort keeps the worker's REWARDSMITH_MODEL."""
    mode = of(req)
    return settings.model if mode.name == EFFORT else mode.scene_model


def describe() -> dict[str, Any]:
    """The modes and models, for /api/health (the board's toggle and picker)."""
    return {
        "models": list(MODELS),
        "modes": {
            name: {"model": m.model, "k": m.k, "max_rounds": m.max_rounds, "steps": m.steps, "coach": m.coach}
            for name, m in MODES.items()
        },
        "default_task_mode": DEFAULT_TASK_MODE,
        "default_followup_mode": DEFAULT_FOLLOWUP_MODE,
    }
