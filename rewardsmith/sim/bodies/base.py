"""The Body protocol: what the loop needs from a robot.

The writer, gate, trainer, renderer and evaluators only talk to a robot through
a Body, so a new robot is a new module in this package (see docs/bodies.md).
The Go1 (bodies/go1.py) is the first one.
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..scene import SceneSpec

KINDS = ("quadruped", "humanoid", "arm")
SKILLS = ("walker", "ee")


class RewardError(ValueError):
    """The reward module is unusable. The message is meant for the reward writer."""


@dataclass(frozen=True)
class Capabilities:
    """What a body can do, for Claude to pick a body for a task."""

    kind: str  # "quadruped" | "humanoid" | "arm"
    legs: int  # 4, 2, 0
    can_walk: bool
    bipedal: bool
    can_grasp: bool
    reach_m: float | None  # arms: workspace radius, m
    # One line Claude reads when choosing a body, e.g. "Unitree Go1: small quadruped dog,
    # 12 joints, walks/trots/turns, pushes objects with its body, no gripper".
    summary: str


@runtime_checkable
class Body(Protocol):
    name: str  # registry key: "go1", "spot", "g1", "panda"
    title: str  # board label: "Unitree Go1"
    base_env: str  # the Playground registry env it builds on
    capabilities: Capabilities
    nq: int  # the robot's qpos slice; scene objects' qpos come after it
    nv: int  # the robot's qvel slice
    # Base-skill controller the policy can drive instead of the joints (steer mode):
    # "walker" (velocity commands), "ee" (end-effector targets), or None.
    skill: str | None
    skill_checkpoint: str | None  # the skill's pretrained params on the pod, or None
    camera: dict[str, Any]  # render camera for the rollout video (see render.py)

    def make_env(
        self,
        reward_source: str,
        scene: SceneSpec | None = None,
        impl: str | None = None,
        overrides: dict | None = None,
        mode: str | None = None,
        behavior_metrics: bool = False,
        **options: Any,
    ) -> Any:
        """An mjx_env.MjxEnv whose step() calls the generated reward(state, action, prev_action).

        mode "steer" drives the body's skill controller (skill_checkpoint, or options["walker"]);
        anything else drives the joints. options are body-specific (the Go1's stock terms).
        Raises RewardError for an unusable reward.
        """
        ...

    def describe(self, scene: SceneSpec | None = None, *, mode: str = "off", **options: Any) -> str:
        """The writer's notes: the state keys and their units, the action, what "fell" means,
        the stock terms."""
        ...

    def metric_names(self) -> dict[str, tuple[str, float, float]]:
        """Behavior metrics the checker can use for this body: name -> (meaning, low, high),
        the shape of go1_spec.POSE_METRICS. Iterating it gives the names."""
        ...

    def randomizer(self) -> Any:
        """A domain randomizer that touches only the robot slice:
        randomizer()(mjx_model, rngs) -> (batched model, in_axes)."""
        ...

    def ppo_overrides(self) -> dict:
        """PPO settings on top of Playground's tuned config for base_env."""
        ...
