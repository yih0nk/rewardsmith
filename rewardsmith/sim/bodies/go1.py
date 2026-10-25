"""The Unitree Go1: Playground's Go1 joystick env with a generated reward (sim/go1.py).

Cheap to import: sim/go1.py (jax, mujoco) loads on the first make_env or randomizer call.
"""

import os
from typing import Any

from .. import go1_spec
from ..scene import EMPTY, SceneSpec
from . import register
from .base import Capabilities

DEFAULT_WALKER = "/workspace/pretrained/go1_walker/params.pkl"  # scripts/train_walker.py


class Go1Body:
    name = "go1"
    title = "Unitree Go1"
    base_env = "Go1JoystickFlatTerrain"
    capabilities = Capabilities(
        kind="quadruped", legs=4, can_walk=True, bipedal=False, can_grasp=False, reach_m=None,
        summary="Unitree Go1: small quadruped dog (12 joints, about 30 cm tall), walks/trots/turns, can rear up "
                "on its hind legs, pushes objects with its body, no gripper",
    )
    nq, nv = 19, 18  # free joint + 12 joints (sim/go1.py ROBOT_NQ, ROBOT_NV)
    skill = "walker"
    camera = {
        # In front of the robot and to its right, looking back at it, a little from above.
        "chase": {"azimuth": 140.0, "elevation": -18.0, "distance": 1.35},
        # Square to the robot's starting left side, near ground level: gait, height and flips.
        "side": {"azimuth": -90.0, "elevation": -14.0, "distance": 1.5},
    }
    # Playground's random pushes: every 1-3 s a velocity kick of up to 3 m/s, for 0.05-0.2 s.
    push_overrides = {"pert_config.enable": True}

    @property
    def skill_checkpoint(self) -> str:
        return os.environ.get("REWARDSMITH_WALKER", DEFAULT_WALKER)

    def make_env(
        self,
        reward_source: str,
        scene: SceneSpec | None = None,
        impl: str | None = None,
        overrides: dict | None = None,
        mode: str | None = None,
        behavior_metrics: bool = False,
        **options: Any,
    ):
        from .. import go1

        if mode == "steer":
            options.setdefault("walker", self.skill_checkpoint)
        return go1.make_env(
            reward_source, impl=impl, overrides=overrides, scene=scene, behavior_metrics=behavior_metrics, **options
        )

    def describe(self, scene: SceneSpec | None = None, *, mode: str = "off", **options: Any) -> str:
        return go1_spec.describe(scene or EMPTY, mode=mode, **options)

    def metric_names(self) -> list[str]:
        return list(go1_spec.POSE_METRICS)

    def randomizer(self):
        from .legged import legged_randomizer

        return legged_randomizer(self.nq, self.nv)

    def ppo_overrides(self) -> dict:
        return {}


GO1 = register(Go1Body())
