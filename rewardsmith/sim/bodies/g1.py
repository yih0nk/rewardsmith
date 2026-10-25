"""The Unitree G1: Playground's G1 joystick humanoid with a generated reward (sim/g1.py).

Cheap to import: sim/g1.py (jax, mujoco) loads on the first make_env or randomizer call.
"""

import os
from typing import Any

from .. import g1_spec
from ..scene import EMPTY, SceneSpec
from . import register
from .base import Capabilities

DEFAULT_WALKER = "/workspace/pretrained/g1_walker/params.pkl"  # scripts/train_walker.py --env g1


class G1Body:
    name = "g1"
    title = "Unitree G1"
    base_env = "G1JoystickFlatTerrain"
    capabilities = Capabilities(
        kind="humanoid", legs=2, can_walk=True, bipedal=True, can_grasp=False, reach_m=None,
        summary="Unitree G1: humanoid (about 1.3 m tall, 29 joints: legs, waist, arms), walks on two legs, "
                "turns, steps sideways, can wave its arms; no hand or gripper control, no scene objects yet",
    )
    nq, nv = 36, 35  # free joint + 29 joints (sim/g1.py ROBOT_NQ, ROBOT_NV)
    skill = "walker"
    camera = {
        # Front-right, a few metres back so the whole 1.3 m body is in frame; aim at the hips.
        "chase": {"azimuth": 140.0, "elevation": -12.0, "distance": 3.0, "lookat_drop": 0.1, "lookat_min": 0.4},
        "side": {"azimuth": -90.0, "elevation": -8.0, "distance": 3.2, "lookat_drop": 0.1, "lookat_min": 0.4},
    }
    # Playground's G1 pushes: every 5-10 s a horizontal velocity kick of 0.1-2 m/s.
    push_overrides = {"push_config.enable": True}

    @property
    def skill_checkpoint(self) -> str:
        return os.environ.get("REWARDSMITH_G1_WALKER", DEFAULT_WALKER)

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
        from .. import g1

        if mode == "steer":
            options.setdefault("walker", self.skill_checkpoint)
        return g1.make_env(
            reward_source, impl=impl, overrides=overrides, scene=scene, behavior_metrics=behavior_metrics, **options
        )

    def describe(self, scene: SceneSpec | None = None, *, mode: str = "off", **options: Any) -> str:
        return g1_spec.describe(scene or EMPTY, mode=mode, **options)

    def metric_names(self) -> dict[str, tuple[str, float, float]]:
        return dict(g1_spec.POSE_METRICS)

    def randomizer(self):
        from .legged import legged_randomizer

        return legged_randomizer(self.nq, self.nv)  # floor is geom 0 and the pelvis body 1, as for the Go1

    def ppo_overrides(self) -> dict:
        return {}


G1 = register(G1Body())
