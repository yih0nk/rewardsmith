"""The Franka Emika Panda arm: Playground's PandaPickCube with a generated reward (sim/panda.py).

Cheap to import: sim/panda.py (jax, mujoco) loads on the first make_env or randomizer call.
"""

from typing import Any

from .. import panda_spec
from ..scene import EMPTY, SceneSpec
from . import register
from .base import Capabilities


class PandaBody:
    name = "panda"
    title = "Franka Panda arm"
    base_env = "PandaPickCube"
    capabilities = Capabilities(
        kind="arm", legs=0, can_walk=False, bipedal=False, can_grasp=True, reach_m=panda_spec.REACH_M,
        summary="Franka Emika Panda: 7-joint arm with a two-finger gripper, bolted to a table, reach about 0.85 m; "
                "picks up, lifts, carries, pushes and places one small block on the table; cannot walk or leave "
                "the table",
    )
    nq, nv = 9, 9  # 7 arm joints + 2 fingers (sim/panda.py ROBOT_NQ, ROBOT_NV); the block's free joint follows
    skill = "ee"  # the policy moves the gripper; Playground's analytic IK moves the joints
    skill_checkpoint = None  # the IK needs no training
    camera = {
        # In front of the arm and to its right, looking back at the gripper and the block, from above.
        "chase": {"azimuth": 140.0, "elevation": -28.0, "distance": 1.25, "lookat": [0.45, 0.0, 0.12]},
        # Square to the arm's left side, low: how high the block goes.
        "side": {"azimuth": -90.0, "elevation": -10.0, "distance": 1.2, "lookat": [0.45, 0.0, 0.15]},
    }

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
        """mode "steer" (the default) drives the end-effector skill; "off" or "tune" drive the joints.
        options (the Go1's walker or stock terms) don't apply to the arm and are ignored."""
        from .. import panda

        return panda.make_env(
            reward_source, impl=impl, overrides=overrides, scene=scene, behavior_metrics=behavior_metrics,
            ee=mode in (None, "steer"),
        )

    def describe(self, scene: SceneSpec | None = None, *, mode: str = "steer", **options: Any) -> str:
        return panda_spec.describe(scene or EMPTY, mode="steer" if mode in (None, "steer") else mode)

    def metric_names(self) -> dict[str, tuple[str, float, float]]:
        return dict(panda_spec.ARM_METRICS)

    def randomizer(self):
        from .. import panda

        return panda.randomizer()

    def ppo_overrides(self) -> dict:
        return {"episode_length": panda_spec.TRAIN_EPISODE_STEPS}


PANDA = register(PandaBody())
