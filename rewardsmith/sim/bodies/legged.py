"""Shared pieces for legged bodies: a floating base (free joint) followed by the joints."""

from collections.abc import Callable
from typing import Any

import jax

FLOOR_GEOM_ID = 0  # the stock and compiled scenes both put the floor first
TORSO_BODY_ID = 1


def legged_randomizer(nq: int, nv: int, floor_geom: int = FLOOR_GEOM_ID, torso_body: int = TORSO_BODY_ID) -> Callable[..., Any]:
    """A legged-robot domain randomizer (the Go1's), one robot per env, that leaves scene objects alone.

    Same ranges as Playground's Go1 randomizer: floor friction U(0.4, 1.0), joint friction
    loss x U(0.9, 1.1), joint armature x U(1.0, 1.05), every link mass x U(0.9, 1.1), torso
    payload +U(-1, 1) kg, torso center of mass +U(-5, 5) cm, joint zero offsets (qpos0)
    +U(-0.05, 0.05) rad. Playground's version slices dof[6:] and qpos[7:] to the end, which
    only fits when the robot is the only moving body: a movable scene object adds a free
    joint (6 dofs, 7 qpos) after the robot, and the shapes stop matching. Here only the
    robot's joints are touched (the body's nq / nv: free joint first, then the joints).
    Called as randomizer()(mjx_model, rngs) -> (batched model, in_axes).
    """

    def domain_randomize(model: Any, rng: jax.Array) -> tuple[Any, Any]:
        @jax.vmap
        def rand_dynamics(rng):
            rng, key = jax.random.split(rng)
            geom_friction = model.geom_friction.at[floor_geom, 0].set(
                jax.random.uniform(key, minval=0.4, maxval=1.0)
            )
            rng, key = jax.random.split(rng)
            dof_frictionloss = model.dof_frictionloss.at[6:nv].multiply(
                jax.random.uniform(key, shape=(nv - 6,), minval=0.9, maxval=1.1)
            )
            rng, key = jax.random.split(rng)
            dof_armature = model.dof_armature.at[6:nv].multiply(
                jax.random.uniform(key, shape=(nv - 6,), minval=1.0, maxval=1.05)
            )
            rng, key = jax.random.split(rng)
            body_ipos = model.body_ipos.at[torso_body].add(
                jax.random.uniform(key, (3,), minval=-0.05, maxval=0.05)
            )
            rng, key = jax.random.split(rng)
            body_mass = model.body_mass * jax.random.uniform(key, shape=(model.nbody,), minval=0.9, maxval=1.1)
            rng, key = jax.random.split(rng)
            body_mass = body_mass.at[torso_body].add(jax.random.uniform(key, minval=-1.0, maxval=1.0))
            rng, key = jax.random.split(rng)
            qpos0 = model.qpos0.at[7:nq].add(
                jax.random.uniform(key, shape=(nq - 7,), minval=-0.05, maxval=0.05)
            )
            return geom_friction, body_ipos, body_mass, qpos0, dof_frictionloss, dof_armature

        friction, body_ipos, body_mass, qpos0, frictionloss, armature = rand_dynamics(rng)
        fields = {
            "geom_friction": friction, "body_ipos": body_ipos, "body_mass": body_mass,
            "qpos0": qpos0, "dof_frictionloss": frictionloss, "dof_armature": armature,
        }
        in_axes = jax.tree_util.tree_map(lambda x: None, model).tree_replace({k: 0 for k in fields})
        return model.tree_replace(fields), in_axes

    return domain_randomize
