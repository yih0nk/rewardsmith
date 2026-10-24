"""brax 0.14.2 still calls jax.device_put_replicated / device_put_sharded, which jax 0.10
removed from the public namespace (the implementations still exist). flax needs
jax>=0.10, so we can't pin jax back; restore the two names instead.

Import this before brax's PPO trainer. Drop it once brax stops using them.
"""

import jax
from jax._src import api as _api

for _name in ("device_put_replicated", "device_put_sharded"):
    if _name not in jax.__dict__ and hasattr(_api, _name):
        setattr(jax, _name, getattr(_api, _name))
