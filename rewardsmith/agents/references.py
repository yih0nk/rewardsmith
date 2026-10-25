"""Reference rewards shown to the writer and editor: one proven reward per task family.

The examples live in examples/rewards/ (each trains on the GPU pod with the round
preset). The family is read off the environment notes, which go1_spec.describe()
writes from the scene, and off the request for scene-less behaviors.
"""

import re
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "rewards"

# family: (example file, what it shows)
FAMILIES = {
    "push": ("push_to_zone.py", "moving an object into a goal"),
    "reach": ("go_to_then_stop.py", "going to an object and stopping beside it"),
    "rough": ("walk_rough.py", "walking on rough ground"),
    "spin": ("spin_in_place.py", "turning in place"),
}
_SPIN = re.compile(r"\b(spin|spins|spinning|turn|turns|turning|rotate|rotates|rotating|twirl)\b", re.I)


def family(request: str, env_doc: str) -> str | None:
    """The task family a request belongs to, or None (no reference for it)."""
    if 'state["target_to_goal_dist"]' in env_doc:
        return "push"
    if 'state["target_dist"]' in env_doc:
        return "reach"
    if _SPIN.search(request):
        return "spin"
    if "rough, bumpy ground" in env_doc:
        return "rough"
    return None


def reference(request: str, env_doc: str) -> str:
    """A prompt section with the family's reference reward, or "" when there is none."""
    fam = family(request, env_doc)
    if fam is None:
        return ""
    name, shows = FAMILIES[fam]
    path = EXAMPLES / name
    if not path.is_file():  # an install without the examples folder
        return ""
    return (
        f"Reference reward for {shows}. It trains to a working policy in this environment with the same "
        "budget; its scene may differ from this one. Build on its structure and adapt it to the request, the "
        "objects' names and the sizes in the environment notes; don't copy it blindly:\n"
        f"```python\n{path.read_text().strip()}\n```\n\n"
    )
