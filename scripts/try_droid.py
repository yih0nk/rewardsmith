"""Run the Droid editor once on a reward and print its diffs. Needs FACTORY_API_KEY and droid on PATH.

    .venv/bin/python scripts/try_droid.py
    .venv/bin/python scripts/try_droid.py my_reward.py --critique "it spins instead of backing up" --k 2

The workspace Droid saw is kept under .rewardsmith/artifacts/droid/ for inspection.
"""

import argparse
import os
import time
from pathlib import Path

from rewardsmith import config
from rewardsmith.agents.editor import DroidEditor
from rewardsmith.sim.go1_spec import describe

DEFAULT_CRITIQUE = (
    "The robot shuffles backwards for a second, then sits down and stops. "
    "Keep it upright and moving backwards for the whole episode."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reward", type=Path, nargs="?", default=Path("examples/rewards/walk_backwards.py"))
    ap.add_argument("--request", default="walk backwards")
    ap.add_argument("--critique", default=DEFAULT_CRITIQUE)
    ap.add_argument("--k", type=int, default=2)
    args = ap.parse_args()

    config.load_dotenv()
    settings = config.load()
    editor = DroidEditor(settings.artifacts_dir / "droid", model=os.environ.get("REWARDSMITH_DROID_MODEL") or None)

    t0 = time.perf_counter()
    drafts = editor.revise(args.request, describe(), args.reward.read_text(), args.critique, {}, args.k)
    print(f"{len(drafts)} revision(s) in {time.perf_counter() - t0:.1f}s")
    for i, d in enumerate(drafts):
        print(f"\n--- revision {i}: {d.idea}\n{d.diff}")


if __name__ == "__main__":
    main()
