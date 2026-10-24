"""Train one reward in its own process. parallel.py starts one of these per candidate.

    python -m rewardsmith.sim.run_one <reward.py> <out_dir> [--preset round] [--impl jax] [--fake]

Writes <out_dir>/result.json, plus frames/NN.png for the critic. On failure,
result.json holds {"error": "..."} and the exit code is 1.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

RESULT_FILE = "result.json"
FRAMES_DIR = "frames"


def write_result(out_dir: Path, result) -> None:
    frames_dir = out_dir / FRAMES_DIR
    frames_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for i, png in enumerate(result.frames):
        name = f"{i:02d}.png"
        (frames_dir / name).write_bytes(png)
        names.append(name)
    payload = {
        "metrics": result.metrics,
        "reward_curve": result.reward_curve,
        "video_path": str(result.video_path) if result.video_path else None,
        "frames": names,
    }
    (out_dir / RESULT_FILE).write_text(json.dumps(payload))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reward", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--preset", default="round")
    ap.add_argument("--impl", default="jax")
    ap.add_argument("--fake", action="store_true", help="use the no-GPU fake simulator (tests)")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.fake:
            from .fake import FakeSimulator

            sim = FakeSimulator()
        else:
            from .playground import PlaygroundSimulator

            sim = PlaygroundSimulator(preset=args.preset, impl=args.impl)
        result = sim.run(args.reward.read_text(), args.out_dir)
    except Exception as exc:
        (args.out_dir / RESULT_FILE).write_text(
            json.dumps({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        )
        traceback.print_exc()
        return 1
    write_result(args.out_dir, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
