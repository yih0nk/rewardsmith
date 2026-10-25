"""Classify the judge-style prompts with the real scene writer, and check each scene it builds.

    set -a; . /path/to/.env; set +a
    python scripts/feasibility_bench.py [--gate] [--out feasibility.json]

Reads tests/data/feasibility_prompts.json (prompt + expected class), calls the Claude scene
writer on each (8 at a time), and prints the verdict, whether the spec parses, and with --gate
whether it passes the physics scene gate (CPU). Ends with the accuracy per class.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import anthropic  # noqa: E402

from rewardsmith import config  # noqa: E402
from rewardsmith.agents.scener import FEASIBILITY, ClaudeScener  # noqa: E402
from rewardsmith.sim import scene  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true", help="also run the physics scene gate on each scene")
    ap.add_argument("--out", type=Path, default=Path("feasibility.json"))
    args = ap.parse_args()
    config.load_dotenv()
    cases = json.loads((ROOT / "tests" / "data" / "feasibility_prompts.json").read_text())
    scener = ClaudeScener(anthropic.Anthropic(), config.load().model)

    def one(case):
        try:
            spec = scener.make(case["prompt"])
        except Exception as exc:
            return {**case, "verdict": None, "error": f"{type(exc).__name__}: {exc}"}
        verdict = spec.pop(FEASIBILITY, {}) or {}
        row = {**case, "verdict": verdict.get("verdict"), "reason": verdict.get("reason"),
               "simplification": verdict.get("simplification"), "suggestion": verdict.get("suggestion"), "spec": spec}
        try:
            parsed = scene.parse(spec)
            row["parsed"] = True
            row["summary"] = parsed.summary()
        except scene.SceneError as exc:
            row["parsed"], row["parse_error"] = False, str(exc)
        return row

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(one, cases))
    if args.gate:
        from rewardsmith.sim import scene_gate

        for row in rows:
            if row.get("parsed") and row["verdict"] != "refuse":
                row["gate_error"] = scene_gate.check_scene(scene.parse(row["spec"]))
    args.out.write_text(json.dumps(rows, indent=2))
    for row in rows:
        ok = "ok " if row["verdict"] == row["expected"] else "XX "
        gate = "" if "gate_error" not in row else (" gate:pass" if row["gate_error"] is None else " gate:FAIL")
        print(f"{ok}{row['expected']:>11} -> {str(row['verdict']):<11} parsed={row.get('parsed')}{gate}  {row['prompt']}")
        if row["verdict"] in ("refuse", "approximate"):
            print(f"      {row.get('reason')} | {row.get('simplification')} | try: {row.get('suggestion')}")
    for cls in ("doable", "approximate", "refuse"):
        mine = [r for r in rows if r["expected"] == cls]
        print(f"{cls}: {sum(r['verdict'] == cls for r in mine)}/{len(mine)}")
    print(f"overall: {sum(r['verdict'] == r['expected'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
