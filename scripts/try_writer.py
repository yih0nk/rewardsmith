"""Ask the real Claude writer for rewards, gate them, and save them. Needs ANTHROPIC_API_KEY.

    .venv/bin/python scripts/try_writer.py "walk backwards"
    .venv/bin/python scripts/try_writer.py "spin in place" --k 2 --out out/spin

Uses the real compile gate when jax and playground are installed (the GPU pod),
otherwise the fake syntax-only gate. Train a saved reward with scripts/train_reward.py.
"""

import argparse
import time
from pathlib import Path

import anthropic

from rewardsmith import config
from rewardsmith.agents.writer import ClaudeWriter
from rewardsmith.orchestrator import GATE_RETRIES
from rewardsmith.sim.go1_spec import describe


def load_gate():
    try:
        from rewardsmith.sim import gate

        return gate.check_reward, "real"
    except ImportError as exc:
        from rewardsmith.sim.fake import FakeSimulator

        print(f"real gate unavailable ({exc}); using the syntax-only fake gate")
        return FakeSimulator().gate, "fake"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("request", help='the behavior, e.g. "walk backwards"')
    ap.add_argument("--k", type=int, default=1, help="how many candidates to write")
    ap.add_argument("--out", type=Path, default=Path("out/writer"))
    args = ap.parse_args()

    config.load_dotenv()
    settings = config.load()
    writer = ClaudeWriter(anthropic.Anthropic(), settings.model)
    check, kind = load_gate()
    env_doc = describe()

    t0 = time.perf_counter()
    drafts = writer.write(args.request, env_doc, args.k)
    print(f"{len(drafts)} candidate(s) from {settings.model} in {time.perf_counter() - t0:.1f}s")

    args.out.mkdir(parents=True, exist_ok=True)
    for i, draft in enumerate(drafts):
        print(f"  c{i} idea: {draft.idea}")
        src = draft.source
        error = check(src)
        for attempt in range(GATE_RETRIES):
            if error is None:
                break
            print(f"  c{i}: gate failed (attempt {attempt + 1}), asking for a fix:\n    {error[:300]}")
            src = writer.fix(src, error, env_doc)
            error = check(src)
        path = args.out / f"c{i}.py"
        path.write_text(src)
        status = "passed" if error is None else f"FAILED: {error[:300]}"
        print(f"  c{i}: {kind} gate {status} -> {path}")


if __name__ == "__main__":
    main()
