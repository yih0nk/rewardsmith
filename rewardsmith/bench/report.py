"""Turn finished benchmark requests into the team plan's four numbers.

1. Success rate after round 1 vs. best by the last round (the check, not the critic).
2. Median rounds to success.
3. Plain forward walking: the generated reward vs. the hand-written one.
4. GPU minutes (and dollars, given a rate) per learned behavior.

Plus how often the critic's score agreed with the check, for the "how do you know the
critic is right" question. Everything here reads db.get_request() dicts, so it runs
on a laptop against a copy of the pod's db.
"""

import statistics
from dataclasses import dataclass, field
from typing import Any

from .suite import CASES, Case

FINISHED = ("done", "failed")


@dataclass
class Outcome:
    """One benchmark request, boiled down."""

    case: Case
    request_id: str
    status: str
    rounds_run: int
    first_pass_round: int | None  # the first round with a candidate that passed the check
    best_score: float | None
    best_metrics: dict[str, Any] = field(default_factory=dict)
    gpu_s: float = 0.0  # compile + train time over every candidate
    wall_s: float | None = None  # request created to its last event
    agree: int = 0  # candidates where critic (score >= success_score) and check agreed
    judged: int = 0  # candidates that had both a score and a check


def _passed(check: Any) -> bool:
    # A check marked unverified (the fallback "don't fall" spec) never counts as success.
    return isinstance(check, dict) and bool(check.get("pass")) and check.get("verified", True) is not False


def outcome(case: Case, detail: dict[str, Any], success_score: float) -> Outcome:
    rounds = sorted(detail.get("rounds", []), key=lambda r: r["n"])
    first_pass = None
    gpu_s, agree, judged = 0.0, 0, 0
    scored: list[tuple[bool, float, dict[str, Any], int]] = []
    for rnd in rounds:
        for c in rnd.get("candidates", []):
            metrics = c.get("metrics") or {}
            gpu_s += sum(float(metrics.get(k) or 0) for k in ("compile_s", "train_s"))
            check = c.get("check_result")
            if _passed(check) and first_pass is None:
                first_pass = rnd["n"]
            if c.get("score") is not None:
                scored.append((_passed(check), float(c["score"]), metrics, c["id"]))
                if isinstance(check, dict):
                    judged += 1
                    agree += (float(c["score"]) >= success_score) == _passed(check)

    best_id = detail.get("best_candidate_id")
    best = next((s for s in scored if s[3] == best_id), None) or max(scored, default=None, key=lambda s: s[:2])
    events = detail.get("events") or []
    wall = events[-1]["created_at"] - detail["created_at"] if events else None
    return Outcome(
        case=case,
        request_id=detail["id"],
        status=detail["status"],
        rounds_run=len(rounds),
        first_pass_round=first_pass,
        best_score=best[1] if best else None,
        best_metrics=best[2] if best else {},
        gpu_s=gpu_s,
        wall_s=wall,
        agree=agree,
        judged=judged,
    )


def summarize(
    outcomes: list[Outcome],
    max_rounds: int,
    baseline: dict[str, Any] | None = None,
    gpu_dollars_per_hour: float | None = None,
) -> dict[str, Any]:
    behaviors = [o for o in outcomes if not o.case.reference and o.case.spec is not None]
    finished = [o for o in behaviors if o.status in FINISHED]

    def rate(r: int) -> float | None:
        if not finished:
            return None
        return sum(o.first_pass_round is not None and o.first_pass_round <= r for o in finished) / len(finished)

    successes = [o.first_pass_round for o in finished if o.first_pass_round is not None]
    gpu_min = [o.gpu_s / 60 for o in finished if o.first_pass_round is not None]
    judged = sum(o.judged for o in outcomes)

    out: dict[str, Any] = {
        "finished": len(finished),
        "pending": len(behaviors) - len(finished),
        "success_rate_by_round": {r: rate(r) for r in range(1, max_rounds + 1)},
        "succeeded": len(successes),
        "median_rounds_to_success": statistics.median(successes) if successes else None,
        "median_gpu_minutes_per_success": statistics.median(gpu_min) if gpu_min else None,
        "total_gpu_minutes": sum(o.gpu_s for o in outcomes) / 60,
        "critic_agreement": sum(o.agree for o in outcomes) / judged if judged else None,
        "critic_judged": judged,
        "unmeasured": [c.slug for c in CASES if c.spec is None],
        "reference": _reference(outcomes, baseline),
    }
    if gpu_dollars_per_hour is not None and out["median_gpu_minutes_per_success"] is not None:
        out["median_gpu_dollars_per_success"] = out["median_gpu_minutes_per_success"] / 60 * gpu_dollars_per_hour
    return out


def _reference(outcomes: list[Outcome], baseline: dict[str, Any] | None) -> dict[str, Any] | None:
    """Generated "walk forward" vs. the hand-written walk_forward reward's baseline run."""
    ours = [o for o in outcomes if o.case.reference and o.status in FINISHED and o.best_metrics]
    hand = ((baseline or {}).get("cases") or {}).get("walk_forward", {}).get("metrics")
    if not ours or not hand:
        return None
    speeds = [float(o.best_metrics.get("forward_velocity", 0.0)) for o in ours]
    generated = statistics.median(speeds)
    hand_speed = float(hand["forward_velocity"])
    return {
        "generated_forward_velocity": generated,
        "hand_written_forward_velocity": hand_speed,
        "percent_of_hand_written": 100 * generated / hand_speed if hand_speed else None,
        "generated_passed": sum(o.first_pass_round is not None for o in ours),
        "runs": len(ours),
    }


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _num(x: float | None, fmt: str = "{:.1f}") -> str:
    return "n/a" if x is None else fmt.format(x)


def markdown(summary: dict[str, Any], outcomes: list[Outcome]) -> str:
    rates = summary["success_rate_by_round"]
    last = max(rates) if rates else 1
    lines = [
        "# Benchmark results",
        "",
        f"{summary['finished']} behavior run(s) finished, {summary['pending']} still running.",
        "",
        "## The four numbers",
        "",
        f"1. **Success rate:** {_pct(rates.get(1))} after round 1, {_pct(rates.get(last))} by round {last}.",
        f"2. **Median rounds to success:** {_num(summary['median_rounds_to_success'], '{:g}')} "
        f"({summary['succeeded']} of {summary['finished']} succeeded).",
    ]
    ref = summary["reference"]
    if ref:
        lines.append(
            f"3. **Plain forward walking:** the generated reward reached "
            f"{ref['generated_forward_velocity']:.2f} m/s, {_num(ref['percent_of_hand_written'], '{:.0f}')}% of the "
            f"hand-written reward's {ref['hand_written_forward_velocity']:.2f} m/s "
            f"(examples/rewards/walk_forward.py, regression baseline)."
        )
    else:
        lines.append("3. **Plain forward walking:** no finished walk_forward run or no regression baseline yet.")
    gpu = f"{_num(summary['median_gpu_minutes_per_success'], '{:.0f}')} GPU minutes per learned behavior (median)"
    if "median_gpu_dollars_per_success" in summary:
        gpu += f", about ${summary['median_gpu_dollars_per_success']:.2f}"
    lines += [
        f"4. **Cost:** {gpu}; {summary['total_gpu_minutes']:.0f} GPU minutes in total.",
        "",
        f"Critic vs. check: agreed on {_pct(summary['critic_agreement'])} of {summary['critic_judged']} scored candidates.",
        "",
        "## Success rate by round",
        "",
        "| Round | " + " | ".join(str(r) for r in rates) + " |",
        "|---|" + "---|" * len(rates),
        "| Passed | " + " | ".join(_pct(v) for v in rates.values()) + " |",
        "",
        "## Per behavior",
        "",
        "| Behavior | Status | Rounds | First pass | Best score | GPU min | Check |",
        "|---|---|---|---|---|---|---|",
    ]
    for o in outcomes:
        check = "not measurable yet" if o.case.spec is None else ", ".join(
            f"{c['metric']} {c['op']} {c['value']:g}" for c in o.case.spec
        )
        lines.append(
            f"| {o.case.slug} | {o.status} | {o.rounds_run} | {o.first_pass_round or '-'} | "
            f"{_num(o.best_score)} | {o.gpu_s / 60:.0f} | {check} |"
        )
    if summary["unmeasured"]:
        lines += ["", "Not run, no metric can check them yet: " + ", ".join(summary["unmeasured"]) + "."]
    proxies = {o.case.slug: o.case.proxy for o in outcomes if o.case.proxy and o.case.spec is not None}
    if proxies:
        lines += ["", "## Where the check differs from the team plan", ""]
        lines += [f"- **{slug}:** {proxy}" for slug, proxy in proxies.items()]
    return "\n".join(lines) + "\n"
