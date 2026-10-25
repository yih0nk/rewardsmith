"""Reward revision between rounds.

DroidEditor is the target (Factory prize): run when a round finishes, its
workspace holds only that round's reward files, and every change is a diff.
ClaudeEditor is the fallback (REWARDSMITH_EDITOR=claude, the default until
Droid is tested on the pod).
"""

import difflib
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Sequence

import anthropic
from pydantic import BaseModel

from .base import Attempt, Draft
from .claude import ask_json
from .references import reference
from .stream import IdeaStream, OnIdea
from .robot import robot
from .writer import WRITE_SCHEMA, system_for

log = logging.getLogger("rewardsmith.editor")


def unified_diff(old: str, new: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True), new.splitlines(keepends=True), "reward.py (before)", "reward.py (after)"
        )
    )


MAX_HISTORY = 12
MAX_CRITIQUE = 500


def format_history(history: Sequence[Attempt]) -> str:
    lines = []
    for a in list(history)[-MAX_HISTORY:]:
        critique = a.critique if len(a.critique) <= MAX_CRITIQUE else a.critique[:MAX_CRITIQUE] + "..."
        where = f"Turn {a.turn}, round {a.round}" if a.turn > 1 else f"Round {a.round}"
        lines.append(f"- {where}, candidate {a.idx}, score {a.score:.1f}: {critique}")
    return "\n".join(lines)


class ClaudeEditor:
    def __init__(self, client: anthropic.Anthropic, model: str, effort: str = "high", body: str | None = None):
        self.client, self.model, self.effort = client, model, effort
        self.system = system_for(body)

    def revise(
        self,
        request: str,
        env_doc: str,
        best_source: str,
        critique: str,
        metrics: dict[str, Any],
        k: int,
        history: Sequence[Attempt] = (),
        on_idea: OnIdea | None = None,
    ) -> list[Draft]:
        """on_idea(i, idea) is called as each revision's idea streams in."""
        earlier = (
            f"Earlier attempts and what the critic said (don't repeat what failed):\n{format_history(history)}\n\n"
            if history
            else ""
        )
        prompt = (
            f"Environment notes:\n{env_doc}\n\n"
            f"Behavior the user wants: {request}\n\n"
            f"{reference(request, env_doc)}"
            f"Best reward so far:\n```python\n{best_source}\n```\n\n"
            f"Metrics from its rollout:\n{json.dumps(metrics, indent=2)}\n\n"
            f"Critic's notes after watching the video:\n{critique}\n\n"
            f"{earlier}"
            f"Write {k} revised reward modules that fix what the critic saw. If the critique names the "
            "phase where the task fails, change the terms for that phase first. "
            "Each idea should say what you changed and why."
        )
        out = ask_json(
            self.client, self.model, self.system, [{"type": "text", "text": prompt}], WRITE_SCHEMA, effort=self.effort,
            on_text=IdeaStream(on_idea) if on_idea else None,
        )
        return [Draft(c["source"], unified_diff(best_source, c["source"]), c["idea"]) for c in out["candidates"]][:k]


class DroidEditor:
    """Factory Droid revises the best reward, locked to a fresh folder per round.

    Each call builds a workspace holding only K copies of the best reward
    (candidate_0.py ...), the critique, metrics, earlier attempts and the env
    notes. Droid runs there at low autonomy: it may edit project files, and
    anything more (shell commands, installs, network) has no permission handler,
    so the SDK cancels it. We read back the candidate files and diff each one.
    """

    def __init__(self, workspace_root: Path, model: str | None = None, timeout: float = 900, body: str | None = None):
        self.workspace_root, self.model, self.timeout, self.body = workspace_root, model, timeout, body

    def revise(
        self,
        request: str,
        env_doc: str,
        best_source: str,
        critique: str,
        metrics: dict[str, Any],
        k: int,
        history: Sequence[Attempt] = (),
    ) -> list[Draft]:
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        ws = Path(tempfile.mkdtemp(prefix="round-", dir=self.workspace_root)).resolve()
        names = [f"candidate_{i}.py" for i in range(k)]
        for name in names:
            (ws / name).write_text(best_source)
        (ws / "critique.md").write_text(critique + "\n")
        (ws / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (ws / "history.md").write_text((format_history(history) or "No earlier attempts.") + "\n")
        (ws / "env.md").write_text(env_doc + "\n")
        before = {p.name for p in ws.iterdir()}

        ideas = self._run(_droid_prompt(request, names, self.body), ws)

        after = {p.name for p in ws.iterdir()}
        if after - before:
            log.info("droid created extra files in %s: %s", ws, sorted(after - before))
        drafts = []
        for i, name in enumerate(names):
            new = (ws / name).read_text()
            if new.strip() == best_source.strip():
                continue  # Droid left this one alone
            idea = ideas[i] if i < len(ideas) else ""
            drafts.append(Draft(new, unified_diff(best_source, new), idea or None))
        if not drafts:
            raise RuntimeError(f"droid changed no candidate files in {ws}")
        return drafts

    def _run(self, prompt: str, cwd: Path) -> list[str]:
        """Run Droid once in cwd and return one idea per candidate."""
        import asyncio

        from droid_sdk import Autonomy, SessionConfig, run

        config = SessionConfig(autonomy=Autonomy.LOW, disabled_tools={"Execute"})
        result = asyncio.run(
            run(prompt, cwd=cwd, model=self.model, config=config, output=DroidIdeas, timeout=self.timeout)
        )
        if not result.success:
            raise RuntimeError(f"droid run {result.subtype}: {result.error or result.text}")
        return result.output.ideas if result.output else []


class DroidIdeas(BaseModel):
    ideas: list[str]


def _droid_prompt(request: str, names: list[str], body: str | None = None) -> str:
    files = ", ".join(names)
    return (
        f"The user wants a simulated {robot(body)} to: {request}\n\n"
        f"{files} each hold the same reward function, the best one so far. A PPO policy was "
        "trained on it; critique.md says what the robot did wrong after watching the video, "
        "metrics.json has the rollout measurements, history.md lists earlier attempts and "
        "env.md describes what reward() can read.\n\n"
        f"Edit each of {files} to fix what the critique describes, with a different fix in "
        "each file. Keep the signature reward(state, action, prev_action) -> dict of named "
        "scalar terms, use only jax and jax.numpy, and use jnp.where instead of Python if on "
        "traced values. Edit only those files; don't create or run anything.\n\n"
        "Return ideas: one short sentence per file, in file order, saying what you changed and why."
    )
