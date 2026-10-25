import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    db_path: Path
    artifacts_dir: Path
    model: str
    env_name: str
    candidates_per_round: int
    max_rounds: int
    success_score: float
    fake_sim: bool
    fake_llm: bool
    min_rounds: int = 2  # always run at least this many, so the editor and its diffs get used
    carry_top: int = 2  # each later round revises this many of the best candidates so far
    halving: tuple[int, ...] = ()  # e.g. (8, 4, 2): successive halving within a round; () = off
    editor: str = "claude"  # claude | droid: who revises rewards from round 2 on
    continue_rounds: bool = True  # round 2+ candidates train on from their parent's policy, not from scratch
    coach: str = "off"  # the live coach's model (e.g. claude-haiku-5-5), or "off"; a request can override it


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip() == "1"


def load_dotenv(path: Path = Path(".env")) -> None:
    """Copy KEY=value lines from .env into the environment. Variables already set win.

    The server module and the worker's main() call this; load() does not, so code
    that only builds Settings (tests included) never picks up a developer's .env.
    """
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if sep and key and not key.startswith("#") and value:
            os.environ.setdefault(key, value)


def _halving(text: str) -> tuple[int, ...]:
    from .search import parse_halving

    return parse_halving(text)


def load() -> Settings:
    home = Path(os.environ.get("REWARDSMITH_HOME", ".rewardsmith"))
    return Settings(
        db_path=Path(os.environ.get("REWARDSMITH_DB", home / "rewardsmith.db")),
        artifacts_dir=Path(os.environ.get("REWARDSMITH_ARTIFACTS", home / "artifacts")),
        model=os.environ.get("REWARDSMITH_MODEL", "claude-opus-5-5"),
        env_name=os.environ.get("REWARDSMITH_ENV", "Go1JoystickFlatTerrain"),
        candidates_per_round=int(os.environ.get("REWARDSMITH_K", "4")),
        max_rounds=int(os.environ.get("REWARDSMITH_ROUNDS", "5")),
        success_score=float(os.environ.get("REWARDSMITH_SUCCESS_SCORE", "8")),
        fake_sim=_flag("REWARDSMITH_FAKE_SIM", "1"),
        fake_llm=_flag("REWARDSMITH_FAKE_LLM", "1"),
        min_rounds=int(os.environ.get("REWARDSMITH_MIN_ROUNDS", "2")),
        carry_top=max(1, int(os.environ.get("REWARDSMITH_CARRY", "2"))),
        halving=_halving(os.environ.get("REWARDSMITH_HALVING", "")),
        editor=os.environ.get("REWARDSMITH_EDITOR", "claude").strip().lower(),
        continue_rounds=_flag("REWARDSMITH_CONTINUE", "1"),
        coach=os.environ.get("REWARDSMITH_COACH", "claude-haiku-5-5").strip() or "off",
    )
