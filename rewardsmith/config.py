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


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip() == "1"


def load_dotenv(path: Path = Path(".env")) -> None:
    """Copy KEY=value lines from .env into the environment. Variables already set win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if sep and key and not key.startswith("#") and value:
            os.environ.setdefault(key, value)


def load() -> Settings:
    load_dotenv()
    home =Path(os.environ.get("REWARDSMITH_HOME", ".rewardsmith"))
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
    )
