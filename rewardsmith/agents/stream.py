"""Pick each candidate's idea out of a writer or editor reply while it streams.

The reply is WRITE_SCHEMA JSON, and structured outputs keep the schema's key order,
so every candidate's "idea" arrives before its "source". IdeaStream is fed the
reply's text deltas (ask_json's on_text) and calls on_idea(i, idea) the moment
candidate i's idea string is complete, a minute or so before the whole reply is.
"""

import json
import logging
import re
from collections.abc import Callable

log = logging.getLogger("rewardsmith.agents")

# A complete "idea": "..." pair. Inside a source string the quotes are escaped (\"idea\"),
# so reward code that happens to say "idea" can't match.
IDEA = re.compile(r'"idea"\s*:\s*"((?:[^"\\]|\\.)*)"')

OnIdea = Callable[[int, str], None]


class IdeaStream:
    def __init__(self, on_idea: OnIdea):
        self.on_idea = on_idea
        self.text = ""
        self.pos = 0  # everything before this is already parsed
        self.count = 0

    def __call__(self, delta: str) -> None:
        self.text += delta
        while (m := IDEA.search(self.text, self.pos)) is not None:
            self.pos = m.end()
            try:
                idea = json.loads(f'"{m.group(1)}"')
            except ValueError:
                idea = m.group(1)
            try:
                self.on_idea(self.count, idea)
            except Exception:  # showing the idea early is a nicety; the reply still matters
                log.exception("on_idea failed")
            self.count += 1
