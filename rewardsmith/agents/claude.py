"""Shared Claude call: structured JSON out, server-side refusal fallback on."""

import json
from typing import Any

import anthropic

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class RefusalError(RuntimeError):
    pass


def ask_json(
    client: anthropic.Anthropic,
    model: str,
    system: str,
    content: list[dict[str, Any]],
    schema: dict[str, Any],
    effort: str = "high",
    max_tokens: int = 16000,
) -> dict[str, Any]:
    response = client.beta.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": content}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        betas=[FALLBACK_BETA],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise RefusalError(str(response.stop_details))
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)
