"""Shared Claude call: structured JSON out, server-side refusal fallback on."""

import json
from typing import Any

import anthropic

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class RefusalError(RuntimeError):
    pass


class TruncatedError(RuntimeError):
    pass


def ask_json(
    client: anthropic.Anthropic,
    model: str,
    system: str,
    content: list[dict[str, Any]],
    schema: dict[str, Any],
    effort: str = "high",
    max_tokens: int = 64000,
) -> dict[str, Any]:
    # Streamed so a large max_tokens doesn't hit the SDK's HTTP timeout.
    # Thinking counts toward max_tokens, so leave room for K full reward modules.
    with client.beta.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": content}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        betas=[FALLBACK_BETA],
        fallbacks="default",
    ) as stream:
        response = stream.get_final_message()
    if response.stop_reason == "refusal":
        raise RefusalError(str(response.stop_details))
    if response.stop_reason == "max_tokens":
        raise TruncatedError(f"reply hit max_tokens={max_tokens} before the JSON was complete")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)
