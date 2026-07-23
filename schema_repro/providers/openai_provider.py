"""OpenAI-backed provider (GPT-5.6 Sol) — best-effort.

Schema reached 95.35% RHAE with GPT-5.6 Sol, which is why the harness keeps a
second provider behind the same interface. The SDK import is lazy; the exact
request surface for GPT-5.6 Sol reasoning effort is provider-specific and may
change — this maps the harness's effort tiers onto a ``reasoning_effort`` hint
and is clearly a reconstruction, not a claim about the original run.
"""

from __future__ import annotations

import time

from schema_repro.providers.base import ModelRequest, ModelResponse
from schema_repro.types import Effort

# GPT-5.6 Sol exposes coarser effort tiers than the harness; map onto them.
_EFFORT_MAP = {
    Effort.LOW: "low",
    Effort.MEDIUM: "medium",
    Effort.HIGH: "high",
    Effort.XHIGH: "high",
}


class OpenAIProvider:
    def __init__(self, model: str = "gpt-5.6-sol", *, max_tokens: int = 8192):
        self.model = model
        self.name = f"openai:{model}"
        self.max_tokens = max_tokens
        self._client = None

    def _client_or_init(self):
        if self._client is None:
            import openai  # deferred

            self._client = openai.OpenAI()
        return self._client

    def complete(self, request: ModelRequest) -> ModelResponse:
        client = self._client_or_init()
        started = time.monotonic()
        messages = [{"role": "system", "content": request.system}, *request.messages]
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=request.max_tokens or self.max_tokens,
            reasoning_effort=_EFFORT_MAP.get(request.effort, "high"),
            **request.extra,
        )
        choice = resp.choices[0]
        return ModelResponse(
            text=choice.message.content or "",
            model=getattr(resp, "model", self.model),
            stop_reason="refusal" if choice.finish_reason == "content_filter" else "end_turn",
            input_tokens=getattr(resp.usage, "prompt_tokens", 0),
            output_tokens=getattr(resp.usage, "completion_tokens", 0),
            latency_s=time.monotonic() - started,
        )
