"""Anthropic-backed provider (Fable 5 primary, Opus 4.8 refusal-fallback target).

Schema reached ~99% RHAE with Opus 4.8 + Fable 5. In this repo's default chain
(``configs/models.yaml``) Fable 5 is the model that is called and Opus 4.8 is the
model its safety refusals fall back to; this provider is one model per instance,
and the registry wires the ordering. It maps the
harness's abstract request onto the current Anthropic Messages API:

* reasoning effort  -> ``output_config.effort`` (low|medium|high|xhigh|max);
* Opus 4.8          -> adaptive thinking (``thinking={"type": "adaptive"}``);
* Fable 5           -> thinking is always on (omit the parameter), and the
                       *server-side refusal fallback* to Opus 4.8 is enabled by
                       default (``betas=["server-side-fallback-2026-06-01"]`` +
                       ``fallbacks=[{"model": "claude-opus-4-8"}]``). That safety
                       fallback is exactly the "Fable-5 fallback logic" the
                       reproduction can only *enable*, not re-implement — it runs
                       inside Anthropic's API. See ``docs/MODEL_AND_FALLBACK.md``.

The SDK import is lazy so the rest of the package (offline tests, RHAE recompute)
works without the dependency installed or an API key present.
"""

from __future__ import annotations

import time

from schema_repro.providers.base import ModelRequest, ModelResponse
from schema_repro.types import Effort

# Model families whose thinking is always on and must NOT receive an explicit
# thinking param. Matched by prefix so a dated snapshot (e.g. "claude-fable-5-...")
# is still recognised — that is what makes the "keep the family prefix intact"
# env-var override guidance in docs/MODEL_AND_FALLBACK.md correct.
_ALWAYS_THINKING = ("claude-fable-5", "claude-mythos-5")


def _is_always_thinking(model: str) -> bool:
    return any(model.startswith(fam) for fam in _ALWAYS_THINKING)
# The server-side refusal fallback target used for the Fable-5 -> Opus-4.8 hand-off.
_DEFAULT_FALLBACK_MODEL = "claude-opus-4-8"
_FALLBACK_BETA = "server-side-fallback-2026-06-01"


class AnthropicProvider:
    """Wraps the official ``anthropic`` SDK. One instance per concrete model."""

    def __init__(self, model: str, *, enable_refusal_fallback: bool = True, max_tokens: int = 8192):
        self.model = model
        self.name = f"anthropic:{model}"
        self.enable_refusal_fallback = enable_refusal_fallback and _is_always_thinking(model)
        self.max_tokens = max_tokens
        self._client = None  # lazily constructed

    def _client_or_init(self):
        if self._client is None:
            import anthropic  # deferred; only needed for live runs

            self._client = anthropic.Anthropic()
        return self._client

    def _params(self, request: ModelRequest) -> dict:
        params: dict = {
            "model": self.model,
            "max_tokens": request.max_tokens or self.max_tokens,
            "system": request.system,
            "messages": request.messages,
            # effort is the depth control across Opus 4.8 / Fable 5 / Sonnet 5.
            "output_config": {"effort": _effort(request.effort)},
        }
        # Opus-tier: adaptive thinking must be requested explicitly. Fable 5:
        # thinking is always on, so the parameter is omitted.
        if not _is_always_thinking(self.model):
            params["thinking"] = {"type": "adaptive"}
        if self.enable_refusal_fallback:
            params["betas"] = [_FALLBACK_BETA]
            params["fallbacks"] = [{"model": _DEFAULT_FALLBACK_MODEL}]
        params.update(request.extra)
        return params

    def complete(self, request: ModelRequest) -> ModelResponse:
        client = self._client_or_init()
        params = self._params(request)
        started = time.monotonic()
        # Fallbacks live on the beta endpoint; otherwise the GA endpoint is fine.
        endpoint = client.beta.messages if "fallbacks" in params else client.messages
        resp = endpoint.create(**params)
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return ModelResponse(
            text=text,
            model=getattr(resp, "model", self.model),
            stop_reason=getattr(resp, "stop_reason", "end_turn"),
            input_tokens=getattr(resp.usage, "input_tokens", 0),
            output_tokens=getattr(resp.usage, "output_tokens", 0),
            latency_s=time.monotonic() - started,
        )


def _effort(effort: Effort) -> str:
    # Our Effort enum values line up 1:1 with output_config.effort levels.
    return effort.value
