"""Model-provider abstraction with an explicit fallback chain.

Schema is model-agnostic: the *same* harness reached ~99% RHAE with Opus 4.8 +
Fable 5 and 95.35% with GPT-5.6 Sol. We therefore isolate all model access
behind a small ``Provider`` protocol and put the "Fable-5 fallback logic" into a
transparent, configurable ``FallbackProvider`` rather than hiding it inside a
single vendor client.

Why a fallback chain matters here (and why it can only be *approximated*):
Anthropic's Fable 5 ships with always-on adaptive thinking and a refusal
``stop_reason`` produced by a two-stage classifier; on some surfaces a refusal
triggers an automatic hand-off to Opus 4.8. That provider-internal behaviour is
not something an external harness can reproduce byte-for-byte, so we model the
*contract* (an ordered chain + trigger predicates) and leave the trigger set
configurable. See ``docs/MODEL_AND_FALLBACK.md``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from schema_repro.types import Effort


@dataclass
class ModelRequest:
    system: str
    messages: list[dict[str, str]]
    effort: Effort = Effort.HIGH
    max_tokens: int = 8192
    # Free-form knobs a concrete provider may honour (temperature, tools, ...).
    extra: dict[str, object] = field(default_factory=dict)


@dataclass
class ModelResponse:
    text: str
    model: str
    stop_reason: str = "end_turn"  # "end_turn" | "max_tokens" | "refusal" | ...
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"


@runtime_checkable
class Provider(Protocol):
    """Anything that can turn a request into a completion."""

    name: str

    def complete(self, request: ModelRequest) -> ModelResponse: ...


# A trigger decides whether the *next* provider in the chain should be tried.
def refused_or_errored(resp: ModelResponse | None, exc: Exception | None) -> bool:
    return exc is not None or (resp is not None and resp.refused)


@dataclass
class FallbackProvider:
    """Try providers in order until a trigger stops firing.

    ``chain`` is ordered primary-first. ``should_fallback`` inspects each
    (response, exception) pair and returns ``True`` to advance to the next
    provider. The default advances on any error or refusal — a faithful
    reconstruction of the *shape* of the Fable-5 -> Opus-4.8 safety fallback,
    not its proprietary internals.
    """

    chain: list[Provider]
    should_fallback: object = refused_or_errored  # Callable[[resp, exc], bool]
    max_attempts: int | None = None
    name: str = "fallback"

    def __post_init__(self) -> None:
        if not self.chain:
            raise ValueError("FallbackProvider needs at least one provider")

    def complete(self, request: ModelRequest) -> ModelResponse:
        limit = self.max_attempts or len(self.chain)
        last: ModelResponse | None = None
        for provider in self.chain[:limit]:
            resp: ModelResponse | None = None
            exc: Exception | None = None
            started = time.monotonic()
            try:
                resp = provider.complete(request)
            except Exception as e:  # noqa: BLE001 - deliberately broad at the boundary
                exc = e
            if resp is not None:
                resp.latency_s = resp.latency_s or (time.monotonic() - started)
                last = resp
            if not self.should_fallback(resp, exc):  # type: ignore[operator]
                if resp is None:  # trigger said stop but we have no response
                    raise RuntimeError("fallback stopped without a response") from exc
                return resp
        if last is not None:
            return last  # exhausted chain; return best-effort last response
        raise RuntimeError("all providers in the fallback chain failed")
