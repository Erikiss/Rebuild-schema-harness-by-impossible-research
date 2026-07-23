"""Build a concrete ``Provider`` from a configured model profile.

Turns the ``models.yaml`` chain into a live ``FallbackProvider``. The ``fake``
provider is always constructible (no dependencies); ``anthropic`` / ``openai``
providers import their SDKs lazily, so a profile referencing them only fails if
you actually run it without the dependency/credentials.
"""

from __future__ import annotations

from schema_repro.config import ModelProfile
from schema_repro.providers.base import FallbackProvider, Provider


def build_provider(profile: ModelProfile) -> Provider:
    stages = []
    for stage in profile.chain:
        stages.append(_build_stage(stage.provider, stage.model))
    if len(stages) == 1:
        return stages[0]
    return FallbackProvider(chain=stages, name=f"fallback:{profile.name}")


def _build_stage(provider: str, model: str) -> Provider:
    if provider == "fake":
        from schema_repro.providers.fake import FakeProvider

        return FakeProvider(model=model or "fake-echo")
    if provider == "anthropic":
        from schema_repro.providers.anthropic_provider import AnthropicProvider

        # Disable the harness-level refusal fallback when the profile already
        # chains to a second model; the two would otherwise stack.
        return AnthropicProvider(model, enable_refusal_fallback=False)
    if provider == "openai":
        from schema_repro.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(model or "gpt-5.6-sol")
    raise ValueError(f"unknown provider {provider!r}")
