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
    # A single-stage anthropic profile relies on the *server-side* refusal
    # fallback (Fable 5 -> Opus 4.8, inside one API call); a multi-stage profile
    # gets a client-side FallbackProvider chain instead, so the server-side
    # fallback is disabled on its stages to avoid stacking the two.
    single_stage = len(profile.chain) == 1
    stages = [_build_stage(s.provider, s.model, server_side_fallback=single_stage) for s in profile.chain]
    if len(stages) == 1:
        return stages[0]
    return FallbackProvider(chain=stages, name=f"fallback:{profile.name}")


def _build_stage(provider: str, model: str, *, server_side_fallback: bool) -> Provider:
    if provider == "fake":
        from schema_repro.providers.fake import FakeProvider

        return FakeProvider(model=model or "fake-echo")
    if provider == "anthropic":
        from schema_repro.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(model, enable_refusal_fallback=server_side_fallback)
    if provider == "openai":
        from schema_repro.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(model or "gpt-5.6-sol")
    raise ValueError(f"unknown provider {provider!r}")
