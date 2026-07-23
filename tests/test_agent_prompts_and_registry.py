"""Regression tests for two bugs a review caught:

1. The reconstructed prompts must actually load (they lived in the wrong dir and
   the agent silently used its inline fallbacks).
2. The registry must enable the server-side refusal fallback for a single-stage
   always-thinking anthropic profile, and disable it when a profile chains to a
   second model (so the two fallback mechanisms never stack).
"""

from schema_repro.agent import _FALLBACK_EXPLORE, _FALLBACK_SYSTEM, PROMPTS_DIR, LLMAgent
from schema_repro.config import ModelProfile, ModelStage
from schema_repro.providers.anthropic_provider import AnthropicProvider, _is_always_thinking
from schema_repro.providers.base import FallbackProvider
from schema_repro.providers.registry import build_provider


def test_reconstructed_prompts_actually_load():
    assert (PROMPTS_DIR / "world_model_engineer.md").exists()
    assert (PROMPTS_DIR / "explorer.md").exists()
    agent = LLMAgent()
    assert agent.system_revise != _FALLBACK_SYSTEM
    assert agent.system_explore != _FALLBACK_EXPLORE


def test_always_thinking_is_prefix_matched():
    assert _is_always_thinking("claude-fable-5")
    assert _is_always_thinking("claude-fable-5-20260601")  # dated snapshot
    assert not _is_always_thinking("claude-opus-4-8")


def test_registry_enables_server_side_fallback_only_when_single_stage():
    single = ModelProfile("a", [ModelStage("anthropic", "claude-fable-5")], ["refusal"])
    p1 = build_provider(single)
    assert isinstance(p1, AnthropicProvider)
    assert p1.enable_refusal_fallback is True

    multi = ModelProfile(
        "a",
        [ModelStage("anthropic", "claude-fable-5"), ModelStage("anthropic", "claude-opus-4-8")],
        ["refusal"],
    )
    p2 = build_provider(multi)
    assert isinstance(p2, FallbackProvider)
    assert all(getattr(s, "enable_refusal_fallback", False) is False for s in p2.chain)
