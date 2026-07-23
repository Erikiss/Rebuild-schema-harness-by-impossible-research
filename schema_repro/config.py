"""Load and resolve the YAML configuration into typed objects.

Keeps a single source of truth for sandbox/time/token/tool budgets and the
model fallback chain, so both the live loop and the offline replay/scoring paths
read the same knobs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # PyYAML is the only config dependency; degrade gracefully if absent.
    import yaml
except Exception:  # pragma: no cover - exercised only without the dep installed
    yaml = None  # type: ignore[assignment]

from schema_repro.types import Effort, Phase

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _load_yaml(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text) or {}
    raise RuntimeError(
        "PyYAML is required to parse configuration; `pip install pyyaml` "
        "or pass an already-parsed dict."
    )


@dataclass
class SandboxConfig:
    backend: str = "subprocess"
    network: bool = False
    wall_clock_s: float = 5.0
    memory_mb: int = 512
    cpu_seconds: float = 5.0
    allowed_imports: list[str] = field(default_factory=list)


@dataclass
class Budgets:
    max_live_actions_per_level: int = 300
    max_model_edits_per_level: int = 40
    max_plan_depth: int = 64
    max_plan_nodes: int = 200_000
    token_budget_per_level: int = 2_000_000
    verify_min_transitions: int = 1


@dataclass
class ModelStage:
    provider: str
    model: str


@dataclass
class ModelProfile:
    name: str
    chain: list[ModelStage]
    fallback_on: list[str]


@dataclass
class RunConfig:
    sandbox: SandboxConfig
    budgets: Budgets
    tools: list[dict[str, str]]
    env_provider: str
    games: list[str]
    seed: int
    effort_by_phase: dict[Phase, Effort]
    profile: ModelProfile

    def effort(self, phase: Phase) -> Effort:
        return self.effort_by_phase.get(phase, Effort.HIGH)


def _resolve_stage(stage: dict[str, Any]) -> ModelStage:
    model = stage.get("model_default", "")
    env = stage.get("model_env")
    if env:
        model = os.environ.get(env, model)
    return ModelStage(provider=stage["provider"], model=model)


def load_config(
    run_yaml: str | Path | None = None,
    models_yaml: str | Path | None = None,
    profile: str | None = None,
) -> RunConfig:
    run_yaml = run_yaml or CONFIGS_DIR / "default.yaml"
    models_yaml = models_yaml or CONFIGS_DIR / "models.yaml"
    run = _load_yaml(run_yaml)
    models = _load_yaml(models_yaml)

    profile_name = profile or run.get("env", {}).get("profile") or models["default_profile"]
    raw_profile = models["profiles"][profile_name]
    model_profile = ModelProfile(
        name=profile_name,
        chain=[_resolve_stage(s) for s in raw_profile["chain"]],
        fallback_on=list(raw_profile.get("fallback_on", ["error"])),
    )

    effort_by_phase = {
        Phase(k): Effort(v) for k, v in models.get("effort_by_phase", {}).items()
    }

    sb = run.get("sandbox", {})
    bud = run.get("budgets", {})
    env = run.get("env", {})
    return RunConfig(
        sandbox=SandboxConfig(**{k: sb[k] for k in sb if k in SandboxConfig.__annotations__}),
        budgets=Budgets(**{k: bud[k] for k in bud if k in Budgets.__annotations__}),
        tools=list(run.get("tools", [])),
        env_provider=env.get("provider", "replay"),
        games=list(env.get("games", [])),
        seed=int(env.get("seed", 0)),
        effort_by_phase=effort_by_phase,
        profile=model_profile,
    )
