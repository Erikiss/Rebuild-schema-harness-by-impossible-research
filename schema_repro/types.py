"""Core datatypes shared across the reproduction harness.

These mirror the *observable* surface of the Schema harness and the ARC-AGI-3
environment as reconstructed from public sources (the Schema project page, the
arXiv report 2605.05138, the ARC-AGI-3 technical report and SDK). They are a
clean-room reconstruction: nothing here is copied from the original, unreleased
implementation. See ``docs/REPRODUCTION_PLAN.md`` for what is known vs inferred.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

# A frame is a stack of 2D grids (ARC-AGI-3 delivers ``frame`` as a 3D list).
# Each cell is a colour index in ``0..15``.
Grid = list[list[int]]


class GameState(str, enum.Enum):
    """Lifecycle of an ARC-AGI-3 game/level, per the ARC-AGI-3 SDK."""

    NOT_PLAYED = "NOT_PLAYED"
    NOT_FINISHED = "NOT_FINISHED"
    WIN = "WIN"
    GAME_OVER = "GAME_OVER"


# The 7 standardized ARC-AGI-3 actions. ACTION1-5 are "simple"; ACTION6 is a
# "complex" action that carries ``{x, y}`` coordinates; RESET restarts a level.
RESET = "RESET"
ACTION1 = "ACTION1"
ACTION2 = "ACTION2"
ACTION3 = "ACTION3"
ACTION4 = "ACTION4"
ACTION5 = "ACTION5"
ACTION6 = "ACTION6"
SIMPLE_ACTIONS = (ACTION1, ACTION2, ACTION3, ACTION4, ACTION5)
ALL_ACTIONS = (RESET, *SIMPLE_ACTIONS, ACTION6)


@dataclass(frozen=True)
class Action:
    """A single action to submit to the environment.

    ``data`` carries coordinates for the complex action, e.g.
    ``Action(ACTION6, {"x": 32, "y": 32})``.
    """

    name: str
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in ALL_ACTIONS:
            raise ValueError(f"unknown action {self.name!r}")
        if self.name == ACTION6 and not {"x", "y"} <= set(self.data):
            raise ValueError("ACTION6 requires 'x' and 'y' in data")


@dataclass(frozen=True)
class Frame:
    """One observation returned by the environment after an action."""

    game_id: str
    grid: Grid  # top 2D grid of the returned stack, for convenience
    frame: list[Grid]  # full 3D stack as returned by the SDK
    state: GameState
    score: int  # 0..254
    available_actions: list[str]
    level: int = 0


class Phase(str, enum.Enum):
    """The controller's fixed phase cycle.

    Schema formalizes play as *state grounding* + *mechanism discovery* over a
    single editable program, then *verify -> refactor -> plan -> execute*. We
    keep the phases explicit so reasoning-effort budgets can be assigned per
    phase (planning/verification get more compute than execution).
    """

    GROUND = "ground"  # raw frame -> objects/variables/relations
    DISCOVER = "discover"  # infer transition rules; edit the world-model program
    VERIFY = "verify"  # world-model + planner verifiers must pass
    REFACTOR = "refactor"  # simplify toward an MDL-like bias
    PLAN = "plan"  # search inside the model for an action sequence
    EXECUTE = "execute"  # commit planned actions to the live game


class Effort(str, enum.Enum):
    """Reasoning-effort tiers exposed by frontier providers."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


@dataclass
class StepResult:
    """Outcome of one controller iteration, for tracing and RHAE accounting."""

    phase: Phase
    action: Action | None
    frame: Frame | None
    model_edits: int = 0
    verifier_passed: bool | None = None
    notes: str = ""
