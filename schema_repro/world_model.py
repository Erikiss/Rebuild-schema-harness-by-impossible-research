"""The editable world-model interface at the heart of Schema.

Schema's core idea: don't ask the model to *play*, ask it to write and keep
editing one small Python program that jointly represents

  * state grounding    - raw 64x64 grid -> objects/variables/relations, and
  * mechanism discovery - how that state changes under each action,

plus a goal predicate and a planner over the resulting dynamics. Observations
from the live game are fed back in; the program is revised to fit them; each
revision is checked by verifiers before any further live actions are taken.

This module defines the *interface* the agent-authored program must satisfy and
a thin, sandbox-loaded wrapper around the program file on disk. The predefined
interface (not the game logic) is what the harness ships — the same interface is
reused across every game, with no game-specific code in the harness itself.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from schema_repro.types import Action, Frame, Grid

# The agent's grounded state is an opaque, JSON-serialisable object owned by the
# agent's program. The harness never inspects its internals — it only round-trips
# it through the program's own functions, which keeps the harness game-agnostic.
State = Any


@runtime_checkable
class WorldModel(Protocol):
    """Contract every agent-authored ``world_model.py`` program must implement.

    A concrete program is loaded from disk (see ``EditableProgram``) and must
    expose these callables at module scope.
    """

    def ground(self, frame: Frame) -> State:
        """Parse a raw frame into structured state (objects/variables/relations)."""

    def step(self, state: State, action: Action) -> State:
        """Predict the next state after ``action`` — the discovered mechanism."""

    def is_goal(self, state: State) -> bool:
        """Whether ``state`` satisfies the (inferred) success condition."""

    def render(self, state: State) -> Grid:
        """Project state back to a grid, so predictions can be checked pixel-wise."""

    def actions(self, state: State) -> list[Action]:
        """Legal actions the planner may consider from ``state``."""


@dataclass
class EditableProgram:
    """A version-tracked handle to the agent's single editable program.

    The program text is authored/edited by the model each turn. We keep the full
    revision history so the trace can show how the world model evolved, and so a
    verifier failure can roll back to the last good revision.
    """

    path: str
    source: str = ""
    revisions: list[str] = field(default_factory=list)

    def revision_id(self) -> str:
        return hashlib.sha256(self.source.encode()).hexdigest()[:12]

    def edit(self, new_source: str) -> str:
        """Record a new revision and return its id. Persisting/loading is done by
        the sandbox (see ``schema_repro.sandbox``) so this stays side-effect free
        and easy to unit-test."""
        self.revisions.append(self.source)
        self.source = new_source
        return self.revision_id()

    def rollback(self) -> None:
        if self.revisions:
            self.source = self.revisions.pop()


@dataclass
class Transition:
    """One recorded (frame, action, next_frame) triple from the live game.

    The world-model verifier replays these: for every transition, the program's
    ``step(ground(frame), action)`` rendered back to a grid must match the
    observed ``next_frame`` grid.
    """

    frame: Frame
    action: Action
    next_frame: Frame
