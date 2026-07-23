"""The scripted, game-agnostic controller.

This is the "harness" proper: a fixed control loop that never contains
game-specific code. It coordinates the injected collaborators — an
``Environment`` (live ARC-AGI-3 or an offline trace replay), an ``Agent`` (the
LLM policy that authors and edits the world-model program), a ``Verifier`` and a
``Planner`` — and assigns per-phase reasoning-effort budgets.

Per level the loop runs the Schema cycle:

    ground/discover -> verify -> refactor -> plan -> execute
          ^                                              |
          +------------------ surprise -------------------+

A "surprise" (the live frame disagreeing with the model's prediction) sends
control back to discovery, so the world model is only trusted after its
predictions have been checked against real observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from schema_repro.config import RunConfig
from schema_repro.providers.base import Provider
from schema_repro.tracing import TraceWriter
from schema_repro.types import Action, Frame, GameState, Phase, StepResult
from schema_repro.world_model import EditableProgram, Transition


class Environment(Protocol):
    """Live ARC-AGI-3 API or an offline replay of published traces."""

    def reset(self, game_id: str) -> Frame: ...
    def step(self, action: Action) -> Frame: ...
    @property
    def game_id(self) -> str: ...
    @property
    def level(self) -> int: ...


class Agent(Protocol):
    """The LLM policy. It edits the world-model program and, when the model is
    still too weak to plan, proposes a single exploratory action."""

    def revise_world_model(
        self, program: EditableProgram, history: list[Transition], provider: Provider, effort
    ) -> str: ...

    def explore(self, frame: Frame, history: list[Transition], provider: Provider, effort) -> Action: ...


class Verifier(Protocol):
    def check_world_model(self, program: EditableProgram, history: list[Transition]) -> bool: ...
    def check_planner(self, program: EditableProgram, frame: Frame) -> bool: ...


class Planner(Protocol):
    def plan(self, program: EditableProgram, frame: Frame) -> list[Action]: ...


@dataclass
class LevelOutcome:
    game_id: str
    level: int
    solved: bool
    action_count: int
    model_edits: int


class Controller:
    def __init__(
        self,
        *,
        env: Environment,
        agent: Agent,
        verifier: Verifier,
        planner: Planner,
        provider: Provider,
        config: RunConfig,
        trace: TraceWriter | None = None,
    ) -> None:
        self.env = env
        self.agent = agent
        self.verifier = verifier
        self.planner = planner
        self.provider = provider
        self.cfg = config
        self.trace = trace

    # -- helpers ----------------------------------------------------------

    def _emit(self, type: str, **payload) -> None:
        if self.trace is not None:
            self.trace.emit(type, game_id=self.env.game_id, level=self.env.level, **payload)

    # -- main loop --------------------------------------------------------

    def run_level(self, program: EditableProgram) -> LevelOutcome:
        """Play the current level to completion, budget exhaustion, or game over."""
        b = self.cfg.budgets
        history: list[Transition] = []
        frame = self.env.reset(self.env.game_id)
        self._emit("level_start", score=frame.score)

        actions_used = 0
        edits = 0

        while frame.state == GameState.NOT_FINISHED and actions_used < b.max_live_actions_per_level:
            # --- ground/discover: revise the single editable program ---------
            if len(history) >= b.verify_min_transitions and edits < b.max_model_edits_per_level:
                self.agent.revise_world_model(
                    program, history, self.provider, self.cfg.effort(Phase.DISCOVER)
                )
                edits += 1
                self._emit("model_edit", revision=program.revision_id(), edits=edits)

                # --- verify: both verifiers must pass, else roll back --------
                ok = self.verifier.check_world_model(program, history) and (
                    self.verifier.check_planner(program, frame)
                )
                self._emit("verify", passed=ok, revision=program.revision_id())
                if not ok:
                    program.rollback()

            # --- plan: search inside the model, else explore ----------------
            plan: list[Action] = []
            if self.verifier.check_world_model(program, history):
                plan = self.planner.plan(program, frame)[: b.max_plan_depth]
                self._emit("plan", length=len(plan))

            step_actions = plan if plan else [
                self.agent.explore(frame, history, self.provider, self.cfg.effort(Phase.EXECUTE))
            ]

            # --- execute: commit actions; a surprise re-opens discovery ------
            for action in step_actions:
                prev = frame
                frame = self.env.step(action)
                actions_used += 1
                history.append(Transition(prev, action, frame))
                self._emit("action", action=action.name, data=action.data)
                self._emit("observation", state=frame.state.value, score=frame.score)
                if frame.state != GameState.NOT_FINISHED or actions_used >= b.max_live_actions_per_level:
                    break
                # Surprise check: did reality diverge from the plan's expectation?
                if self._surprised(program, prev, action, frame):
                    break  # abandon the rest of the plan; re-discover

        solved = frame.state == GameState.WIN
        self._emit("level_end", solved=solved, action_count=actions_used)
        return LevelOutcome(self.env.game_id, self.env.level, solved, actions_used, edits)

    def _surprised(self, program: EditableProgram, prev: Frame, action: Action, obs: Frame) -> bool:
        """True if the model's prediction for this step disagrees with reality.

        Delegated to the verifier's single-transition check via a one-item
        history; kept here so the loop reads top-to-bottom.
        """
        return not self.verifier.check_world_model(program, [Transition(prev, action, obs)])
