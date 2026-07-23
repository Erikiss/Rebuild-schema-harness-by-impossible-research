"""Formal verifiers gate every world-model revision.

Two checks, mirroring the public description of Schema's workspace:

* ``check_world_model`` - the world-model verifier: the program must *reproduce
  recorded observations*. For each recorded transition ``(frame, action,
  next_frame)`` we require ``render(step(ground(frame), action))`` to equal the
  observed ``next_frame`` grid.
* ``check_planner`` - the planner verifier: a bounded search inside the learned
  model must be able to reach a goal state, i.e. the planner *can* produce a plan
  that completes inside the model. (Whether that plan works in the real game is
  decided later, by execution + the surprise check.)

Neither verifier contains game-specific logic; both operate purely through the
program's own interface.
"""

from __future__ import annotations

from schema_repro.config import SandboxConfig
from schema_repro.sandbox import SandboxError, load
from schema_repro.types import Frame
from schema_repro.world_model import EditableProgram, Transition


def _grids_equal(a, b) -> bool:
    return [list(row) for row in a] == [list(row) for row in b]


class SchemaVerifier:
    def __init__(self, sandbox: SandboxConfig, *, plan_node_cap: int = 20_000, plan_depth_cap: int = 64):
        self.sandbox = sandbox
        self.plan_node_cap = plan_node_cap
        self.plan_depth_cap = plan_depth_cap

    def check_world_model(self, program: EditableProgram, history: list[Transition]) -> bool:
        if not program.source.strip():
            return False
        try:
            model = load(program, self.sandbox)
        except SandboxError:
            return False
        for tr in history:
            try:
                predicted = model.render(model.step(model.ground(tr.frame), tr.action))
            except Exception:  # noqa: BLE001 - any program crash is a verification failure
                return False
            if not _grids_equal(predicted, tr.next_frame.grid):
                return False
        return True

    def check_planner(self, program: EditableProgram, frame: Frame) -> bool:
        """Bounded reachability: is any goal state reachable from ``frame`` inside
        the model? Uses the program's own ``actions``/``step``/``is_goal``."""
        try:
            model = load(program, self.sandbox)
            start = model.ground(frame)
        except SandboxError:
            return False
        except Exception:  # noqa: BLE001
            return False

        seen: set = set()
        frontier = [(start, 0)]
        nodes = 0
        while frontier and nodes < self.plan_node_cap:
            state, depth = frontier.pop()
            nodes += 1
            try:
                if model.is_goal(state):
                    return True
                if depth >= self.plan_depth_cap:
                    continue
                key = _key(state)
                if key in seen:
                    continue
                seen.add(key)
                for action in model.actions(state):
                    frontier.append((model.step(state, action), depth + 1))
            except Exception:  # noqa: BLE001
                return False
        return False


def _key(state) -> object:
    """A hashable identity for a state, tolerant of unhashable containers."""
    try:
        hash(state)
        return state
    except TypeError:
        return repr(state)
