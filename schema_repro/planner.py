"""Plan inside the learned model before acting on the live game.

Schema plans *through* the world model: it searches over the program's own
``actions``/``step``/``is_goal`` to find a short action sequence that reaches a
goal state, then executes that sequence against the real environment. Because the
model is only trusted after the verifiers pass, planning on it is meaningful.

We use a breadth-first search (shortest action sequence, which aligns with RHAE's
preference for fewer actions). Any admissible search would do; the harness is
game-agnostic, so the planner never looks at raw grids — only at model states.
"""

from __future__ import annotations

from collections import deque

from schema_repro.config import SandboxConfig
from schema_repro.sandbox import SandboxError, load
from schema_repro.types import Action, Frame
from schema_repro.verifier import _key
from schema_repro.world_model import EditableProgram


class BFSPlanner:
    def __init__(self, sandbox: SandboxConfig, *, max_depth: int = 64, max_nodes: int = 200_000):
        self.sandbox = sandbox
        self.max_depth = max_depth
        self.max_nodes = max_nodes

    def plan(self, program: EditableProgram, frame: Frame) -> list[Action]:
        try:
            model = load(program, self.sandbox)
            start = model.ground(frame)
        except SandboxError:
            return []
        except Exception:  # noqa: BLE001
            return []

        seen: set = {_key(start)}
        queue: deque[tuple[object, list[Action]]] = deque([(start, [])])
        nodes = 0
        while queue and nodes < self.max_nodes:
            state, path = queue.popleft()
            nodes += 1
            try:
                if model.is_goal(state):
                    return path
                if len(path) >= self.max_depth:
                    continue
                for action in model.actions(state):
                    nxt = model.step(state, action)
                    key = _key(nxt)
                    if key in seen:
                        continue
                    seen.add(key)
                    queue.append((nxt, path + [action]))
            except Exception:  # noqa: BLE001 - a crashing model yields no plan
                return []
        return []
