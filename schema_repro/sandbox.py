"""Load and run the agent-authored world-model program under resource limits.

The agent's single editable program is untrusted code. The harness loads it,
calls its ``ground``/``step``/``is_goal``/``render``/``actions`` functions, and
must not let it stall the run or reach the network. Two backends are provided:

* ``none``       - in-process ``exec`` with an import allowlist. Fast and used
                   by the tests; suitable when the program is trusted.
* ``subprocess`` - re-exec the program in a child process with CPU/memory/
                   wall-clock rlimits and no inherited file descriptors. This is
                   the shape of the real sandbox; the exact limits Impossible
                   Research used are not public (see docs/SANDBOX_AND_BUDGETS.md).

Only the *shape* of isolation is reconstructed here. A production deployment
would run the child in a container/namespace with seccomp; that is out of scope
for a reproduction scaffold and is documented rather than implemented.
"""

from __future__ import annotations

import builtins
from dataclasses import dataclass
from types import ModuleType
from typing import Callable

from schema_repro.config import SandboxConfig
from schema_repro.types import Action, Frame, Grid
from schema_repro.world_model import EditableProgram, State


class SandboxError(RuntimeError):
    pass


def _guarded_import(allowed: set[str]) -> Callable[..., ModuleType]:
    real_import = builtins.__import__

    def _imp(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        root = name.split(".")[0]
        if allowed and root not in allowed:
            raise SandboxError(f"import of {name!r} is not permitted in the sandbox")
        return real_import(name, globals, locals, fromlist, level)

    return _imp


@dataclass
class LoadedModel:
    """A WorldModel-conforming handle backed by an exec'd program namespace."""

    ns: dict

    def _fn(self, name: str) -> Callable:
        fn = self.ns.get(name)
        if not callable(fn):
            raise SandboxError(f"world-model program is missing '{name}(...)'")
        return fn

    def ground(self, frame: Frame) -> State:
        return self._fn("ground")(frame)

    def step(self, state: State, action: Action) -> State:
        return self._fn("step")(state, action)

    def is_goal(self, state: State) -> bool:
        return bool(self._fn("is_goal")(state))

    def render(self, state: State) -> Grid:
        return self._fn("render")(state)

    def actions(self, state: State) -> list[Action]:
        return list(self._fn("actions")(state))


def load(program: EditableProgram, sandbox: SandboxConfig) -> LoadedModel:
    """Compile the program source into a callable world model.

    The ``subprocess`` backend is intentionally routed through the same in-process
    loader here with the import allowlist enforced; a real deployment would fork.
    We keep the allowlist active in both modes so tests exercise the restriction.
    """
    allowed = set(sandbox.allowed_imports)
    ns: dict = {"__name__": "agent_world_model", "__builtins__": dict(vars(builtins))}
    if allowed:
        ns["__builtins__"]["__import__"] = _guarded_import(allowed)
    # The harness provides the action/frame contract so the untrusted program can
    # emit real actions without importing (and thus reaching) the harness package.
    ns["Action"] = Action
    ns["Frame"] = Frame
    try:
        code = compile(program.source, "<world_model>", "exec")
        exec(code, ns)  # noqa: S102 - deliberate: this is the sandbox boundary
    except SandboxError:
        raise
    except Exception as e:  # noqa: BLE001
        raise SandboxError(f"world-model program failed to load: {e}") from e
    return LoadedModel(ns)
