"""Replay a recorded ``events.jsonl`` as an environment.

Useful for inspecting a trace or re-deriving per-level action counts through the
same ``Environment`` interface the live loop uses. Note: our compact trace stores
state/score per observation but not full grids (the published dataset keeps grids
in separate snapshot files), so replayed frames carry an empty grid unless the
trace was written with ``log_grids=True``. RHAE recompute does not need grids —
see ``schema_repro.scoring.rhae``.
"""

from __future__ import annotations

from schema_repro.tracing import read_events
from schema_repro.types import Action, Frame, GameState


class ReplayEnv:
    def __init__(self, events_path: str):
        self._events = [e for e in read_events(events_path) if e.type in ("observation", "level_start", "level_end")]
        self._i = 0
        self._game_id = next((e.game_id for e in self._events if e.game_id), "replay")
        self._level = 0

    @property
    def game_id(self) -> str:
        return self._game_id

    @property
    def level(self) -> int:
        return self._level

    def reset(self, game_id: str) -> Frame:  # noqa: ARG002
        self._i = 0
        return self._current()

    def step(self, action: Action) -> Frame:  # noqa: ARG002 - replay ignores the action
        self._i = min(self._i + 1, len(self._events) - 1)
        return self._current()

    def _current(self) -> Frame:
        if not self._events:
            return Frame(self._game_id, [], [], GameState.GAME_OVER, 0, [])
        ev = self._events[self._i]
        self._level = ev.level
        grid = ev.payload.get("grid", [])
        state = GameState(ev.payload.get("state", GameState.NOT_FINISHED.value))
        return Frame(
            game_id=ev.game_id or self._game_id,
            grid=grid,
            frame=[grid] if grid else [],
            state=state,
            score=int(ev.payload.get("score", 0)),
            available_actions=list(ev.payload.get("available_actions", [])),
            level=ev.level,
        )
