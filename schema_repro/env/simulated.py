"""A tiny, deterministic ARC-AGI-3-shaped game for end-to-end testing.

`GridPursuit` is NOT an ARC-AGI-3 game and is not meant to be one. It is the
smallest environment that exercises the full harness plumbing — grounding,
mechanism discovery, verification, planning, execution, tracing and RHAE — with
zero network and no frontier model. A cursor (colour 2) must reach a target
(colour 3) using the four move actions; the level is WON when they coincide.

The matching *correct* world-model program lives in
``fixtures/example_world_model.py``; the offline end-to-end test scripts a
``FakeProvider`` to "author" that program, so a full run is reproducible.
"""

from __future__ import annotations

from schema_repro.types import (
    ACTION1,
    ACTION2,
    ACTION3,
    ACTION4,
    Action,
    Frame,
    GameState,
    Grid,
)

_MOVES = {ACTION1: (-1, 0), ACTION2: (1, 0), ACTION3: (0, -1), ACTION4: (0, 1)}


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def render_grid(h: int, w: int, cursor: tuple[int, int], target: tuple[int, int]) -> Grid:
    grid = [[0 for _ in range(w)] for _ in range(h)]
    tr, tc = target
    grid[tr][tc] = 3
    cr, cc = cursor
    grid[cr][cc] = 2  # cursor wins the overlap at the goal
    return grid


class GridPursuit:
    def __init__(self, *, h: int = 6, w: int = 6, cursor=(0, 0), target=(4, 5), game_id="grid_pursuit"):
        self.h, self.w = h, w
        self.start_cursor = cursor
        self.target = target
        self._game_id = game_id
        self.cursor = cursor
        self._level = 0

    @property
    def game_id(self) -> str:
        return self._game_id

    @property
    def level(self) -> int:
        return self._level

    def _frame(self, state: GameState) -> Frame:
        grid = render_grid(self.h, self.w, self.cursor, self.target)
        return Frame(
            game_id=self._game_id,
            grid=grid,
            frame=[grid],
            state=state,
            score=1 if state == GameState.WIN else 0,
            available_actions=[ACTION1, ACTION2, ACTION3, ACTION4],
            level=self._level,
        )

    def reset(self, game_id: str) -> Frame:  # noqa: ARG002 - single game here
        self.cursor = self.start_cursor
        state = GameState.WIN if self.cursor == self.target else GameState.NOT_FINISHED
        return self._frame(state)

    def step(self, action: Action) -> Frame:
        dr, dc = _MOVES.get(action.name, (0, 0))
        self.cursor = (_clamp(self.cursor[0] + dr, 0, self.h - 1), _clamp(self.cursor[1] + dc, 0, self.w - 1))
        state = GameState.WIN if self.cursor == self.target else GameState.NOT_FINISHED
        return self._frame(state)
