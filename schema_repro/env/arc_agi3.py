"""Live ARC-AGI-3 environment adapter (best-effort).

Wraps the public ``arc-agi-3`` client behind the harness's ``Environment``
protocol. The exact SDK surface is external and may drift; the import is lazy and
the mapping is a documented reconstruction from the ARC-AGI-3 docs (64x64 grid of
16 colours; RESET + ACTION1..6; states NOT_PLAYED/NOT_FINISHED/WIN/GAME_OVER;
ACTION6 carries x,y). RHAE is computed offline from traces, so this adapter is
only needed for live evaluation, not for the tests.
"""

from __future__ import annotations

from schema_repro.types import ACTION6, Action, Frame, GameState


class ArcAgi3Env:
    def __init__(self, game_id: str, *, api_key: str | None = None):
        self._game_id = game_id
        self._level = 0
        self._api_key = api_key
        self._client = None
        self._card = None  # scorecard/session handle from the SDK

    def _client_or_init(self):
        if self._client is None:
            import arc_agi_3  # deferred; only needed for live runs

            self._client = arc_agi_3.Client(api_key=self._api_key)
        return self._client

    @property
    def game_id(self) -> str:
        return self._game_id

    @property
    def level(self) -> int:
        return self._level

    def reset(self, game_id: str) -> Frame:
        client = self._client_or_init()
        self._card = client.reset(game_id=game_id)
        return self._to_frame(self._card)

    def step(self, action: Action) -> Frame:
        client = self._client_or_init()
        payload = dict(action.data) if action.name == ACTION6 else None
        self._card = client.act(self._game_id, action=action.name, data=payload)
        return self._to_frame(self._card)

    def _to_frame(self, raw) -> Frame:
        # ``raw`` is the SDK's frame object/dict; read defensively.
        get = raw.get if isinstance(raw, dict) else lambda k, d=None: getattr(raw, k, d)
        frame_stack = get("frame", []) or []
        grid = frame_stack[0] if frame_stack else []
        self._level = int(get("level", self._level) or self._level)
        return Frame(
            game_id=get("game_id", self._game_id) or self._game_id,
            grid=grid,
            frame=frame_stack,
            state=GameState(get("state", GameState.NOT_FINISHED.value)),
            score=int(get("score", 0) or 0),
            available_actions=list(get("available_actions", []) or []),
            level=self._level,
        )
