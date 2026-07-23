"""Streamed event log, shaped to match the published Schema traces.

The ``arc-agi-3-schema-traces`` dataset stores, per run, a ``run.json`` and a
streamed ``events.jsonl``; a recompute tool reconstructs per-level action counts
from those events and recomputes RHAE. We reconstruct a *compatible* event
schema so (a) our runs can be scored by the same recompute logic and (b) the
published traces can be replayed by our tooling. Field names not visible in the
public material are inferred and marked in ``docs/TRACE_FORMAT.md``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass
class Event:
    """One line of ``events.jsonl``.

    ``t`` is a monotonic step index (not wall-clock, so traces are
    deterministic to replay). ``type`` is one of a small closed vocabulary:
    ``run_start``, ``level_start``, ``action``, ``observation``, ``model_edit``,
    ``verify``, ``plan``, ``level_end``, ``run_end``.
    """

    t: int
    type: str
    game_id: str = ""
    level: int = 0
    payload: dict[str, Any] = field(default_factory=dict)


class TraceWriter:
    """Append-only JSONL writer used while a run executes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self._t = 0

    def emit(self, type: str, *, game_id: str = "", level: int = 0, **payload: Any) -> Event:
        ev = Event(t=self._t, type=type, game_id=game_id, level=level, payload=payload)
        self._fh.write(json.dumps(asdict(ev), separators=(",", ":")) + "\n")
        self._fh.flush()
        self._t += 1
        return ev

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_events(path: str | Path) -> Iterator[Event]:
    """Stream events back from a JSONL file (ours or a published trace)."""
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            yield Event(
                t=raw.get("t", 0),
                type=raw.get("type", ""),
                game_id=raw.get("game_id", ""),
                level=raw.get("level", 0),
                payload=raw.get("payload", {}),
            )


def action_counts_by_level(events: Iterable[Event]) -> dict[tuple[str, int], int]:
    """Reconstruct per-(game, level) action counts from an event stream.

    This is the bridge to RHAE: the official metric compares an agent's
    per-level action count against a first-exposure human baseline.
    """
    counts: dict[tuple[str, int], int] = {}
    for ev in events:
        if ev.type == "action":
            key = (ev.game_id, ev.level)
            counts[key] = counts.get(key, 0) + 1
    return counts
