"""Relative Human Action Efficiency (RHAE).

Official definition (ARC-AGI-3): RHAE compares an agent's per-level action count
with a first-exposure human baseline and aggregates across environments; 100%
means completing every level of every environment at or above human-baseline
action efficiency.

The exact clipping/aggregation is not fully specified in public material, so this
is a documented reconstruction that satisfies the stated properties:

* per level, if the agent did not solve the level -> efficiency 0;
* if solved with ``a`` actions against baseline ``h`` -> ``min(1, h / a)``
  (using at most the human budget scores 100%; using more scores proportionally
  less);
* per game (environment) -> mean over that game's levels;
* overall -> mean over games (macro-average across environments).

``recompute_from_traces`` reproduces the dataset's workflow: discover trajectory
directories, stream their ``events.jsonl``, reconstruct per-level action counts,
join the shared ``baseline_actions.csv``, and recompute every score.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from schema_repro.tracing import action_counts_by_level, read_events


@dataclass
class LevelScore:
    game_id: str
    level: int
    agent_actions: int
    baseline_actions: int
    solved: bool

    @property
    def efficiency(self) -> float:
        if not self.solved or self.agent_actions <= 0:
            return 0.0
        return min(1.0, self.baseline_actions / self.agent_actions)


@dataclass
class RhaeReport:
    levels: list[LevelScore] = field(default_factory=list)

    def per_game(self) -> dict[str, float]:
        by_game: dict[str, list[float]] = {}
        for ls in self.levels:
            by_game.setdefault(ls.game_id, []).append(ls.efficiency)
        return {g: sum(v) / len(v) for g, v in by_game.items() if v}

    def overall(self) -> float:
        pg = self.per_game()
        return sum(pg.values()) / len(pg) if pg else 0.0


def load_baselines(path: str | Path) -> dict[tuple[str, int], int]:
    """Read ``baseline_actions.csv`` -> {(game_id, level): baseline_actions}."""
    out: dict[tuple[str, int], int] = {}
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            out[(row["game_id"], int(row["level"]))] = int(row["baseline_actions"])
    return out


def _solved_levels(events) -> set[tuple[str, int]]:
    solved: set[tuple[str, int]] = set()
    for ev in events:
        if ev.type == "level_end" and ev.payload.get("solved"):
            solved.add((ev.game_id, ev.level))
    return solved


def score_events(events_path: str | Path, baselines: dict[tuple[str, int], int]) -> RhaeReport:
    """Score a single ``events.jsonl`` against the baselines."""
    events = list(read_events(events_path))
    counts = action_counts_by_level(events)
    solved = _solved_levels(events)
    report = RhaeReport()
    for key, agent_actions in sorted(counts.items()):
        game_id, level = key
        report.levels.append(
            LevelScore(
                game_id=game_id,
                level=level,
                agent_actions=agent_actions,
                baseline_actions=baselines.get(key, 0),
                solved=key in solved,
            )
        )
    return report


def recompute_from_traces(root: str | Path, baseline_csv: str | Path) -> RhaeReport:
    """Discover every ``events.jsonl`` under ``root`` and score them together.

    Mirrors the dataset's recompute command, which streams all traces,
    reconstructs per-level action counts, and recomputes every RHAE score.
    """
    baselines = load_baselines(baseline_csv)
    combined = RhaeReport()
    for events_path in sorted(Path(root).rglob("events.jsonl")):
        combined.levels.extend(score_events(events_path, baselines).levels)
    return combined
