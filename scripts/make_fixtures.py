"""Generate tiny synthetic traces that mirror the published dataset layout.

Produces, under ``fixtures/traces/``:

    baseline_actions.csv                 # shared human baselines
    gpt_5_6_sol/<game>/events.jsonl      # one collection
    claude_fable_opus/<game>/events.jsonl # the other collection

The event schema matches ``schema_repro.tracing`` so RHAE recompute runs offline.
These are illustrative, not real ARC-AGI-3 data.
"""

from __future__ import annotations

import csv
from pathlib import Path

from schema_repro.tracing import TraceWriter

ROOT = Path(__file__).resolve().parent.parent / "fixtures" / "traces"

# (game_id, level): (baseline_actions, agent_actions, solved)
PLAN = {
    "gpt_5_6_sol": {
        ("ls20", 0): (10, 9, True),
        ("ls20", 1): (14, 20, True),
        ("ft09", 0): (8, 8, True),
        ("ft09", 1): (12, 30, False),
    },
    "claude_fable_opus": {
        ("ls20", 0): (10, 8, True),
        ("ls20", 1): (14, 12, True),
        ("ft09", 0): (8, 7, True),
        ("ft09", 1): (12, 12, True),
    },
}
BASELINES = {("ls20", 0): 10, ("ls20", 1): 14, ("ft09", 0): 8, ("ft09", 1): 12}


def write_baselines() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "baseline_actions.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["game_id", "level", "baseline_actions"])
        for (game, level), base in sorted(BASELINES.items()):
            w.writerow([game, level, base])


def write_collection(name: str, plan: dict) -> None:
    by_game: dict[str, list] = {}
    for (game, level), spec in plan.items():
        by_game.setdefault(game, []).append((level, spec))
    for game, levels in by_game.items():
        out = ROOT / name / game / "events.jsonl"
        with TraceWriter(out) as tw:
            tw.emit("run_start", game_id=game, model=name)
            for level, (_base, agent_actions, solved) in sorted(levels):
                tw.emit("level_start", game_id=game, level=level)
                for i in range(agent_actions):
                    tw.emit("action", game_id=game, level=level, action="ACTION1", data={})
                    tw.emit("observation", game_id=game, level=level, state="NOT_FINISHED", score=i)
                tw.emit("level_end", game_id=game, level=level, solved=solved, action_count=agent_actions)
            tw.emit("run_end", game_id=game)


def main() -> None:
    write_baselines()
    for name, plan in PLAN.items():
        write_collection(name, plan)
    print(f"wrote fixtures under {ROOT}")


if __name__ == "__main__":
    main()
