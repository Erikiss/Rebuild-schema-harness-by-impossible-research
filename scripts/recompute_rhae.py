"""Recompute RHAE from a directory of Schema traces.

Mirrors the published dataset's recompute command: discover every trajectory
directory, stream its ``events.jsonl``, reconstruct per-level action counts, join
the shared ``baseline_actions.csv``, recompute every RHAE score, and print a
per-level table followed by a per-collection summary.

Usage:
    python scripts/recompute_rhae.py <traces_root> [--baseline baseline_actions.csv]

If ``--baseline`` is omitted, ``<traces_root>/baseline_actions.csv`` is used.
Collections are the immediate subdirectories of the root (e.g. ``gpt_5_6_sol/``
and ``claude_fable_opus/``), scored separately.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from schema_repro.scoring.rhae import load_baselines, recompute_from_traces  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Recompute RHAE from Schema traces.")
    ap.add_argument("root", type=Path, help="directory containing trace collections")
    ap.add_argument("--baseline", type=Path, default=None, help="baseline_actions.csv")
    args = ap.parse_args(argv)

    baseline_csv = args.baseline or (args.root / "baseline_actions.csv")
    if not baseline_csv.exists():
        ap.error(f"baseline CSV not found: {baseline_csv}")

    baselines = load_baselines(baseline_csv)
    collections = sorted(p for p in args.root.iterdir() if p.is_dir())
    if not collections:
        collections = [args.root]

    print(f"{'collection':<20}{'game':<10}{'lvl':>4}{'agent':>8}{'base':>8}{'solved':>8}{'eff%':>8}")
    print("-" * 66)
    summary = {}
    for coll in collections:
        report = recompute_from_traces(coll, baseline_csv)
        for ls in report.levels:
            print(
                f"{coll.name:<20}{ls.game_id:<10}{ls.level:>4}{ls.agent_actions:>8}"
                f"{ls.baseline_actions:>8}{str(ls.solved):>8}{ls.efficiency * 100:>7.1f}"
            )
        summary[coll.name] = report.overall()

    print("-" * 66)
    print("RHAE by collection:")
    for name, overall in summary.items():
        print(f"  {name:<24}{overall * 100:6.2f}%")
    _ = baselines  # loaded above to fail fast on a malformed CSV
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
