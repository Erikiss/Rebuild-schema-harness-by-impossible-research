from pathlib import Path

from conftest import FIXTURES

from schema_repro.scoring.rhae import LevelScore, recompute_from_traces, score_events


def test_level_efficiency_math():
    # Solved under budget -> capped at 1.0.
    assert LevelScore("g", 0, agent_actions=8, baseline_actions=10, solved=True).efficiency == 1.0
    # Solved over budget -> proportional.
    ls = LevelScore("g", 1, agent_actions=20, baseline_actions=10, solved=True)
    assert abs(ls.efficiency - 0.5) < 1e-9
    # Unsolved -> zero regardless of actions.
    assert LevelScore("g", 2, agent_actions=5, baseline_actions=10, solved=False).efficiency == 0.0


def test_recompute_over_fixture_traces():
    root = FIXTURES / "traces"
    if not (root / "baseline_actions.csv").exists():
        import scripts.make_fixtures as mk  # generate on demand

        mk.main()
    report = recompute_from_traces(root, root / "baseline_actions.csv")
    assert len(report.levels) == 8
    overall = report.overall()
    assert 0.0 <= overall <= 1.0
    # The claude collection was scripted to be at/under baseline on every level.
    claude = score_events(root / "claude_fable_opus" / "ls20" / "events.jsonl", _baselines(root))
    assert all(ls.efficiency == 1.0 for ls in claude.levels)


def _baselines(root: Path):
    from schema_repro.scoring.rhae import load_baselines

    return load_baselines(root / "baseline_actions.csv")
