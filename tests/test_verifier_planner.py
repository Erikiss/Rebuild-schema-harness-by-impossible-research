from pathlib import Path

from conftest import FIXTURES

from schema_repro.config import SandboxConfig
from schema_repro.env.simulated import GridPursuit
from schema_repro.planner import BFSPlanner
from schema_repro.types import ACTION4, Action, GameState
from schema_repro.verifier import SchemaVerifier
from schema_repro.world_model import EditableProgram, Transition

SANDBOX = SandboxConfig(backend="none", allowed_imports=[])
PROGRAM = (FIXTURES / "example_world_model.py").read_text()


def _one_transition():
    env = GridPursuit(cursor=(2, 2), target=(4, 5))
    f0 = env.reset("grid_pursuit")
    f1 = env.step(Action(ACTION4))  # move right
    return Transition(f0, Action(ACTION4), f1)


def test_world_model_verifier_accepts_correct_program():
    v = SchemaVerifier(SANDBOX)
    prog = EditableProgram(path="wm.py", source=PROGRAM)
    assert v.check_world_model(prog, [_one_transition()]) is True


def test_world_model_verifier_rejects_broken_program():
    v = SchemaVerifier(SANDBOX)
    broken = PROGRAM.replace('"ACTION4": (0, 1)', '"ACTION4": (0, -1)')  # wrong dynamics
    prog = EditableProgram(path="wm.py", source=broken)
    assert v.check_world_model(prog, [_one_transition()]) is False


def test_planner_finds_shortest_path():
    env = GridPursuit(cursor=(0, 0), target=(0, 3))
    frame = env.reset("grid_pursuit")
    planner = BFSPlanner(SANDBOX)
    prog = EditableProgram(path="wm.py", source=PROGRAM)
    plan = planner.plan(prog, frame)
    assert len(plan) == 3
    assert all(a.name == ACTION4 for a in plan)  # three steps right


def test_planner_verifier_reachability():
    env = GridPursuit(cursor=(0, 0), target=(2, 2))
    frame = env.reset("grid_pursuit")
    v = SchemaVerifier(SANDBOX, plan_depth_cap=16)
    prog = EditableProgram(path="wm.py", source=PROGRAM)
    assert v.check_planner(prog, frame) is True
