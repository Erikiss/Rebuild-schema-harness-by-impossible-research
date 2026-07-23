from pathlib import Path

from conftest import FIXTURES

from schema_repro.agent import LLMAgent
from schema_repro.config import load_config
from schema_repro.controller import Controller
from schema_repro.env.simulated import GridPursuit
from schema_repro.planner import BFSPlanner
from schema_repro.providers.fake import FakeProvider
from schema_repro.verifier import SchemaVerifier
from schema_repro.world_model import EditableProgram

PROGRAM = (FIXTURES / "example_world_model.py").read_text()


def test_controller_solves_gridpursuit_offline(tmp_path):
    """The full ground->verify->plan->execute loop wins with zero network by
    scripting a FakeProvider to 'author' the correct world-model program."""
    cfg = load_config(profile="fake")
    fake = FakeProvider(scripted=["ACTION2", f"```python\n{PROGRAM}\n```"])
    env = GridPursuit(cursor=(0, 0), target=(4, 5))
    ctrl = Controller(
        env=env,
        agent=LLMAgent(),
        verifier=SchemaVerifier(cfg.sandbox),
        planner=BFSPlanner(cfg.sandbox),
        provider=fake,
        config=cfg,
    )
    outcome = ctrl.run_level(EditableProgram(path=str(tmp_path / "wm.py")))
    assert outcome.solved is True
    # 1 exploratory action + Manhattan distance from (1,0) to (4,5) = 3 + 5 = 8.
    assert outcome.action_count == 9
    assert outcome.model_edits >= 1
