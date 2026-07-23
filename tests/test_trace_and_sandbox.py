import pytest

from schema_repro.config import SandboxConfig
from schema_repro.sandbox import SandboxError, load
from schema_repro.tracing import TraceWriter, action_counts_by_level, read_events
from schema_repro.world_model import EditableProgram


def test_trace_roundtrip(tmp_path):
    path = tmp_path / "events.jsonl"
    with TraceWriter(path) as tw:
        tw.emit("run_start", game_id="g")
        tw.emit("level_start", game_id="g", level=0)
        tw.emit("action", game_id="g", level=0, action="ACTION1")
        tw.emit("action", game_id="g", level=0, action="ACTION2")
        tw.emit("level_end", game_id="g", level=0, solved=True)
    events = list(read_events(path))
    assert [e.type for e in events][:2] == ["run_start", "level_start"]
    assert action_counts_by_level(events) == {("g", 0): 2}


def test_sandbox_blocks_disallowed_import():
    sb = SandboxConfig(backend="none", allowed_imports=["math"])
    prog = EditableProgram(path="wm.py", source="import os\n")
    with pytest.raises(SandboxError):
        load(prog, sb)


def test_sandbox_allows_listed_import():
    sb = SandboxConfig(backend="none", allowed_imports=["math"])
    prog = EditableProgram(path="wm.py", source="import math\nVALUE = math.floor(3.7)\n")
    model = load(prog, sb)
    assert model.ns["VALUE"] == 3
