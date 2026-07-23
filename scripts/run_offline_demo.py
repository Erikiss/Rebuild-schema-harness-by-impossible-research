"""Run the full harness loop offline on the GridPursuit toy game.

No network, no API key, no frontier model: a scripted FakeProvider "authors" the
correct world-model program, and the controller grounds -> verifies -> plans ->
executes its way to a WIN. This exercises every seam of the reconstructed harness
(controller, sandbox, verifier, planner, provider interface, tracing) so you can
see the structure run end to end before wiring in a real provider.

    python scripts/run_offline_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from schema_repro.agent import LLMAgent  # noqa: E402
from schema_repro.config import load_config  # noqa: E402
from schema_repro.controller import Controller  # noqa: E402
from schema_repro.env.simulated import GridPursuit  # noqa: E402
from schema_repro.planner import BFSPlanner  # noqa: E402
from schema_repro.providers.fake import FakeProvider  # noqa: E402
from schema_repro.tracing import TraceWriter  # noqa: E402
from schema_repro.verifier import SchemaVerifier  # noqa: E402
from schema_repro.world_model import EditableProgram  # noqa: E402

PROGRAM = (ROOT / "fixtures" / "example_world_model.py").read_text()
TRACE = ROOT / "fixtures" / "demo_run" / "events.jsonl"


def main() -> int:
    cfg = load_config(profile="fake")
    fake = FakeProvider(scripted=["ACTION2", f"```python\n{PROGRAM}\n```"])
    env = GridPursuit(cursor=(0, 0), target=(4, 5))
    with TraceWriter(TRACE) as trace:
        ctrl = Controller(
            env=env,
            agent=LLMAgent(),
            verifier=SchemaVerifier(cfg.sandbox),
            planner=BFSPlanner(cfg.sandbox),
            provider=fake,
            config=cfg,
            trace=trace,
        )
        outcome = ctrl.run_level(EditableProgram(path="wm.py"))

    print(f"game={outcome.game_id} solved={outcome.solved} "
          f"actions={outcome.action_count} model_edits={outcome.model_edits}")
    print(f"trace written to {TRACE}")
    return 0 if outcome.solved else 1


if __name__ == "__main__":
    raise SystemExit(main())
