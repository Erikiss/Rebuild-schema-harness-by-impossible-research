"""The LLM policy: authors and edits the single world-model program.

The agent is deliberately thin. It does not "play" the game; it (a) proposes
edits to the editable Python program so it fits the recorded history, and
(b) when the model is too weak to plan, proposes one exploratory action to
gather information. All game understanding lives in the program the model writes,
never in the harness.

Prompts are reconstructions (see ``schema_repro/prompts``); they contain no
game-specific content, matching Schema's "same agent and prompts across games".
The agent works with any ``Provider`` — the offline ``FakeProvider`` for tests
and trace replay, or a real frontier provider for live runs.
"""

from __future__ import annotations

import re
from pathlib import Path

from schema_repro.providers.base import ModelRequest, Provider
from schema_repro.types import Action, Effort, Frame, SIMPLE_ACTIONS
from schema_repro.world_model import EditableProgram, Transition

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

_FALLBACK_SYSTEM = (
    "You are a world-model engineer. You are given the interaction history of an "
    "unknown turn-based game as (frame, action, next_frame) transitions. Maintain "
    "a single Python program implementing ground(frame), step(state, action), "
    "is_goal(state), render(state) and actions(state). Edit it so that for every "
    "recorded transition, render(step(ground(frame), action)) reproduces next_frame. "
    "Prefer the simplest program that fits all observations. Reply with the full "
    "program in one ```python code block."
)

_FALLBACK_EXPLORE = (
    "The world model cannot yet plan to the goal. Propose ONE action to reduce "
    "uncertainty. Reply with a single token: ACTION1..ACTION6 (for ACTION6 append "
    "x=<int> y=<int>)."
)

_CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
_ACTION_RE = re.compile(r"\b(ACTION[1-6])\b(?:.*?x\s*=\s*(\d+).*?y\s*=\s*(\d+))?", re.DOTALL)


def _load_prompt(name: str, fallback: str) -> str:
    path = PROMPTS_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return fallback


class LLMAgent:
    def __init__(self) -> None:
        self.system_revise = _load_prompt("world_model_engineer.md", _FALLBACK_SYSTEM)
        self.system_explore = _load_prompt("explorer.md", _FALLBACK_EXPLORE)

    # -- world-model authoring -------------------------------------------

    def revise_world_model(
        self, program: EditableProgram, history: list[Transition], provider: Provider, effort: Effort
    ) -> str:
        req = ModelRequest(
            system=self.system_revise,
            messages=[{"role": "user", "content": self._history_prompt(program, history)}],
            effort=effort,
        )
        resp = provider.complete(req)
        m = _CODE_BLOCK.search(resp.text)
        if m and m.group(1).strip():
            program.edit(m.group(1).strip())
        return program.revision_id()

    # -- exploration ------------------------------------------------------

    def explore(
        self, frame: Frame, history: list[Transition], provider: Provider, effort: Effort
    ) -> Action:
        req = ModelRequest(
            system=self.system_explore,
            messages=[{"role": "user", "content": self._explore_prompt(frame)}],
            effort=effort,
        )
        resp = provider.complete(req)
        return self._parse_action(resp.text, frame)

    # -- prompt construction / parsing -----------------------------------

    def _history_prompt(self, program: EditableProgram, history: list[Transition]) -> str:
        lines = ["# Current program\n", program.source or "(empty)", "\n# Recent transitions"]
        for tr in history[-16:]:
            lines.append(
                f"action={tr.action.name}{tr.action.data or ''} "
                f"grid={tr.frame.grid} -> next={tr.next_frame.grid} "
                f"state={tr.next_frame.state.value} score={tr.next_frame.score}"
            )
        return "\n".join(lines)

    def _explore_prompt(self, frame: Frame) -> str:
        return (
            f"grid={frame.grid}\navailable_actions={frame.available_actions}\n"
            f"score={frame.score} state={frame.state.value}"
        )

    def _parse_action(self, text: str, frame: Frame) -> Action:
        m = _ACTION_RE.search(text or "")
        if m and (m.group(1) in frame.available_actions or not frame.available_actions):
            name = m.group(1)
            if name == "ACTION6" and m.group(2) and m.group(3):
                return Action(name, {"x": int(m.group(2)), "y": int(m.group(3))})
            if name != "ACTION6":
                return Action(name)
        # Fallback: first legal simple action, else ACTION1.
        for a in frame.available_actions:
            if a in SIMPLE_ACTIONS:
                return Action(a)
        return Action(SIMPLE_ACTIONS[0])
