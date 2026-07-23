# `schema_repro/prompts/` — reconstructed agent-role prompts

These files are **clean-room reconstructions** of the two agent roles Schema uses
to author and drive its world-model program: a **world-model engineer** (writes
and continually edits the single Python program that *is* the game's mechanism)
and an **explorer** (proposes one information-gaining action when the model is
still too weak to plan). They are inferred from public descriptions of the Schema
harness (paper, project page, dataset card) and copy **no original text** — the
real system prompts were described, not released ("project code released after
acceptance"), so exact reproduction is impossible. They are deliberately
**game-agnostic**: no game id, object list, rule sheet, or stated goal appears in
either prompt, matching Schema's design of using the *same agent and prompts
across all games*. At start-up `schema_repro/agent.py` (`LLMAgent`) resolves each
file through `_load_prompt(...)`; when a file is present it becomes that role's
system prompt, and when it is absent the agent falls back to a terse inline
default (`_FALLBACK_SYSTEM` for the engineer, `_FALLBACK_EXPLORE` for the
explorer), so the harness runs with or without this directory. Both prompts are
written to satisfy the parsers in `agent.py` — a single fenced `python` block for
the engineer, a single `ACTION1..ACTION6` token (with optional `x=/y=`) for the
explorer.

## Files

- `world_model_engineer.md` — system prompt for the program-authoring role
  (`LLMAgent.revise_world_model`): maintain and edit one Python program exposing
  `ground` / `step` / `is_goal` / `render` / `actions` so every recorded
  transition is reproduced, preferring the simplest fitting program. Fallback:
  `_FALLBACK_SYSTEM`.
- `explorer.md` — system prompt for the exploration policy
  (`LLMAgent.explore`): invoked only when the world model cannot yet plan to the
  goal; picks the single action most likely to reduce uncertainty. Fallback:
  `_FALLBACK_EXPLORE`.
- `README.md` — this note.
