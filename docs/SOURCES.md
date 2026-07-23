# Sources

An annotated bibliography of the **public** sources this clean-room reproduction
relies on. Each entry notes what it established for this project. Nothing here is
copied from the unreleased *Schema* implementation; details that appear in none
of these sources are **inferred** and are labelled as such in the code, the
configs, and the docs (see [`REPRODUCTION_PLAN.md`](REPRODUCTION_PLAN.md)).

The project reconstructs the harness *design* by **Impossible Research**
(Guanning Zeng, Andrea Zanette) for [ARC-AGI-3](https://arcprize.org/); it copies
no original prompts, weights, or traces.

---

## Primary sources

### 1. Schema project page — `schema-harness.github.io`

<https://schema-harness.github.io>

The canonical hub for the harness. Establishes the framing this reproduction is
built around: an LLM plays each game **"like a physicist,"** writing and
continually editing a single small Python program that *is* the game's mechanism,
and the claim that the **same agent and prompts are used across all games** with
**no game-specific code, prompts, heuristics, or hidden solutions**. Source for
the headline results quoted throughout the repo (~99% RHAE / self-reported 98.98%
with Claude Opus 4.8 + Fable 5; 95.35% with GPT-5.6 Sol on the ARC-AGI-3 Public
set) and for the "project code released after acceptance" note that makes the
concrete prompts and agent configuration non-public. Also the stated reason this
is a *conceptual* reproduction rather than a fork.

### 2. Harness paper — arXiv 2605.05138

*Executable World Models for ARC-AGI-3 in the Era of Coding Agents*
<https://arxiv.org/abs/2605.05138>

The design document this repo tracks most closely. Establishes the two formal
sub-problems solved **jointly** in one editable program — **state grounding**
(raw 64x64 grid of 16 colour indices to objects/variables/relations) and
**mechanism discovery** (how state changes under an action, written as an
executable rule) — and the role of **formal verifiers** that gate every revision:
a world-model verifier (the program must *reproduce* recorded observations) and a
planner verifier (the plan must reach completion *inside* the learned model)
before live actions are committed. Source for the **MDL-like simplicity bias**:
refactoring the program toward simpler abstractions as a practical proxy for
minimum description length. Directly shapes
[`schema_repro/world_model.py`](../schema_repro/world_model.py),
[`schema_repro/verifier.py`](../schema_repro/verifier.py),
[`schema_repro/planner.py`](../schema_repro/planner.py), and the controller phase
cycle in [`schema_repro/controller.py`](../schema_repro/controller.py). Note: the
arXiv identifier and title are used exactly as given in the fact sheet; no authors
or dates beyond those are asserted here.

### 3. ARC Prize — ARC-AGI-3 technical report and environment spec

<https://arcprize.org/arc-agi/3> · <https://arcprize.org/>

Establishes the **environment and the metric** the reproduction targets. The
environment: an interactive POMDP where the agent is dropped into a novel game
with no instructions, object list, rule sheet, stated goal, or shaped reward, and
per step receives a 64x64 grid of 16 colour indices plus a set of legal actions.
Source for the frame fields (`game_id`, `frame`, `state`, `score` 0–254,
`available_actions`), the action set (`RESET`, `ACTION1..ACTION5` simple,
`ACTION6` carrying `{x,y}`), the states (`NOT_PLAYED`, `NOT_FINISHED`, `WIN`,
`GAME_OVER`), the 6-game / 3-public-3-private / 8–10-levels structure, and the
definition of **RHAE (Relative Human Action Efficiency)** — per-level action count
versus a *first-exposure* human baseline, aggregated across environments, 100% =
completing every level at or above the human baseline. Grounds
[`schema_repro/types.py`](../schema_repro/types.py),
[`schema_repro/env/arc_agi3.py`](../schema_repro/env/arc_agi3.py), and
[`schema_repro/scoring/rhae.py`](../schema_repro/scoring/rhae.py).

### 4. Hugging Face dataset — `schema-harness/arc-agi-3-schema-traces`

<https://huggingface.co/datasets/schema-harness/arc-agi-3-schema-traces>

Establishes the **trace layout** the scoring pipeline is validated against: two
collections (`gpt_5_6_sol/` and `claude_fable_opus/`), a shared
`baseline_actions.csv` of human baselines, and per-trajectory directories holding
`run.json`, a streamed `events.jsonl` event log, sanitized session data,
snapshots, and shareable text/image files (50 `events.jsonl` files total). Source
for the recompute workflow — stream the event logs, reconstruct per-level action
counts, recompute every RHAE score, print a 50-row table plus a summary — mirrored
by [`schema_repro/tracing.py`](../schema_repro/tracing.py),
[`schema_repro/scoring/rhae.py`](../schema_repro/scoring/rhae.py), and
[`scripts/recompute_rhae.py`](../scripts/recompute_rhae.py). **Important:** the
exact `events.jsonl` field names are **not public**; this repo reconstructs a
*compatible* schema and says so in [`TRACE_FORMAT.md`](TRACE_FORMAT.md).

### 5. Announcement — Andrea Zanette / Guanning Zeng (Impossible Research)

Linked from the project page above.

The authors' public announcement of Schema. Source for attribution (Impossible
Research; Guanning Zeng, Andrea Zanette) and for the model line-up and framing
echoed on the project page — a **model-agnostic** harness (it changes the process
around a model, not the weights) evaluated with Claude Opus 4.8 + Fable 5 and
GPT-5.6 Sol. No specific post URL is asserted here to avoid fabricating a
citation; treat `schema-harness.github.io` as the canonical entry point.

---

## Supporting reference (provider API facts)

Provider-side facts about the model chain — Opus 4.8's `thinking={"type":
"adaptive"}`, Fable 5's always-on thinking, `output_config.effort` in
`{low, medium, high, xhigh, max}`, the 400s on `budget_tokens`/`temperature`, and
the server-side refusal fallback (`betas=["server-side-fallback-2026-06-01"]`,
`fallbacks=[{"model":"claude-opus-4-8"}]`) — come from the respective providers'
API documentation, not from any Schema source. They are recorded in
[`MODEL_AND_FALLBACK.md`](MODEL_AND_FALLBACK.md). The **Fable-5 fallback logic**
itself is a provider-internal, two-stage classifier: a reproduction can *enable*
it but cannot re-implement it.

---

## Provenance & integrity

- **Nothing here is copied from the unreleased implementation.** The original
  system prompts, agent configuration, and project code are described in the
  sources above but not released ("project code released after acceptance"); no
  weights, prompts, or original traces are reproduced.
- **Public vs. inferred is kept explicit.** Where these sources fix a fact (the
  environment spec, the RHAE definition, the harness's structure, the headline
  results), the repo states it as such. Where they do not (concrete prompts,
  exact model/API versions, sandbox/time/token/tool limits, internal eval
  infrastructure, the trace field names), the repo marks the detail as
  **reconstructed / inferred / not public** — item by item in
  [`REPRODUCTION_PLAN.md`](REPRODUCTION_PLAN.md).
- **Illustrative numbers are labelled.** RHAE figures produced from `fixtures/`
  are synthetic and illustrative; they measure the *pipeline*, not any ARC-AGI-3
  performance. This repo makes **no ARC-AGI-3 score claim**.
- **Validation, not replication.** The one thing these sources let anyone check —
  recomputing RHAE from the published trace format — is exactly what the scoring
  pipeline reproduces; exact byte-for-byte reproduction of the original runs is
  impossible by nature (non-deterministic sampling) and is not attempted.

---

## URLs

- <https://schema-harness.github.io>
- <https://arxiv.org/abs/2605.05138>
- <https://arcprize.org/arc-agi/3>
- <https://arcprize.org/>
- <https://huggingface.co/datasets/schema-harness/arc-agi-3-schema-traces>
