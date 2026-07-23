# Trace format

This document reverse-engineers the trace format published by the
`schema-harness/arc-agi-3-schema-traces` dataset and specifies the **compatible
reconstruction** used by this repository. The goal is that the same recompute
logic can score both: a run produced by this repo can be scored by the dataset's
workflow, and a published trace can be replayed and re-scored by this repo's
tooling.

> **Honesty note.** The *exact* `events.jsonl` field names in the published
> dataset are **not public**. Everything in the [Reconstructed event schema](#reconstructed-event-schema)
> section below is an **inferred, compatible** schema authored by this repo, not
> a transcription of the originals. Where a name or payload key is our invention,
> it is called out as **inferred**. The dataset's *directory layout* and the fact
> that a recompute command streams the events and reprints RHAE **are** described
> in public material; those parts are marked **published**.

## 1. What the dataset publishes

**(published)** The Hugging Face dataset `arc-agi-3-schema-traces` is organized
as two top-level **collections**, one per model chain, plus a single shared
human-baseline file:

```
arc-agi-3-schema-traces/
  baseline_actions.csv          # shared first-exposure human baselines (both collections)
  gpt_5_6_sol/                  # collection: GPT-5.6 Sol runs
    <trajectory>/
      run.json                  # run-level metadata
      events.jsonl              # streamed per-step event log
      ...                       # sanitized session data, snapshots, shareable text/image files
  claude_fable_opus/            # collection: Claude Opus 4.8 + Fable 5 runs
    <trajectory>/
      run.json
      events.jsonl
      ...
```

Per-trajectory contents described publicly: a `run.json`, a streamed
`events.jsonl` event log, sanitized session data, snapshots, and shareable
text/image files. There are **50 `events.jsonl` files** in total across the two
collections. A recompute command streams every `events.jsonl`, reconstructs
per-level action counts, recomputes every RHAE score, and prints a 50-row table
plus a summary.

This repo reproduces the **layout** and the **recompute workflow**, not the
original byte contents. The only field-level facts we can rely on are the ones
the metric itself forces: an event stream must let you (a) count actions per
level and (b) tell which levels were solved, and there must be a baseline table
keyed by game and level. Everything else in our schema is a reconstruction.

## 2. Reconstructed event schema

**(inferred)** Source of truth: [`schema_repro/tracing.py`](../schema_repro/tracing.py).

Each line of `events.jsonl` is one JSON object serialized from the `Event`
dataclass. The schema is deliberately small and flat:

| Field     | Type            | Meaning |
|-----------|-----------------|---------|
| `t`       | `int`           | Monotonic **step index**, not wall-clock time. Starts at 0 and increments once per emitted event, so replays are deterministic and order-stable. |
| `type`    | `str`           | Event kind, drawn from the closed vocabulary below. |
| `game_id` | `str`           | The environment/game identifier (e.g. `ls20`). Empty string if not applicable. |
| `level`   | `int`           | Zero-based level index within the game. Defaults to `0`. |
| `payload` | `object` (dict) | Type-specific fields. May be empty (`{}`). |

`TraceWriter.emit(type, *, game_id="", level=0, **payload)` assigns the running
`t`, wraps the keyword arguments into `payload`, writes one compact JSON line
(`separators=(",", ":")`), and flushes. `read_events` streams the file back into
`Event` objects, tolerating missing keys via `.get(...)` defaults, so a
partially-shaped external trace still parses.

The `t`/`type`/`game_id`/`level`/`payload` split — in particular pushing all
type-specific data down into `payload` rather than a flat record — is **inferred**.
The dataset may use flat records or different key names; our shape is chosen so
the closed vocabulary can grow without schema migrations.

### Closed vocabulary of event types

**(inferred)** The `type` field takes one of nine values. The list is closed;
`schema_repro/tracing.py` enumerates it in the `Event` docstring, and the
controller and fixtures only ever emit these. Payload keys are **inferred** and
listed with the module that emits them.

| `type`        | Emitted by | Payload keys (inferred) | Purpose |
|---------------|------------|-------------------------|---------|
| `run_start`   | run driver ([`scripts/make_fixtures.py`](../scripts/make_fixtures.py)) | `model` | Opens a trajectory; records which model chain produced it. |
| `level_start` | [`controller.py`](../schema_repro/controller.py) | `score` | A level begins; carries the starting score. |
| `action`      | `controller.py` | `action` (name, e.g. `ACTION1`), `data` (dict, e.g. `{x,y}` for `ACTION6`) | One committed live action. **These are the events RHAE counts.** |
| `observation` | `controller.py` | `state` (e.g. `NOT_FINISHED`, `WIN`, `GAME_OVER`), `score` | The frame returned after an action. |
| `model_edit`  | `controller.py` | `revision` (revision id), `edits` (running edit count) | The agent revised the single editable world-model program (ground/discover). |
| `verify`      | `controller.py` | `passed` (bool), `revision` | Result of the world-model + planner verifier gate; a failed gate triggers rollback. |
| `plan`        | `controller.py` | `length` (plan length) | The planner searched inside the model and produced a plan of this length. |
| `level_end`   | `controller.py` | `solved` (bool), `action_count` (int) | A level finished. `solved` marks whether it was a WIN. |
| `run_end`     | run driver (`scripts/make_fixtures.py`) | — | Closes the trajectory. |

Notes:

- **`run_start` / `run_end` are emitted at the run/trajectory level, not by the
  controller.** The controller (`schema_repro/controller.py`) owns everything
  from `level_start` to `level_end`; the surrounding driver that opens and closes
  the trajectory emits `run_start`/`run_end`. In this repo the fixtures generator
  is that driver. The offline demo ([`scripts/run_offline_demo.py`](../scripts/run_offline_demo.py))
  calls `run_level` once directly, so its trace begins at `level_start`.
- The `model_edit`, `verify`, and `plan` events are **structural** — they make the
  ground → discover → verify → refactor → plan → execute cycle visible in the
  trace — but they are **not** consumed by the RHAE scorer. Only `action` and
  `level_end` matter for scoring (see §4). They exist so a reader can see the
  harness's reasoning loop, and so a richer scorer could be built later.

## 3. `baseline_actions.csv`

**(published layout, reconstructed values)** Source of truth for the reader:
[`schema_repro/scoring/rhae.py`](../schema_repro/scoring/rhae.py) (`load_baselines`);
for the fixture values: [`fixtures/traces/baseline_actions.csv`](../fixtures/traces/baseline_actions.csv).

A single CSV, shared across both collections, gives the **first-exposure human
baseline** action count per game and level:

```csv
game_id,level,baseline_actions
ft09,0,8
ft09,1,12
ls20,0,10
ls20,1,14
```

| Column             | Type  | Meaning |
|--------------------|-------|---------|
| `game_id`          | `str` | Game identifier, matching the `game_id` in the event streams. |
| `level`            | `int` | Zero-based level index. |
| `baseline_actions` | `int` | Human first-exposure action count for that level; the RHAE denominator. |

`load_baselines` reads it into a `{(game_id, level): baseline_actions}` map.
`recompute_rhae.py` loads it once up front, partly to **fail fast** on a malformed
CSV. The `game_id`/`level` names above are shared across the CSV and the events,
so joins are exact.

> The four rows shown are illustrative fixture data (games `ft09`, `ls20`, two
> levels each), **not** real ARC-AGI-3 baselines. The real dataset has more games
> and 8–10 levels each.

## 4. From events to RHAE

**(reconstructed)** Two small functions bridge the trace to the metric.

**Per-level action counts** — `action_counts_by_level(events)` in
`schema_repro/tracing.py`:

```python
counts: dict[tuple[str, int], int] = {}
for ev in events:
    if ev.type == "action":
        key = (ev.game_id, ev.level)
        counts[key] = counts.get(key, 0) + 1
```

It counts one action per `action` event, keyed by `(game_id, level)`. Nothing
else contributes; `observation`, `model_edit`, `verify`, and `plan` events are
ignored for counting. This is the reconstruction of "reconstruct per-level action
counts from the event stream."

**Solved set** — `_solved_levels(events)` in `schema_repro/scoring/rhae.py`
collects every `(game_id, level)` whose `level_end` event has a truthy
`payload["solved"]`.

**Scoring** — `score_events(events_path, baselines)`:
1. reads all events;
2. builds per-level action counts and the solved set;
3. emits one `LevelScore(game_id, level, agent_actions, baseline_actions, solved)`
   per counted level, joining the baseline map (missing baseline → `0`).

`LevelScore.efficiency` is the reconstructed per-level RHAE term. The exact
clipping/aggregation is **not fully specified in public material**; this repo
documents a reconstruction satisfying the metric's stated properties:

```
efficiency = 0                       if not solved (or agent_actions <= 0)
           = min(1, baseline / agent) if solved
```

Using at most the human budget scores 100%; using more scores proportionally
less. `RhaeReport.per_game()` averages the level efficiencies within each game;
`RhaeReport.overall()` macro-averages across games (mean of per-game means).

**Whole-dataset recompute** — `recompute_from_traces(root, baseline_csv)`
discovers every `events.jsonl` under a root via `rglob`, scores each against the
shared baselines, and concatenates the level scores. This mirrors the dataset's
recompute command that streams all traces and recomputes every score.

### How the CLI groups by collection

**(reconstructed)** [`scripts/recompute_rhae.py`](../scripts/recompute_rhae.py)
treats each **immediate subdirectory** of the traces root as one collection
(e.g. `gpt_5_6_sol/`, `claude_fable_opus/`) and scores them **separately**:

```
python scripts/recompute_rhae.py fixtures/traces
```

For each collection it runs `recompute_from_traces(collection, baseline_csv)`,
prints a per-level row (`collection · game · level · agent · base · solved · eff%`),
and finally a per-collection RHAE summary (`RhaeReport.overall()`). If the root
has no subdirectories, the root itself is treated as a single collection. The
baseline CSV defaults to `<root>/baseline_actions.csv` and can be overridden with
`--baseline`.

> The RHAE percentages this prints for the fixtures are **illustrative** and
> follow from the synthetic action counts in `scripts/make_fixtures.py`. They are
> not the published results and this repo makes no ARC-AGI-3 score claim. The only
> real reported figures are Schema's own: ~99% (self-reported 98.98%) RHAE with
> Opus 4.8 + Fable 5, and 95.35% with GPT-5.6 Sol.

## 5. Annotated example

**(inferred)** These are the first lines of a real fixture,
`fixtures/traces/claude_fable_opus/ls20/events.jsonl`, generated by
`scripts/make_fixtures.py`:

```jsonl
{"t":0,"type":"run_start","game_id":"ls20","level":0,"payload":{"model":"claude_fable_opus"}}
{"t":1,"type":"level_start","game_id":"ls20","level":0,"payload":{}}
{"t":2,"type":"action","game_id":"ls20","level":0,"payload":{"action":"ACTION1","data":{}}}
{"t":3,"type":"observation","game_id":"ls20","level":0,"payload":{"state":"NOT_FINISHED","score":0}}
{"t":4,"type":"action","game_id":"ls20","level":0,"payload":{"action":"ACTION1","data":{}}}
{"t":5,"type":"observation","game_id":"ls20","level":0,"payload":{"state":"NOT_FINISHED","score":1}}
```

Reading it line by line:

- `t=0` `run_start` — trajectory opens; `payload.model` names the collection/model chain.
- `t=1` `level_start` — level `0` of game `ls20` begins. (This fixture emits an
  empty payload; the controller's `level_start` carries `score`.)
- `t=2` `action` — one live action (`ACTION1`); `data` is empty because `ACTION1`
  is a simple action. A complex `ACTION6` would carry `{"x":.., "y":..}` in `data`.
  **This line increments the `(ls20, 0)` action count.**
- `t=3` `observation` — the resulting frame: still `NOT_FINISHED`, score `0`.
- `t=4`/`t=5` — the next action/observation pair; `action`/`observation` events
  alternate through the level.

A level then closes with, e.g.:

```jsonl
{"t":18,"type":"level_end","game_id":"ls20","level":0,"payload":{"solved":true,"action_count":8}}
```

Here `payload.solved` puts `(ls20, 0)` in the solved set and `action_count`
records 8 committed actions — which equals the number of `action` events the
scorer counted for that level, so `action_count` is a redundant convenience
field, not the counting source of truth.

A trace produced by the full controller (rather than the fixtures generator)
additionally interleaves `model_edit`, `verify`, and `plan` events between the
`action`/`observation` pairs, exposing the ground → discover → verify → refactor
→ plan → execute cycle. See `scripts/run_offline_demo.py` for a trace that
contains them.

## 6. Compatibility summary

| Aspect | Status |
|--------|--------|
| Two-collection layout + shared `baseline_actions.csv` | **published** — reproduced faithfully |
| Per-trajectory `run.json` + `events.jsonl` + session/snapshot/shareable files | **published** (existence); this repo focuses on `events.jsonl` + baselines |
| "Recompute command streams events, reconstructs per-level counts, reprints RHAE" | **published** — reconstructed in `scripts/recompute_rhae.py` |
| `events.jsonl` field names (`t`, `type`, `game_id`, `level`, `payload`) | **inferred** — originals not public |
| Closed event vocabulary + payload keys | **inferred** — chosen to satisfy the metric and expose the loop |
| Exact RHAE clipping/aggregation | **reconstructed** — public definition under-specifies it; see `schema_repro/scoring/rhae.py` |
| Fixture baseline values and RHAE numbers | **illustrative/synthetic** — not real ARC-AGI-3 data |

## Sources

- https://schema-harness.github.io
- https://arxiv.org/abs/2605.05138
- https://huggingface.co/datasets/schema-harness/arc-agi-3-schema-traces
- https://arcprize.org
