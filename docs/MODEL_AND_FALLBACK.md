# Model and fallback layer

This document explains the model layer of the reproduction: which frontier models
Schema used, how this repo abstracts model access behind a provider protocol, how
per-phase reasoning-effort budgets are assigned, and how the "Fable-5 fallback
logic" works — what it actually is, why a reproduction can only *enable* it rather
than re-implement it, and how the generic `FallbackProvider` models its *contract*
for the offline and cross-provider cases.

Schema is a **model-agnostic harness**: it changes the process around a model, not
the model's weights. The same controller, prompts, verifiers and planner reached
top scores with two different vendor stacks. Everything in this repo is therefore
built so that the model is one swappable component behind a narrow interface.

---

## 1. The three models and the results

| Model | Vendor | Role in Schema | Reported result (ARC-AGI-3 Public) |
| --- | --- | --- | --- |
| Claude Fable 5 | Anthropic | First-called ("primary") model of the Anthropic stack in this repo's default chain | Part of the ~99% RHAE run |
| Claude Opus 4.8 | Anthropic | The more-capable model Fable 5's safety refusals fall back to (the refusal-fallback target) | Part of the ~99% RHAE run |
| GPT-5.6 Sol | OpenAI | Sole model of the OpenAI stack | 95.35% RHAE |

> The public sources report only that Schema used "Opus 4.8 + Fable 5" — they do
> not designate a primary model. The Fable-5-first / Opus-4.8-fallback ordering is
> **this repo's own config choice** (`configs/models.yaml`), matching the direction
> of the real server-side refusal fallback (§5): you call Fable 5, and Opus 4.8
> catches the refusals.

- **Opus 4.8 + Fable 5 (Anthropic):** self-reported **98.98% RHAE** (rounded to
  ~99% throughout the fact sheet).
- **GPT-5.6 Sol (OpenAI):** **95.35% RHAE**.

RHAE (Relative Human Action Efficiency) compares the agent's per-level action
count against a first-exposure human baseline and aggregates across environments;
100% means completing every level of every environment at or above human-baseline
action efficiency. These two numbers are the only quantitative results this repo
treats as fact — see `docs/REPRODUCTION_PLAN.md` for the known/inferred split, and
note that **this repo makes no ARC-AGI-3 score claim of its own**: it reconstructs
the structure, validates the *scoring* by recomputing RHAE from traces, and runs a
toy game offline. The RHAE numbers produced by the synthetic fixtures under
`fixtures/traces/` are illustrative plumbing tests, not measurements.

Model identities themselves are a moving target. The exact API versions
Impossible Research used are **not public**, and frontier models drift and
deprecate over time, so this repo deliberately does not pin dated snapshots — see
§7 for how model IDs are resolved from env vars.

---

## 2. Where model access lives: the `Provider` abstraction

All model calls go through one narrow protocol so that "which model" and "how the
harness works" stay independent. The relevant files:

- `schema_repro/providers/base.py` — the `Provider` protocol, the
  `ModelRequest` / `ModelResponse` dataclasses, and the generic `FallbackProvider`.
- `schema_repro/providers/anthropic_provider.py` — Opus 4.8 / Fable 5.
- `schema_repro/providers/openai_provider.py` — GPT-5.6 Sol.
- `schema_repro/providers/fake.py` — offline deterministic providers for tests.
- `schema_repro/providers/registry.py` — builds a concrete provider from a
  configured profile.

The protocol (`schema_repro/providers/base.py`) is intentionally tiny:

```python
@runtime_checkable
class Provider(Protocol):
    name: str
    def complete(self, request: ModelRequest) -> ModelResponse: ...
```

A `ModelRequest` carries a `system` string, a `messages` list, an `effort`
(`schema_repro/types.py:Effort`, default `HIGH`), a `max_tokens`, and a free-form
`extra` dict for provider-specific knobs. A `ModelResponse` carries `text`,
`model`, a `stop_reason` (`"end_turn" | "max_tokens" | "refusal" | ...`), token
counts and latency, plus a convenience property:

```python
@property
def refused(self) -> bool:
    return self.stop_reason == "refusal"
```

The `refused` flag is the hinge the fallback machinery turns on (§5–§6).

The controller (`schema_repro/controller.py`) never imports a vendor SDK. It is
handed a `Provider` and calls `provider.complete(...)` indirectly through the
agent's `revise_world_model(...)` and `explore(...)` methods, passing a per-phase
effort value. That keeps the harness genuinely model-agnostic: swapping the
`Provider` swaps the model with no change to the control loop.

---

## 3. Per-phase reasoning-effort budget

Public guidance on Schema is that **planning and verification warrant extra-high
reasoning compute, while execution warrants high**. This repo encodes that as a
phase-to-effort map rather than a single global setting, because the controller
runs a fixed phase cycle and each phase deserves a different depth.

### Where it lives

`configs/models.yaml` holds the map verbatim:

```yaml
effort_by_phase:
  ground: high
  discover: xhigh
  verify: xhigh
  refactor: high
  plan: xhigh
  execute: high
```

The effort tiers are the `Effort` enum in `schema_repro/types.py`
(`LOW | MEDIUM | HIGH | XHIGH`), and the phases are the `Phase` enum in the same
file (`GROUND -> DISCOVER -> VERIFY -> REFACTOR -> PLAN -> EXECUTE`).

### How it is consumed

`schema_repro/config.py` parses the YAML into a typed `RunConfig`. It converts the
raw map into `dict[Phase, Effort]` and exposes a lookup with a safe default:

```python
effort_by_phase = {Phase(k): Effort(v) for k, v in models.get("effort_by_phase", {}).items()}
...
def effort(self, phase: Phase) -> Effort:
    return self.effort_by_phase.get(phase, Effort.HIGH)
```

The controller (`schema_repro/controller.py`) reads that lookup when it drives the
agent. In `run_level(...)` the discovery/refinement edit is made at the
`DISCOVER` effort and the exploratory fallback action is taken at the `EXECUTE`
effort:

```python
self.agent.revise_world_model(program, history, self.provider, self.cfg.effort(Phase.DISCOVER))
...
self.agent.explore(frame, history, self.provider, self.cfg.effort(Phase.EXECUTE))
```

The effort value ultimately becomes each provider's depth control (§4). Because
the assignment is data in `configs/models.yaml`, tuning the compute profile is a
config edit, not a code change.

> **Honesty note.** The *shape* of the budget (plan/verify high, execute lower) is
> reconstructed from public guidance. The exact per-phase effort tiers Impossible
> Research used are **not public**; the specific `xhigh`/`high` assignment here is
> this repo's reconstruction.

---

## 4. Current Anthropic / OpenAI API correctness

`schema_repro/providers/anthropic_provider.py` maps the abstract request onto the
current Anthropic Messages API and is careful about the API facts that differ
between model tiers:

- **Reasoning effort → `output_config.effort`.** The `Effort` enum values line up
  1:1 with the API's `{low, medium, high, xhigh, max}` levels, so the mapping is a
  passthrough (`_effort(effort)` returns `effort.value`). This is the depth control
  shared across current Anthropic tiers (Opus 4.8, Fable 5, and others such as
  Sonnet 5). *Only Opus 4.8 + Fable 5 and GPT-5.6 Sol are part of the reported
  Schema evaluation; any other model names here are illustrative of the provider's
  handling, not claims about what Schema ran.*
- **Opus 4.8 → adaptive thinking.** Opus-tier models must request thinking
  explicitly, so the provider sets `thinking={"type": "adaptive"}` for any model
  *not* in the always-thinking set.
- **Fable 5 → thinking always on.** For models whose id starts with an
  `_ALWAYS_THINKING` family prefix (`"claude-fable-5"`, `"claude-mythos-5"` — the
  latter an illustrative sibling, not a Schema model) the `thinking` parameter is
  **omitted** entirely — passing it would be wrong because thinking is always on.
  Membership is a **prefix** match (`_is_always_thinking`), so a dated snapshot
  like `claude-fable-5-20260601` is still recognised.
- **No `budget_tokens`, no `temperature`/`top_p`/`top_k`.** Opus 4.8 and Fable 5
  reject these (HTTP 400). The provider never sends them; depth is controlled only
  through `output_config.effort`. (The `extra` dict is available for other
  passthrough knobs, but the harness does not put sampling params there.)

The OpenAI provider (`schema_repro/providers/openai_provider.py`) maps the same
`Effort` tiers onto GPT-5.6 Sol's coarser `reasoning_effort` hint. Note that
GPT-5.6 Sol exposes fewer tiers, so `XHIGH` collapses onto `"high"` in
`_EFFORT_MAP`. That collapse is a reconstruction of a plausible mapping, not a
claim about how Impossible Research configured OpenAI effort; the exact request
surface for GPT-5.6 Sol reasoning effort is provider-specific and may change. The
OpenAI provider also translates a `content_filter` finish reason into
`stop_reason="refusal"`, so refusals from either vendor present identically to the
fallback machinery.

Both concrete providers import their vendor SDK **lazily** (inside
`_client_or_init`), so the rest of the package — offline tests, trace replay, RHAE
recompute — works with no SDK installed and no API key present.

---

## 5. The "Fable-5 fallback logic"

There are two distinct things called "fallback" here, and it matters to keep them
apart:

1. **The provider-internal server-side refusal fallback** (this section) — a real
   Anthropic API behaviour that runs *inside* a single API call. A reproduction can
   only **enable** it.
2. **The repo's generic `FallbackProvider`** (§6) — a client-side ordered chain
   this repo implements to model the *contract* for offline and cross-provider
   cases.

### What the server-side fallback is

Anthropic's Fable 5 produces a refusal `stop_reason` from a two-stage safety
classifier. On the Messages API you can opt a Fable 5 request into a **server-side
fallback**: when the request is declined by the safety classifiers, the API
returns HTTP `200` with `stop_reason` `"refusal"` and **re-serves the same call
with Opus 4.8**, inside that one request. The caller gets a completed response; the
hand-off happened on Anthropic's side.

It is enabled with two parameters on the beta Messages endpoint:

```python
betas=["server-side-fallback-2026-06-01"]
fallbacks=[{"model": "claude-opus-4-8"}]
```

In this repo that is exactly what `AnthropicProvider` sets when
`enable_refusal_fallback` is on (`schema_repro/providers/anthropic_provider.py`):

```python
_FALLBACK_BETA = "server-side-fallback-2026-06-01"
_DEFAULT_FALLBACK_MODEL = "claude-opus-4-8"
...
if self.enable_refusal_fallback:
    params["betas"] = [_FALLBACK_BETA]
    params["fallbacks"] = [{"model": _DEFAULT_FALLBACK_MODEL}]
...
# fallbacks live on the beta endpoint; otherwise the GA endpoint is used
endpoint = client.beta.messages if "fallbacks" in params else client.messages
```

`enable_refusal_fallback` is only meaningful for an always-thinking model (it is
AND-ed with `model in _ALWAYS_THINKING` in `__init__`), i.e. Fable 5 is the model
that can hand off to Opus 4.8, not the other way around.

### Why it can only be *enabled*, not reproduced

The two-stage refusal classifier and the re-serving logic are **provider-internal**.
They run inside Anthropic's infrastructure and are not exposed. An external harness
can flip the feature on with the beta header and the `fallbacks` list, but it
**cannot re-implement the classifier or reproduce the exact re-serve decision**.
This is one of the six things `docs/REPRODUCTION_PLAN.md` records as impossible to
byte-reproduce (item 3): the behaviour is *enableable* and its interface is known,
but its internals are not public and not reconstructable.

---

## 6. Modelling the contract: the generic `FallbackProvider`

For everything that is not a live Anthropic call — offline tests, the OpenAI
stack, or simply reasoning about the failure behaviour — this repo reconstructs the
*contract* of a fallback: **an ordered, primary-first chain of providers plus a
trigger predicate that decides whether to advance to the next provider**. That is
`FallbackProvider` in `schema_repro/providers/base.py`.

```python
@dataclass
class FallbackProvider:
    chain: list[Provider]
    should_fallback: object = refused_or_errored  # Callable[[resp, exc], bool]
    max_attempts: int | None = None
    name: str = "fallback"
```

`complete(...)` walks the chain, calling each provider, and after each attempt asks
`should_fallback(resp, exc)` whether to try the next one. The default trigger:

```python
def refused_or_errored(resp: ModelResponse | None, exc: Exception | None) -> bool:
    return exc is not None or (resp is not None and resp.refused)
```

So the default advances on **any exception or any refusal** and returns the first
non-triggering response; if the chain is exhausted it returns the best-effort last
response, and if every provider errored it raises. This is a faithful
reconstruction of the *shape* of the Fable-5 → Opus-4.8 hand-off (refuse →
try the next model), and explicitly not its proprietary internals.

### How a profile becomes a provider

`schema_repro/providers/registry.py` turns a configured `ModelProfile` (from
`configs/models.yaml`) into a live provider:

- A single-stage profile returns that one provider directly.
- A multi-stage profile is wrapped in a `FallbackProvider` whose `chain` is the
  stages in order.

One important interaction, controlled by the chain length so the two fallback
mechanisms never stack:

```python
single_stage = len(profile.chain) == 1
stages = [_build_stage(s.provider, s.model, server_side_fallback=single_stage)
          for s in profile.chain]
```

- The **shipped `anthropic` profile is two-stage** (Fable 5 then Opus 4.8), so its
  Anthropic stages are built with `server_side_fallback=False` and the Fable→Opus
  hand-off is modelled **client-side** by the wrapping `FallbackProvider`. This is
  the path the offline tests exercise.
- A **single-stage Fable-5 profile** is built with `server_side_fallback=True`, so
  the registry turns on the genuine Anthropic **server-side** refusal fallback
  (`betas=["server-side-fallback-2026-06-01"]` + `fallbacks=[{"model":
  "claude-opus-4-8"}]`) inside one API call.

So you get exactly one mechanism: the client-side chain (multi-stage profile) *or*
the server-side fallback (single-stage always-thinking profile), never both. You
can also construct `AnthropicProvider(model, enable_refusal_fallback=True)`
directly to force the server-side path regardless of profile shape.

### Tests

`RefusingProvider` in `schema_repro/providers/fake.py` always returns
`stop_reason="refusal"` specifically to exercise the chain: a `FallbackProvider`
with `[RefusingProvider(), FakeProvider()]` must skip the refuser and return the
fake's completion. This is how the fallback *contract* is validated offline,
independent of the non-deterministic model responses that cannot be reproduced.

---

## 7. Pointing model IDs via environment variables

Because exact API versions are not public and drift over time, `configs/models.yaml`
does **not** hard-code dated snapshots. Each chain stage carries a default model ID
and an env var that overrides it:

```yaml
profiles:
  anthropic:
    chain:
      - provider: anthropic
        model_env: ANTHROPIC_DEFAULT_FABLE_MODEL
        model_default: claude-fable-5
      - provider: anthropic
        model_env: ANTHROPIC_DEFAULT_OPUS_MODEL
        model_default: claude-opus-4-8
    fallback_on: [refusal, error, timeout]

  openai:
    chain:
      - provider: openai
        model_env: OPENAI_DEFAULT_MODEL
        model_default: gpt-5.6-sol
    fallback_on: [error, timeout]
```

Resolution happens in `schema_repro/config.py:_resolve_stage`: if `model_env` is
set and that variable is present in the environment, its value wins; otherwise the
`model_default` is used.

```python
def _resolve_stage(stage):
    model = stage.get("model_default", "")
    env = stage.get("model_env")
    if env:
        model = os.environ.get(env, model)
    return ModelStage(provider=stage["provider"], model=model)
```

So to pin or update the concrete snapshots without editing YAML:

```bash
export ANTHROPIC_DEFAULT_FABLE_MODEL="claude-fable-5-<snapshot>"
export ANTHROPIC_DEFAULT_OPUS_MODEL="claude-opus-4-8-<snapshot>"
export OPENAI_DEFAULT_MODEL="gpt-5.6-sol-<snapshot>"
```

Two constants in `schema_repro/providers/anthropic_provider.py` interact with these
overrides:

- The always-thinking guard uses a **prefix** match (`_is_always_thinking`), so
  **keeping the family prefix intact** (`claude-fable-5-<snapshot>`) is enough — the
  snapshot is still recognised as always-thinking and won't get an illegal
  `thinking` param.
- The server-side fallback **target** is the exact constant
  `_DEFAULT_FALLBACK_MODEL = "claude-opus-4-8"`. Pinning a dated *Opus* snapshot as
  the fallback target is the one case that requires editing that constant (or the
  models.yaml default), because the target is not resolved from the env vars above.

The `fake` profile (`provider: fake`) is fully offline and deterministic, and is
what the tests and the RHAE-recompute path use so that neither needs network access
or credentials.

---

## Known / inferred / impossible — model layer summary

| Item | Status | Notes |
| --- | --- | --- |
| ~99% (98.98%) RHAE with Opus 4.8 + Fable 5; 95.35% with GPT-5.6 Sol | **Known** | Self-reported public results; the only quantitative facts used here. |
| Model-agnostic harness (same agent/prompts across stacks) | **Known** | Stated in public sources; reflected by the `Provider` abstraction. |
| Server-side refusal fallback interface (`betas`, `fallbacks`) | **Known / enableable** | Documented API surface; enabled in `anthropic_provider.py`. |
| The two-stage refusal classifier and re-serve decision | **Impossible** | Provider-internal; cannot be re-implemented, only enabled. |
| Exact API versions / dated snapshots | **Not public** | Left to env-var overrides; defaults are family IDs. |
| Per-phase effort assignment (`xhigh` plan/verify, `high` execute) | **Inferred** | Shape follows public guidance; exact tiers not public. |
| GPT-5.6 Sol effort mapping (`xhigh`→`high`) | **Reconstructed** | Plausible mapping onto coarser tiers; not an original-run claim. |
| Fixtures' RHAE numbers | **Illustrative** | Synthetic; validate scoring plumbing, not performance. |

---

## Sources

- https://schema-harness.github.io/
- https://arxiv.org/abs/2605.05138
- https://arcprize.org/
- https://huggingface.co/datasets/schema-harness/arc-agi-3-schema-traces
