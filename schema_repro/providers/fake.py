"""A deterministic, offline provider for tests and trace replay.

It never touches the network. By default it echoes a canned response, but it can
be seeded with a scripted queue of responses (e.g. a sequence of world-model
program edits) so an end-to-end controller run is fully reproducible without any
model API. This is how we validate harness *plumbing* independently of the
non-deterministic model responses we cannot reproduce.
"""

from __future__ import annotations

from collections import deque

from schema_repro.providers.base import ModelRequest, ModelResponse


class FakeProvider:
    name = "fake"

    def __init__(self, scripted: list[str] | None = None, *, model: str = "fake-echo"):
        self._queue: deque[str] = deque(scripted or [])
        self.model = model
        self.calls: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        text = self._queue.popleft() if self._queue else _echo(request)
        return ModelResponse(
            text=text,
            model=self.model,
            stop_reason="end_turn",
            input_tokens=sum(len(m.get("content", "")) for m in request.messages) // 4,
            output_tokens=len(text) // 4,
        )


class RefusingProvider:
    """Always refuses — used to exercise the fallback chain in tests."""

    name = "refusing"

    def __init__(self, model: str = "refuser"):
        self.model = model

    def complete(self, request: ModelRequest) -> ModelResponse:  # noqa: ARG002
        return ModelResponse(text="", model=self.model, stop_reason="refusal")


def _echo(request: ModelRequest) -> str:
    last = request.messages[-1]["content"] if request.messages else ""
    return f"[fake:{request.effort.value}] {last[:200]}"
