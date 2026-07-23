from schema_repro.providers.base import FallbackProvider, ModelRequest
from schema_repro.providers.fake import FakeProvider, RefusingProvider
from schema_repro.types import Effort


def _req():
    return ModelRequest(system="s", messages=[{"role": "user", "content": "hi"}], effort=Effort.HIGH)


def test_fallback_advances_past_refusal():
    # Primary refuses (the shape of the Fable-5 safety refusal), secondary answers.
    chain = FallbackProvider(chain=[RefusingProvider("primary"), FakeProvider(model="secondary")])
    resp = chain.complete(_req())
    assert resp.model == "secondary"
    assert not resp.refused


def test_no_fallback_when_primary_succeeds():
    primary = FakeProvider(model="primary")
    chain = FallbackProvider(chain=[primary, FakeProvider(model="secondary")])
    resp = chain.complete(_req())
    assert resp.model == "primary"


def test_fallback_advances_past_exception():
    class Boom:
        name = "boom"

        def complete(self, request):
            raise RuntimeError("api down")

    chain = FallbackProvider(chain=[Boom(), FakeProvider(model="backup")])
    resp = chain.complete(_req())
    assert resp.model == "backup"
