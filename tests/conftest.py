import pytest

from core import model_loader


class FakePipeline:
    """
    Stands in for the Hugging Face pipeline.

    The real one is a 250 MB download; the suite must stay offline and fast, so
    every test drives this instead and asserts on how it was called.
    """

    def __init__(self, label: str = "POSITIVE", score: float = 0.9876):
        self.label = label
        self.score = score
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return [{"label": self.label, "score": self.score}]


@pytest.fixture
def fake_pipeline():
    return FakePipeline()


@pytest.fixture(autouse=True)
def clear_model_cache():
    """The loader is a module-level singleton; leaking it across tests hides bugs."""
    model_loader.reset_model()
    yield
    model_loader.reset_model()
