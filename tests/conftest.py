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


class FakeIrony:
    """
    Stands in for the ONNX irony head.

    Loading the real one means a 65 MB graph on disk and an ONNX Runtime
    session, neither of which a unit test should need.
    """

    def __init__(self, ironic: bool = False, score: float = 0.1234):
        self.ironic = ironic
        self.score = score
        self.calls = []

    def predict(self, texts):
        self.calls.append(texts)
        return [{"ironic": self.ironic, "score": self.score} for _ in texts]


@pytest.fixture
def fake_pipeline():
    return FakePipeline()


@pytest.fixture
def fake_irony():
    return FakeIrony()


@pytest.fixture(autouse=True)
def clear_model_cache():
    """The loader is a module-level singleton; leaking it across tests hides bugs."""
    model_loader.reset_model()
    yield
    model_loader.reset_model()
