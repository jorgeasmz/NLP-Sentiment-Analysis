import numpy as np
import pytest

from core import irony


class FakeTokenizer:
    """Returns fixed ids and records the call, standing in for the real vocabulary."""

    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        width = 4
        return {
            "input_ids": np.ones((len(texts), width), dtype=np.int32),
            "attention_mask": np.ones((len(texts), width), dtype=np.int32),
            # Present in some tokenisers and absent from the exported graph, so
            # the classifier has to drop it rather than pass it through.
            "token_type_ids": np.zeros((len(texts), width), dtype=np.int32),
        }


class FakeInput:
    def __init__(self, name):
        self.name = name


class FakeSession:
    def __init__(self, logits):
        self.logits = logits
        self.feeds = []

    def get_inputs(self):
        return [FakeInput("input_ids"), FakeInput("attention_mask")]

    def run(self, outputs, feeds):
        self.feeds.append(feeds)
        return [np.asarray(self.logits)]


@pytest.fixture
def build(monkeypatch):
    def make(logits, threshold=0.5):
        tokenizer = FakeTokenizer()
        session = FakeSession(logits)
        monkeypatch.setattr(
            irony.AutoTokenizer, "from_pretrained", staticmethod(lambda source: tokenizer)
        )
        monkeypatch.setattr(irony.onnxruntime, "InferenceSession", lambda *a, **k: session)
        classifier = irony.IronyClassifier(
            onnx_path="graph.onnx",
            tokenizer_source="tokenizer",
            threshold=threshold,
            max_length=96,
        )
        return classifier, tokenizer, session

    return make


def test_softmax_is_stable_on_large_logits():
    """Exponentiating raw logits overflows; the shift is what keeps this finite."""
    probabilities = irony._softmax(np.array([[1000.0, 999.0]]))

    assert np.isfinite(probabilities).all()
    assert probabilities.sum() == pytest.approx(1.0)


def test_reports_the_ironic_probability(build):
    classifier, _, _ = build([[0.0, np.log(3.0)]])

    result = classifier.predict(["oh good"])[0]

    assert result["score"] == pytest.approx(0.75)
    assert result["ironic"] is True


def test_the_threshold_decides(build):
    classifier, _, _ = build([[0.0, np.log(3.0)]], threshold=0.8)

    assert classifier.predict(["oh good"])[0]["ironic"] is False


def test_drops_inputs_the_graph_does_not_declare(build):
    """Feeding an undeclared tensor makes ONNX Runtime raise on every call."""
    classifier, _, session = build([[0.0, 1.0]])

    classifier.predict(["anything"])

    assert set(session.feeds[0]) == {"input_ids", "attention_mask"}


def test_feeds_int64_tensors(build):
    """The exported graph declares int64 inputs; int32 is rejected at run time."""
    classifier, _, session = build([[0.0, 1.0]])

    classifier.predict(["anything"])

    assert all(tensor.dtype == np.int64 for tensor in session.feeds[0].values())


def test_truncates_and_pads_the_batch(build):
    classifier, tokenizer, _ = build([[0.0, 1.0], [1.0, 0.0]])

    results = classifier.predict(["short", "a much longer piece of text"])

    _, kwargs = tokenizer.calls[0]
    assert kwargs["truncation"] is True
    assert kwargs["padding"] is True
    assert kwargs["max_length"] == 96
    assert len(results) == 2


def test_scores_are_plain_floats(build):
    classifier, _, _ = build([[0.0, 1.0]])

    result = classifier.predict(["x"])[0]

    assert type(result["score"]) is float
    assert type(result["ironic"]) is bool


def test_serving_truncates_at_the_training_width():
    """
    Train and serve read the same number of tokens.

    The head never saw a sequence longer than the training width, so a serving
    path that truncates somewhere else feeds it inputs it was not fitted on.
    """
    from core.config import IRONY_MAX_LENGTH
    from training.config import MAX_LENGTH

    assert IRONY_MAX_LENGTH == MAX_LENGTH
