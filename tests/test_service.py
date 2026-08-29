from core import service
from core.config import MAX_LENGTH


def patch_heads(monkeypatch, pipeline, irony):
    monkeypatch.setattr(service, "get_model", lambda: pipeline)
    monkeypatch.setattr(service, "get_irony_model", lambda: irony)


def test_returns_sentiment_and_irony(monkeypatch, fake_pipeline, fake_irony):
    patch_heads(monkeypatch, fake_pipeline, fake_irony)

    result = service.analyze_text("what a great day")

    assert result == {
        "label": "POSITIVE",
        "score": fake_pipeline.score,
        "irony": {"detected": False, "score": fake_irony.score},
    }


def test_score_is_a_plain_float(monkeypatch, fake_pipeline, fake_irony):
    """Response serialisation should not depend on a numpy scalar type."""
    patch_heads(monkeypatch, fake_pipeline, fake_irony)

    result = service.analyze_text("fine")

    assert type(result["score"]) is float
    assert type(result["irony"]["score"]) is float


def test_truncation_is_requested_explicitly(monkeypatch, fake_pipeline, fake_irony):
    """Without it the tokenizer raises on input beyond the 512-token window."""
    patch_heads(monkeypatch, fake_pipeline, fake_irony)

    service.analyze_text("word " * 2000)

    _, kwargs = fake_pipeline.calls[0]
    assert kwargs["truncation"] is True
    assert kwargs["max_length"] == MAX_LENGTH


def test_reads_the_first_prediction(monkeypatch, fake_pipeline, fake_irony):
    fake_pipeline.label = "NEGATIVE"
    patch_heads(monkeypatch, fake_pipeline, fake_irony)

    assert service.analyze_text("awful")["label"] == "NEGATIVE"


def test_irony_verdict_is_independent_of_the_sentiment_label(
    monkeypatch, fake_pipeline, fake_irony
):
    """The failure the head exists for: a confident positive label on ironic text."""
    fake_pipeline.label = "POSITIVE"
    fake_pipeline.score = 1.0
    fake_irony.ironic = True
    fake_irony.score = 0.93
    patch_heads(monkeypatch, fake_pipeline, fake_irony)

    result = service.analyze_text("Oh brilliant, another update that breaks everything.")

    assert result["label"] == "POSITIVE"
    assert result["irony"] == {"detected": True, "score": 0.93}


def test_the_irony_head_receives_a_batch_of_one(monkeypatch, fake_pipeline, fake_irony):
    """The classifier is batch-shaped; the service must not hand it a bare string."""
    patch_heads(monkeypatch, fake_pipeline, fake_irony)

    service.analyze_text("hello")

    assert fake_irony.calls == [["hello"]]
