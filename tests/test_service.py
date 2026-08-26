from core import service
from core.config import MAX_LENGTH


def test_returns_label_and_score(monkeypatch, fake_pipeline):
    monkeypatch.setattr(service, "get_model", lambda: fake_pipeline)

    result = service.analyze_text("what a great day")

    assert result == {"label": "POSITIVE", "score": fake_pipeline.score}


def test_score_is_a_plain_float(monkeypatch, fake_pipeline):
    """Response serialisation should not depend on a numpy scalar type."""
    monkeypatch.setattr(service, "get_model", lambda: fake_pipeline)

    result = service.analyze_text("fine")

    assert type(result["score"]) is float


def test_truncation_is_requested_explicitly(monkeypatch, fake_pipeline):
    """Without it the tokenizer raises on input beyond the 512-token window."""
    monkeypatch.setattr(service, "get_model", lambda: fake_pipeline)

    service.analyze_text("word " * 2000)

    _, kwargs = fake_pipeline.calls[0]
    assert kwargs["truncation"] is True
    assert kwargs["max_length"] == MAX_LENGTH


def test_reads_the_first_prediction(monkeypatch, fake_pipeline):
    fake_pipeline.label = "NEGATIVE"
    monkeypatch.setattr(service, "get_model", lambda: fake_pipeline)

    assert service.analyze_text("awful")["label"] == "NEGATIVE"
