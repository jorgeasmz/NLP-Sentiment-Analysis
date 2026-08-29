import threading

from core import model_loader


def test_loads_the_model_only_once(monkeypatch, fake_pipeline):
    calls = []

    def fake_factory(task, model):
        calls.append((task, model))
        return fake_pipeline

    monkeypatch.setattr(model_loader, "pipeline", fake_factory)

    first = model_loader.get_model()
    second = model_loader.get_model()

    assert first is second
    assert len(calls) == 1


def test_requests_the_configured_checkpoint(monkeypatch, fake_pipeline):
    seen = {}

    def fake_factory(task, model):
        seen["task"] = task
        seen["model"] = model
        return fake_pipeline

    monkeypatch.setattr(model_loader, "pipeline", fake_factory)

    model_loader.get_model()

    assert seen["task"] == "sentiment-analysis"
    assert seen["model"] == model_loader.MODEL_NAME


def test_reset_forces_a_reload(monkeypatch, fake_pipeline):
    calls = []
    monkeypatch.setattr(
        model_loader, "pipeline", lambda task, model: calls.append(1) or fake_pipeline
    )

    model_loader.get_model()
    model_loader.reset_model()
    model_loader.get_model()

    assert len(calls) == 2


def test_concurrent_callers_still_load_once(monkeypatch, fake_pipeline):
    """FastAPI serves sync endpoints from a threadpool; the loader must hold."""
    calls = []

    def slow_factory(task, model):
        # Widen the race window the lock is there to close.
        threading.Event().wait(0.05)
        calls.append(1)
        return fake_pipeline

    monkeypatch.setattr(model_loader, "pipeline", slow_factory)

    threads = [threading.Thread(target=model_loader.get_model) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(calls) == 1


def test_irony_artifact_prefers_a_local_directory(monkeypatch, tmp_path):
    """A candidate export has to be servable before it is published to the Hub."""
    (tmp_path / "decision.json").write_text('{"threshold": 0.37}')
    monkeypatch.setattr(model_loader, "IRONY_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(model_loader, "IRONY_THRESHOLD", "")

    onnx_path, tokenizer_source, threshold = model_loader._resolve_irony_artifact()

    assert onnx_path == str(tmp_path / model_loader.IRONY_ONNX_FILE)
    assert tokenizer_source == str(tmp_path)
    assert threshold == 0.37


def test_the_threshold_travels_with_the_artifact(monkeypatch, tmp_path):
    """It is chosen during training, so serving reads it instead of restating it."""
    (tmp_path / "decision.json").write_text('{"threshold": 0.61}')
    monkeypatch.setattr(model_loader, "IRONY_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(model_loader, "IRONY_THRESHOLD", "")

    assert model_loader._resolve_irony_artifact()[2] == 0.61


def test_the_environment_overrides_the_shipped_threshold(monkeypatch, tmp_path):
    (tmp_path / "decision.json").write_text('{"threshold": 0.61}')
    monkeypatch.setattr(model_loader, "IRONY_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(model_loader, "IRONY_THRESHOLD", "0.8")

    assert model_loader._resolve_irony_artifact()[2] == 0.8


def test_a_missing_decision_file_falls_back_to_argmax(monkeypatch, tmp_path):
    monkeypatch.setattr(model_loader, "IRONY_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(model_loader, "IRONY_THRESHOLD", "")

    assert model_loader._resolve_irony_artifact()[2] == model_loader.DEFAULT_THRESHOLD


def test_loads_the_irony_head_only_once(monkeypatch, fake_irony):
    calls = []

    monkeypatch.setattr(
        model_loader, "_resolve_irony_artifact", lambda: ("graph.onnx", "tokenizer", 0.5)
    )
    monkeypatch.setattr(
        model_loader,
        "IronyClassifier",
        lambda **kwargs: calls.append(kwargs) or fake_irony,
    )

    first = model_loader.get_irony_model()
    second = model_loader.get_irony_model()

    assert first is second
    assert len(calls) == 1
    assert calls[0]["max_length"] == model_loader.IRONY_MAX_LENGTH


def test_reset_drops_both_heads(monkeypatch, fake_pipeline, fake_irony):
    monkeypatch.setattr(model_loader, "pipeline", lambda task, model: fake_pipeline)
    monkeypatch.setattr(
        model_loader, "_resolve_irony_artifact", lambda: ("graph.onnx", "tokenizer", 0.5)
    )
    monkeypatch.setattr(model_loader, "IronyClassifier", lambda **kwargs: fake_irony)

    model_loader.get_model()
    model_loader.get_irony_model()
    model_loader.reset_model()

    assert model_loader._model_pipeline is None
    assert model_loader._irony_classifier is None
