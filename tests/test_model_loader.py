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
