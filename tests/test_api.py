import pytest
from fastapi.testclient import TestClient

from api import main


@pytest.fixture
def client():
    return TestClient(main.app)


def test_health_check_reports_readiness(client):
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_ready"] is False


def test_predict_returns_the_documented_schema(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "analyze_text",
        lambda text: {
            "label": "POSITIVE",
            "score": 0.99,
            "irony": {"detected": True, "score": 0.87},
        },
    )

    response = client.post("/predict", json={"text": "wonderful"})

    assert response.status_code == 200
    assert response.json() == {
        "label": "POSITIVE",
        "score": 0.99,
        "irony": {"detected": True, "score": 0.87},
    }


def test_predict_rejects_empty_text(client):
    response = client.post("/predict", json={"text": ""})

    assert response.status_code == 422


def test_predict_rejects_a_missing_field(client):
    response = client.post("/predict", json={})

    assert response.status_code == 422


def test_predict_does_not_leak_internal_errors(client, monkeypatch):
    def explode(text):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(main, "analyze_text", explode)

    response = client.post("/predict", json={"text": "anything"})

    assert response.status_code == 500
    assert "secret internal detail" not in response.text


def test_lifespan_marks_the_model_ready(monkeypatch, fake_pipeline, fake_irony):
    monkeypatch.setattr(main, "get_model", lambda: fake_pipeline)
    monkeypatch.setattr(main, "get_irony_model", lambda: fake_irony)

    with TestClient(main.app) as client:
        assert client.get("/").json()["model_ready"] is True


def test_lifespan_survives_a_failed_load(monkeypatch, fake_irony):
    """A missing checkpoint must degrade the service, not crash-loop it."""
    def explode():
        raise OSError("no checkpoint")

    monkeypatch.setattr(main, "get_model", explode)
    monkeypatch.setattr(main, "get_irony_model", lambda: fake_irony)

    with TestClient(main.app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json()["model_ready"] is False


def test_readiness_covers_the_irony_head(monkeypatch, fake_pipeline):
    """Every response carries an irony verdict, so one head loaded is not ready."""
    def explode():
        raise OSError("no graph")

    monkeypatch.setattr(main, "get_model", lambda: fake_pipeline)
    monkeypatch.setattr(main, "get_irony_model", explode)

    with TestClient(main.app) as client:
        assert client.get("/").json()["model_ready"] is False


def test_the_example_reaches_the_openapi_document(client):
    schema = client.get("/openapi.json").json()

    assert "example" in schema["components"]["schemas"]["SentimentRequest"]
