import json

import numpy as np
import pytest

from training import release


def test_the_digest_is_stable_across_calls():
    texts = ["a tweet", "another one"]
    labels = np.array([1, 0])

    assert release.holdout_digest(texts, labels) == release.holdout_digest(texts, labels)


def test_a_changed_text_changes_the_digest():
    labels = np.array([1, 0])

    first = release.holdout_digest(["a tweet", "another one"], labels)
    second = release.holdout_digest(["a tweet", "another one!"], labels)

    assert first != second


def test_a_changed_label_changes_the_digest():
    texts = ["a tweet", "another one"]

    first = release.holdout_digest(texts, np.array([1, 0]))
    second = release.holdout_digest(texts, np.array([0, 0]))

    assert first != second


def test_reordering_changes_the_digest():
    """The split is official and ordered, so its order is part of its identity."""
    first = release.holdout_digest(["a", "b"], np.array([1, 0]))
    second = release.holdout_digest(["b", "a"], np.array([0, 1]))

    assert first != second


def test_texts_and_labels_of_different_lengths_are_refused():
    with pytest.raises(ValueError):
        release.holdout_digest(["only one"], np.array([1, 0]))


def test_the_measurement_reports_the_metric_the_gate_turns_on(monkeypatch):
    """Scored at the served threshold, not at argmax: the gate compares what ships."""
    labels = np.array([1, 1, 0, 0])
    probabilities = np.array([0.9, 0.7, 0.9, 0.1])

    monkeypatch.setattr(release, "IronyClassifier", lambda **kwargs: object())
    monkeypatch.setattr(release, "scores", lambda classifier, texts: probabilities)

    record = release.measure(0.84, ["a", "b", "c", "d"], labels)

    assert record["metric"] == "f1_ironic"
    assert record["threshold"] == 0.84
    assert record["rows"] == 4
    # At 0.84 two are called ironic, one of them correctly: one true positive,
    # one false positive, one false negative.
    assert record["value"] == pytest.approx(0.5)


def test_a_different_threshold_gives_a_different_measurement(monkeypatch):
    labels = np.array([1, 1, 0, 0])
    probabilities = np.array([0.9, 0.7, 0.9, 0.1])

    monkeypatch.setattr(release, "IronyClassifier", lambda **kwargs: object())
    monkeypatch.setattr(release, "scores", lambda classifier, texts: probabilities)

    served = release.measure(0.84, ["a", "b", "c", "d"], labels)
    argmax = release.measure(0.50, ["a", "b", "c", "d"], labels)

    assert served["value"] != argmax["value"]


def test_the_record_carries_the_evidence_the_gate_needs(monkeypatch, tmp_path):
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps({"threshold": 0.84, "artifact": "model-int8.onnx"}))

    monkeypatch.setattr(release, "DECISION_PATH", decision)
    monkeypatch.setattr(release, "holdout", lambda: (["a", "b"], np.array([1, 0])))
    monkeypatch.setattr(release, "measure", lambda threshold, texts, labels: {
        "metric": "f1_ironic", "value": 0.6279, "threshold": threshold, "rows": len(labels),
    })

    record = release.build_record()

    assert record["dataset"] == release.holdout_digest(["a", "b"], np.array([1, 0]))
    assert record["artifact"] == "model-int8.onnx"
    assert record["threshold"] == 0.84
    assert "packages" in record["environment"]


def test_the_recorded_environment_covers_what_the_catalogue_declares(monkeypatch, tmp_path):
    """The platform refuses to serve an artifact whose declared packages moved."""
    from registry.models import get

    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps({"threshold": 0.84, "artifact": "model-int8.onnx"}))
    monkeypatch.setattr(release, "DECISION_PATH", decision)
    monkeypatch.setattr(release, "holdout", lambda: (["a"], np.array([1])))
    monkeypatch.setattr(release, "measure", lambda *a: {"metric": "f1_ironic", "value": 0.1})

    recorded = release.build_record()["environment"]["packages"]

    assert set(recorded) == set(get("irony").packages)
