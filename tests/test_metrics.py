import numpy as np
import pytest

from training.metrics import best_threshold, classification_metrics, ironic_probability


def test_ironic_class_metrics_ignore_the_majority():
    """Accuracy stays high while the ironic class is never recovered."""
    labels = np.array([0] * 9 + [1])
    predictions = np.zeros(10, dtype=int)

    metrics = classification_metrics(labels, predictions)

    assert metrics["accuracy"] == pytest.approx(0.9)
    assert metrics["f1_ironic"] == 0.0
    assert metrics["recall_ironic"] == 0.0


def test_perfect_predictions_score_one():
    labels = np.array([0, 1, 1, 0])

    metrics = classification_metrics(labels, labels)

    assert metrics["f1_ironic"] == 1.0
    assert metrics["f1_macro"] == 1.0


def test_ironic_probability_reads_the_second_column():
    """Column order follows the label mapping; reading the wrong one inverts the head."""
    probabilities = ironic_probability(np.array([[0.0, np.log(3.0)], [np.log(3.0), 0.0]]))

    assert probabilities[0] == pytest.approx(0.75)
    assert probabilities[1] == pytest.approx(0.25)


def test_best_threshold_finds_the_separating_cut():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    threshold, score = best_threshold(labels, probabilities)

    assert score == 1.0
    assert 0.2 < threshold <= 0.8


def test_best_threshold_beats_argmax_when_the_scale_shifts():
    """The reason the operating point is selected rather than assumed to be 0.5."""
    labels = np.array([1, 1, 1, 0])
    probabilities = np.array([0.3, 0.35, 0.45, 0.1])

    threshold, score = best_threshold(labels, probabilities)

    argmax = classification_metrics(labels, (probabilities >= 0.5).astype(int))["f1_macro"]
    assert threshold < 0.5
    assert score > argmax


def test_the_criterion_changes_the_operating_point():
    """Ironic-class F1 prices only missed irony, so it buys recall with false alarms."""
    labels = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    probabilities = np.array([0.4, 0.45, 0.1, 0.2, 0.3, 0.35, 0.5, 0.6, 0.7, 0.8])

    balanced, _ = best_threshold(labels, probabilities, "f1_macro")
    ironic, _ = best_threshold(labels, probabilities, "f1_ironic")

    assert ironic < balanced


def test_best_threshold_returns_a_value_from_the_grid():
    labels = np.array([0, 1])
    probabilities = np.array([0.4, 0.6])

    threshold, _ = best_threshold(labels, probabilities)

    assert 0.05 <= threshold <= 0.95
