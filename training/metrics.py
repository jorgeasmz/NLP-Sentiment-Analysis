"""Metric definitions shared by the fine-tuned model, the baselines and the benchmark."""

import numpy as np
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

IRONIC = 1


def classification_metrics(labels, predictions) -> dict:
    """
    Reports F1 on the ironic class alongside the aggregate figures.

    Accuracy and macro-F1 both reward a model that plays the majority class, and
    the useful question here is how well the minority label is recovered, so the
    ironic-class figures are the ones model selection reads.
    """
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro")),
        "f1_ironic": float(f1_score(labels, predictions, pos_label=IRONIC, zero_division=0)),
        "precision_ironic": float(
            precision_score(labels, predictions, pos_label=IRONIC, zero_division=0)
        ),
        "recall_ironic": float(
            recall_score(labels, predictions, pos_label=IRONIC, zero_division=0)
        ),
    }


def trainer_metrics(eval_prediction) -> dict:
    """Adapter for the Trainer, which hands over raw logits."""
    logits, labels = eval_prediction
    return classification_metrics(labels, np.asarray(logits).argmax(axis=-1))


def confusion(labels, predictions) -> np.ndarray:
    return confusion_matrix(labels, predictions, labels=[0, IRONIC])


def ironic_probability(logits) -> np.ndarray:
    return softmax(np.asarray(logits), axis=-1)[:, IRONIC]


def best_threshold(labels, probabilities, criterion: str = "f1_macro") -> tuple[float, float]:
    """
    Returns the threshold maximising the given metric, and the value it reaches.

    Argmax is the threshold 0.5, which is only optimal when the prior the model
    was fitted on matches the one it is asked about. The default criterion is
    macro F1 because both error directions carry cost for a flag: a missed
    ironic text leaves a wrong sentiment label unqualified, and a false alarm on
    plain text makes the flag uninformative. Ironic-class F1 prices only the
    first of those.
    """
    grid = np.linspace(0.05, 0.95, 91)
    scored = [
        (classification_metrics(labels, (probabilities >= t).astype(int))[criterion], float(t))
        for t in grid
    ]
    best_value, threshold = max(scored)
    return threshold, best_value
