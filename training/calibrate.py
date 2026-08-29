"""
Chooses the decision threshold on the graph that actually gets served.

Quantisation shifts the probability scale, so an operating point picked on the
PyTorch checkpoint does not land in the same place on the int8 export. Selecting
it here, after the export, is what keeps the threshold and the scores it is
compared against on the same scale.

Usage: python -m training.calibrate
"""

import json

import mlflow
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from core.irony import IronyClassifier
from training import config
from training.data import load_splits
from training.metrics import best_threshold, classification_metrics, ironic_probability

BATCH = 64
ARGMAX = 0.5


def served_scores(split) -> np.ndarray:
    """Ironic probabilities from the int8 graph, read exactly as the service reads it."""
    classifier = IronyClassifier(
        onnx_path=str(config.ONNX_INT8),
        tokenizer_source=str(config.ONNX_DIR),
        threshold=ARGMAX,
        max_length=config.MAX_LENGTH,
    )
    texts = list(split["text"])
    scores = []
    for start in range(0, len(texts), BATCH):
        scores += [result["score"] for result in classifier.predict(texts[start : start + BATCH])]
    return np.asarray(scores)


def checkpoint_scores(split) -> np.ndarray:
    """The same probabilities from the unquantised checkpoint, for the comparison below."""
    tokenizer = AutoTokenizer.from_pretrained(str(config.TORCH_MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(config.TORCH_MODEL_DIR)
    ).eval()

    texts = list(split["text"])
    logits = []
    for start in range(0, len(texts), BATCH):
        encoded = tokenizer(
            texts[start : start + BATCH],
            truncation=True,
            max_length=config.MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits.append(model(**encoded).logits.numpy())
    return ironic_probability(np.concatenate(logits))


def main() -> None:
    if not config.ONNX_INT8.exists():
        raise SystemExit(f"No export at {config.ONNX_INT8}. Run training.export first.")

    split = load_splits()["validation"]
    labels = np.asarray(split["label"])
    scores = served_scores(split)

    threshold, criterion_value = best_threshold(labels, scores, config.CALIBRATION_CRITERION)

    # The same criterion on the unquantised checkpoint, which is the measurement
    # that puts this step after the export rather than inside training.
    checkpoint_threshold, _ = best_threshold(
        labels, checkpoint_scores(split), config.CALIBRATION_CRITERION
    )
    selected = classification_metrics(labels, (scores >= threshold).astype(int))
    argmax = classification_metrics(labels, (scores >= ARGMAX).astype(int))

    decision = {
        "threshold": threshold,
        "artifact": config.ONNX_INT8.name,
        "criterion": config.CALIBRATION_CRITERION,
        "selected_on": "validation",
        f"validation_{config.CALIBRATION_CRITERION}": criterion_value,
    }
    (config.ONNX_DIR / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    mlflow.set_tracking_uri(config.MLFLOW_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="calibration"):
        mlflow.log_params(
            {
                "artifact": config.ONNX_INT8.name,
                "criterion": config.CALIBRATION_CRITERION,
                "threshold": threshold,
                "checkpoint_threshold": checkpoint_threshold,
            }
        )
        mlflow.log_metrics({f"validation_{name}": value for name, value in selected.items()})
        mlflow.log_metrics({f"argmax_{name}": value for name, value in argmax.items()})

    header = (
        f"{'Threshold':<12}{'Accuracy':>10}{'Macro F1':>10}"
        f"{'F1 ironic':>11}{'Precision':>11}{'Recall':>9}"
    )
    print(f"Validation split ({len(labels)} tweets)\n")
    print(header)
    print("-" * len(header))
    for name, point, metrics in (("argmax", ARGMAX, argmax), ("selected", threshold, selected)):
        print(
            f"{name} {point:<5.2f}{metrics['accuracy']:>10.3f}{metrics['f1_macro']:>10.3f}"
            f"{metrics['f1_ironic']:>11.3f}{metrics['precision_ironic']:>11.3f}"
            f"{metrics['recall_ironic']:>9.3f}"
        )
    print(
        f"\nSame criterion on the unquantised checkpoint: {checkpoint_threshold:.2f} "
        f"against {threshold:.2f} on the served graph"
    )
    print(f"Wrote {config.ONNX_DIR / 'decision.json'}")


if __name__ == "__main__":
    main()
