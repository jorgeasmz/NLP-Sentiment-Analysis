"""
Compares the runtimes the irony head can be served on: size, latency and quality.

Every runtime is held to one intra-op thread and decides at argmax, so the
differences reported come from the execution graph and the weight precision.
Argmax rather than the shipped threshold: that threshold is selected on the int8
graph, and scoring the others against it would credit quantisation for the fit
between the two.

Usage: python -m training.benchmark
"""

import statistics
import time

import mlflow
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from core.irony import IronyClassifier
from training import config
from training.data import load_splits
from training.metrics import classification_metrics

IRONIC = 1
ARGMAX = 0.5
LATENCY_SAMPLE = 200
WARMUP = 10


class TorchRuntime:
    """The saved checkpoint executed by PyTorch, which is the export's reference."""

    name = "torch-fp32"

    def __init__(self, model_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval()
        self.size_mb = sum(f.stat().st_size for f in model_dir.glob("*.safetensors")) / 1e6

    def scores(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(
            texts, truncation=True, max_length=config.MAX_LENGTH, padding=True, return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(**encoded).logits
        return torch.softmax(logits, dim=-1)[:, IRONIC].numpy()


class OnnxRuntime:
    """The exported graph, served exactly as the API serves it."""

    def __init__(self, name, onnx_path, threshold):
        self.name = name
        self.size_mb = onnx_path.stat().st_size / 1e6
        self.classifier = IronyClassifier(
            onnx_path=str(onnx_path),
            tokenizer_source=str(config.ONNX_DIR),
            threshold=threshold,
            max_length=config.MAX_LENGTH,
        )

    def scores(self, texts: list[str]) -> np.ndarray:
        return np.asarray([result["score"] for result in self.classifier.predict(texts)])


def latency_ms(runtime, texts: list[str]) -> dict:
    """Single-item latency after warm-up, which is the shape of an API call."""
    for text in texts[:WARMUP]:
        runtime.scores([text])

    measured = []
    for text in texts[:LATENCY_SAMPLE]:
        started = time.perf_counter()
        runtime.scores([text])
        measured.append((time.perf_counter() - started) * 1000)

    ordered = sorted(measured)
    return {
        "latency_p50_ms": statistics.median(ordered),
        "latency_p95_ms": ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)],
    }


def evaluate(runtime, texts, labels) -> dict:
    probabilities = np.concatenate(
        [runtime.scores(texts[start : start + 32]) for start in range(0, len(texts), 32)]
    )
    metrics = classification_metrics(labels, (probabilities >= ARGMAX).astype(int))
    metrics["size_mb"] = runtime.size_mb
    metrics.update(latency_ms(runtime, texts))
    return metrics


def main() -> None:
    # A single thread is what a free-tier container effectively gets, and it is
    # the only setting under which the runtimes are comparable at all.
    torch.set_num_threads(1)

    split = load_splits()["test"]
    texts = list(split["text"])
    labels = np.asarray(split["label"])

    runtimes = [
        TorchRuntime(config.TORCH_MODEL_DIR),
        OnnxRuntime("onnx-fp32", config.ONNX_FP32, ARGMAX),
        OnnxRuntime("onnx-int8", config.ONNX_INT8, ARGMAX),
    ]

    mlflow.set_tracking_uri(config.MLFLOW_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    results = {}
    for runtime in runtimes:
        results[runtime.name] = evaluate(runtime, texts, labels)
        with mlflow.start_run(run_name=runtime.name):
            mlflow.log_params({"runtime": runtime.name, "threshold": ARGMAX, "threads": 1})
            mlflow.log_metrics(results[runtime.name])

    header = (
        f"{'Runtime':<14}{'Size MB':>10}{'p50 ms':>9}{'p95 ms':>9}"
        f"{'Accuracy':>10}{'F1 ironic':>11}"
    )
    print("\n" + header)
    print("-" * len(header))
    for name, metrics in results.items():
        print(
            f"{name:<14}{metrics['size_mb']:>10.1f}{metrics['latency_p50_ms']:>9.1f}"
            f"{metrics['latency_p95_ms']:>9.1f}{metrics['accuracy']:>10.3f}"
            f"{metrics['f1_ironic']:>11.3f}"
        )


if __name__ == "__main__":
    main()
