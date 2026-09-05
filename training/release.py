"""What the served head scores, at the threshold it is served at.

The benchmark reports each runtime at argmax, which is the right comparison
between runtimes: it isolates the effect of the export and the quantisation from
the effect of the operating point. It is the wrong figure to promote on. The
service scores at the calibrated threshold in `decision.json`, so that is the
figure a gate deciding what to serve has to compare.

The held-out split is TweetEval's official test split, unmodified, and its digest
travels with the score. A candidate measured on anything else is refused as
unjudged rather than compared.

Usage: python -m training.release [--publish]
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
from registry.environment import capture
from registry.gate import digest_bytes
from registry.models import get

from core.irony import IronyClassifier
from training import config
from training.data import load_splits
from training.metrics import classification_metrics

log = logging.getLogger(__name__)

MODEL_NAME = "irony"
RELEASE_PATH = config.ONNX_DIR / "release.json"
DECISION_PATH = config.ONNX_DIR / "decision.json"
BATCH = 32


def holdout() -> tuple[list[str], np.ndarray]:
    split = load_splits()["test"]
    return list(split["text"]), np.asarray(split["label"])


def holdout_digest(texts: list[str], labels: np.ndarray) -> str:
    """Identifies the held-out examples, so a changed split reports as changed data."""
    body = "\n".join(
        f"{label}\t{text}" for label, text in zip(labels.tolist(), texts, strict=True)
    )
    return digest_bytes(body.encode("utf-8"))


def scores(classifier: IronyClassifier, texts: list[str]) -> np.ndarray:
    return np.concatenate([
        np.asarray([r["score"] for r in classifier.predict(texts[start : start + BATCH])])
        for start in range(0, len(texts), BATCH)
    ])


def measure(threshold: float, texts: list[str], labels: np.ndarray) -> dict:
    """The served graph at the served threshold, on the official test split."""
    classifier = IronyClassifier(
        onnx_path=str(config.ONNX_INT8),
        tokenizer_source=str(config.ONNX_DIR),
        threshold=threshold,
        max_length=config.MAX_LENGTH,
    )
    probabilities = scores(classifier, texts)
    metrics = classification_metrics(labels, (probabilities >= threshold).astype(int))

    return {
        "metric": "f1_ironic",
        "value": round(float(metrics["f1_ironic"]), 4),
        "threshold": threshold,
        "rows": int(len(labels)),
        "accuracy": round(float(metrics["accuracy"]), 4),
        "f1_macro": round(float(metrics["f1_macro"]), 4),
    }


def build_record() -> dict:
    decision = json.loads(DECISION_PATH.read_text())
    texts, labels = holdout()

    record = measure(decision["threshold"], texts, labels)
    record["dataset"] = holdout_digest(texts, labels)
    record["artifact"] = decision["artifact"]
    record["environment"] = capture(get(MODEL_NAME).packages)
    return record


def publish(repo: str) -> str:
    """Uploads the release record beside the artifact. Returns the commit."""
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.upload_file(
        path_or_fileobj=str(RELEASE_PATH),
        path_in_repo="release.json",
        repo_id=repo,
        repo_type="model",
    )
    return getattr(info, "oid", None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--repo", default=get(MODEL_NAME).repo)
    arguments = parser.parse_args()

    record = build_record()
    RELEASE_PATH.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    print(f"{record['metric']}: {record['value']} at threshold {record['threshold']}"
          f"  on {record['rows']} held-out examples")
    print(f"held-out digest: {record['dataset']}")
    print(f"written to {RELEASE_PATH}")

    if arguments.publish:
        commit = publish(arguments.repo)
        record["revision"] = commit
        RELEASE_PATH.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"published to https://huggingface.co/{arguments.repo} at {commit}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
