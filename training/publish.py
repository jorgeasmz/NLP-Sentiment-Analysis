"""
Publishes the exported irony head as a model repository, with a card built from the tracked run.

The metrics in the card are read from MLflow rather than typed in, so the
documented numbers and the recorded run cannot drift apart.

Usage: python -m training.publish [--repo jorgeasmz/distilbert-irony-tweeteval]
"""

import argparse
import json

import mlflow
from huggingface_hub import HfApi

from core.config import IRONY_MODEL_REPO
from training import config

CRITERION_NAMES = {"f1_macro": "macro F1", "f1_ironic": "ironic-class F1"}

CARD = """---
license: apache-2.0
language: en
library_name: onnx
pipeline_tag: text-classification
base_model: {base}
datasets:
  - {dataset}
tags:
  - irony-detection
  - onnx
  - quantized
---

# DistilBERT irony detection (int8 ONNX)

Binary irony detection for English tweets. The head is fine-tuned from
`{base}` on the `{configuration}` configuration of TweetEval, exported to ONNX
and quantised to int8 for CPU inference.

It exists to qualify a sentiment prediction rather than to replace one. A
sentence-level sentiment classifier reads surface lexical cues, so ironic text
inverts its label while leaving its confidence untouched, which makes the
confidence useless as a filter. This head answers the separate question of
whether the input is ironic at all.

## Metrics

Measured on the held-out TweetEval test split ({test_size} tweets), deciding at
argmax, which is the rule the published benchmark numbers use.

| Metric | Value |
|---|---:|
| F1, ironic class | {f1_ironic:.3f} |
| Precision, ironic class | {precision_ironic:.3f} |
| Recall, ironic class | {recall_ironic:.3f} |
| Macro F1 | {f1_macro:.3f} |
| Accuracy | {accuracy:.3f} |

**Decision threshold: {threshold:.2f}**, selected on the validation split to
maximise {criterion} and shipped in `decision.json`. It is selected on this int8
graph rather than on the checkpoint it was exported from: quantisation shifts the
probability scale, so the same criterion lands elsewhere on the two artifacts.

## Usage

```python
import json

import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

repo = "{repo}"
session = onnxruntime.InferenceSession(hf_hub_download(repo, "model-int8.onnx"))
tokenizer = AutoTokenizer.from_pretrained(repo)
threshold = json.loads(open(hf_hub_download(repo, "decision.json")).read())["threshold"]

text = "oh brilliant, another update that breaks everything"
encoded = tokenizer([text], truncation=True, max_length={max_length},
                    padding=True, return_tensors="np")
feeds = {{name: value.astype(np.int64) for name, value in encoded.items()
         if name in {{i.name for i in session.get_inputs()}}}}
logits = session.run(None, feeds)[0]
exponentiated = np.exp(logits - logits.max(-1, keepdims=True))
probability = (exponentiated / exponentiated.sum(-1, keepdims=True))[:, 1]
print(probability >= threshold)
```

## Limitations

Trained on English tweets from 2015 and after. Irony is signalled differently
across registers, and on formal written English the score distribution shifts
upward and separates the classes poorly, so the figures above characterise the
head on tweets and not beyond them. A deployment on other text needs its
threshold selected on a sample of that text.

The fp32 graph is not distributed. `training/export.py` in the source repository
reproduces it from the fine-tuned checkpoint.

## Training

Source, hyperparameters and the evaluation harness:
https://github.com/jorgeasmz/NLP-Sentiment-Analysis
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=IRONY_MODEL_REPO)
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def tracked_metrics(run_name: str) -> dict:
    """Reads the recorded test metrics for the run named in the configuration."""
    mlflow.set_tracking_uri(config.MLFLOW_URI)
    runs = mlflow.search_runs(
        experiment_names=[config.MLFLOW_EXPERIMENT],
        filter_string=f"attributes.run_name = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise SystemExit(f"No MLflow run named '{run_name}'. Run training.train first.")

    row = runs.iloc[0]
    metrics = {
        name.removeprefix("metrics.test_"): float(row[name])
        for name in runs.columns
        if name.startswith("metrics.test_")
    }
    metrics["test_size"] = int(row["params.test_size"])
    return metrics


def main() -> None:
    args = parse_args()

    decision = json.loads((config.ONNX_DIR / "decision.json").read_text())
    metrics = tracked_metrics(config.SERVED_RUN)

    card = CARD.format(
        base=config.BASE_CHECKPOINT,
        dataset=config.DATASET_ID,
        configuration=config.DATASET_CONFIG,
        repo=args.repo,
        max_length=config.MAX_LENGTH,
        threshold=decision["threshold"],
        criterion=CRITERION_NAMES.get(decision["criterion"], decision["criterion"]),
        **metrics,
    )
    (config.ONNX_DIR / "README.md").write_text(card)

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo,
        folder_path=str(config.ONNX_DIR),
        # The fp32 graph stays local: serving reads the int8 copy and the export
        # script rebuilds the other from the checkpoint.
        ignore_patterns=[config.ONNX_FP32.name],
        commit_message=f"Publish irony head from run '{config.SERVED_RUN}'",
    )
    print(f"Published to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
