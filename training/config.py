"""Hyperparameters and paths for the irony head, in one place so a run is reproducible."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# TweetEval ships an official split, so no resampling happens here: results stay
# comparable with the numbers published for the benchmark.
DATASET_ID = "cardiffnlp/tweet_eval"
DATASET_CONFIG = "irony"

BASE_CHECKPOINT = "distilbert/distilbert-base-uncased"
LABELS = ["non_ironic", "ironic"]

# Tweets are capped at 280 characters; 96 word-pieces covers the split's 99.9th
# percentile and keeps the attention matrix small enough to train on CPU.
MAX_LENGTH = 96

LEARNING_RATE = 2e-5
BATCH_SIZE = 16
EPOCHS = 5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42

# The benchmark reports F1 on the ironic class, so checkpoint selection uses the
# same quantity: it keeps these numbers comparable with the published ones, and
# picking on accuracy would favour a model that abstains from the minority label
# it is meant to find.
METRIC_FOR_BEST = "eval_f1_ironic"
EARLY_STOPPING_PATIENCE = 2

# The operating point answers a different question from the checkpoint, so it is
# selected on a different quantity. See training/metrics.best_threshold.
CALIBRATION_CRITERION = "f1_macro"

ARTIFACTS = ROOT / "artifacts"

# Each run writes to its own directory; this names the one that gets exported
# and served, so promoting a different run is a one-line change.
SERVED_RUN = "full"
TORCH_MODEL_DIR = ARTIFACTS / f"irony-{SERVED_RUN}"
ONNX_DIR = ARTIFACTS / "irony-onnx"
ONNX_FP32 = ONNX_DIR / "model.onnx"
ONNX_INT8 = ONNX_DIR / "model-int8.onnx"

# MLflow 3 put the filesystem store into maintenance mode, so tracking goes to
# SQLite. The path is relative: the scripts are run from the repository root.
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT = "irony-detection"
