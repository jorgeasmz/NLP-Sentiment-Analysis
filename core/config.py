"""
Configuration settings for the Core ML module.
"""

import os

# Overridable so the same image can serve a different checkpoint without a
# rebuild; docker-compose sets it.
MODEL_NAME = os.getenv(
    "MODEL_NAME", "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

# DistilBERT accepts 512 tokens. Longer input raises unless truncation is
# requested explicitly, so the limit is declared rather than discovered.
MAX_LENGTH = 512
TRUNCATION = True

# The irony head is distributed as a model repository instead of being committed:
# the graph is too large for git, and the Hub versions it and carries its card.
IRONY_MODEL_REPO = os.getenv("IRONY_MODEL_REPO", "jorgeasmz/distilbert-irony-tweeteval")
IRONY_ONNX_FILE = os.getenv("IRONY_ONNX_FILE", "model-int8.onnx")

# Points at a local export instead, which is how a candidate is served before it
# is published.
IRONY_MODEL_DIR = os.getenv("IRONY_MODEL_DIR", "")

# Tweets are short and the head was trained at this width; anything longer is
# truncated rather than rejected, matching the sentiment path.
IRONY_MAX_LENGTH = 96

# Empty means the threshold chosen during training travels with the artifact.
# Setting it overrides that choice without retraining.
IRONY_THRESHOLD = os.getenv("IRONY_THRESHOLD", "")
