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
