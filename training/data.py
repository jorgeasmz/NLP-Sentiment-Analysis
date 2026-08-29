"""Loads the TweetEval irony split and encodes it for DistilBERT."""

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

from training.config import BASE_CHECKPOINT, DATASET_CONFIG, DATASET_ID, MAX_LENGTH


def load_splits() -> DatasetDict:
    """Returns the official train/validation/test splits, unmodified."""
    return load_dataset(DATASET_ID, DATASET_CONFIG)


def build_tokenizer(checkpoint: str = BASE_CHECKPOINT):
    return AutoTokenizer.from_pretrained(checkpoint)


def encode(splits: DatasetDict, tokenizer) -> DatasetDict:
    """
    Tokenises without padding so the collator can pad each batch to its own
    longest sequence: on CPU, padding everything to 96 tokens wastes most of the
    attention computation on a corpus whose median tweet is under 30.
    """

    def apply(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    return splits.map(apply, batched=True, remove_columns=["text"])


def build_collator(tokenizer) -> DataCollatorWithPadding:
    return DataCollatorWithPadding(tokenizer=tokenizer)
