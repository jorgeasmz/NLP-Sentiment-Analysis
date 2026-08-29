import json
import logging
import threading
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import pipeline

from core.config import (
    IRONY_MAX_LENGTH,
    IRONY_MODEL_DIR,
    IRONY_MODEL_REPO,
    IRONY_ONNX_FILE,
    IRONY_THRESHOLD,
    MODEL_NAME,
)
from core.irony import IronyClassifier

logger = logging.getLogger(__name__)

DECISION_FILE = "decision.json"
DEFAULT_THRESHOLD = 0.5

_model_pipeline = None
_irony_classifier = None
_load_lock = threading.Lock()
_irony_lock = threading.Lock()


def get_model():
    """
    Returns the sentiment pipeline, loading it at most once.

    FastAPI runs synchronous endpoints in a worker threadpool, so two
    concurrent first requests could each start loading a 250 MB checkpoint.
    The lock makes the load happen once; the check outside it keeps the common
    path lock-free.
    """
    global _model_pipeline

    if _model_pipeline is None:
        with _load_lock:
            if _model_pipeline is None:
                logger.info("Loading NLP model (%s)...", MODEL_NAME)
                _model_pipeline = pipeline("sentiment-analysis", model=MODEL_NAME)
                logger.info("Model loaded successfully.")

    return _model_pipeline


def _resolve_irony_artifact() -> tuple[str, str, float]:
    """
    Locates the graph, the tokeniser and the decision threshold.

    A local directory takes precedence so an export can be served before it is
    published; otherwise the files come from the model repository.
    """
    if IRONY_MODEL_DIR:
        source = Path(IRONY_MODEL_DIR)
        onnx_path = source / IRONY_ONNX_FILE
        decision_path = source / DECISION_FILE
    else:
        source = IRONY_MODEL_REPO
        onnx_path = Path(hf_hub_download(IRONY_MODEL_REPO, IRONY_ONNX_FILE))
        decision_path = Path(hf_hub_download(IRONY_MODEL_REPO, DECISION_FILE))

    if IRONY_THRESHOLD:
        threshold = float(IRONY_THRESHOLD)
    elif decision_path.exists():
        threshold = float(json.loads(decision_path.read_text())["threshold"])
    else:
        threshold = DEFAULT_THRESHOLD

    return str(onnx_path), str(source), threshold


def get_irony_model() -> IronyClassifier:
    """Returns the irony classifier, loading it at most once. Locked as above."""
    global _irony_classifier

    if _irony_classifier is None:
        with _irony_lock:
            if _irony_classifier is None:
                onnx_path, tokenizer_source, threshold = _resolve_irony_artifact()
                logger.info("Loading irony head (%s, threshold %.2f)...", onnx_path, threshold)
                _irony_classifier = IronyClassifier(
                    onnx_path=onnx_path,
                    tokenizer_source=tokenizer_source,
                    threshold=threshold,
                    max_length=IRONY_MAX_LENGTH,
                )
                logger.info("Irony head loaded successfully.")

    return _irony_classifier


def reset_model() -> None:
    """Drops the cached pipelines. Exists so tests can exercise the loaders."""
    global _model_pipeline, _irony_classifier
    _model_pipeline = None
    _irony_classifier = None
