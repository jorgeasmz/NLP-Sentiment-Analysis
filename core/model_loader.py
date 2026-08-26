import logging
import threading

from transformers import pipeline

from core.config import MODEL_NAME

logger = logging.getLogger(__name__)

_model_pipeline = None
_load_lock = threading.Lock()


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


def reset_model() -> None:
    """Drops the cached pipeline. Exists so tests can exercise the loader."""
    global _model_pipeline
    _model_pipeline = None
