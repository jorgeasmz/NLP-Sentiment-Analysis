from transformers import pipeline
from core.config import MODEL_NAME
import logging

# Configure basic logging to see when the model is loading in the console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to hold the loaded model in memory
_model_pipeline = None

def get_model():
    """
    Returns the loaded Hugging Face pipeline. 
    Loads it into memory if it hasn't been loaded yet (Singleton pattern).
    """
    global _model_pipeline

    if _model_pipeline is None:
        logger.info(f"Loading NLP Model ({MODEL_NAME})...")
        try:
            _model_pipeline = pipeline("sentiment-analysis", model=MODEL_NAME)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e
    
    return _model_pipeline