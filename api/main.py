from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from api.schemas import SentimentRequest, SentimentResponse
from core.service import analyze_text, get_model
import logging

# Initialize logging
logger = logging.getLogger("uvicorn")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI app.
    """
    logger.info("Server starting up... Pre-loading NLP Model.")
    try:
        get_model()
        logger.info("NLP Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
    
    yield
    
    logger.info("Server shutting down...")

app = FastAPI(
    title="NLP Sentiment Analysis API",
    description="A lightweight API for real-time text classification using DistilBERT.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def health_check():
    """
    Simple health check endpoint to verify the service is up.
    """
    return {"status": "ok", "message": "Sentiment Analysis API is running"}

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: SentimentRequest):
    """
    Receives text, runs the ML model, and returns the sentiment.
    """
    try:
        result = analyze_text(request.text)
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        # Return a 500 Internal Server Error if something goes wrong in the ML core
        raise HTTPException(status_code=500, detail="Internal server error during prediction")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)