import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from api.schemas import SentimentRequest, SentimentResponse
from core.model_loader import get_model
from core.service import analyze_text

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pre-loads the model so the first caller does not pay for it.

    A failure is recorded rather than raised: the service still answers the
    health check, which is what lets an orchestrator report a degraded state
    instead of restarting the container in a loop.
    """
    logger.info("Server starting up: pre-loading NLP model.")
    try:
        get_model()
        app.state.model_ready = True
        logger.info("NLP model loaded successfully.")
    except Exception:
        logger.exception("Failed to load model on startup.")
        app.state.model_ready = False

    yield

    logger.info("Server shutting down.")


app = FastAPI(
    title="NLP Sentiment Analysis API",
    description="A lightweight API for real-time text classification using DistilBERT.",
    version="1.0.0",
    lifespan=lifespan,
)

# Set before startup so the health check answers even if lifespan never ran.
app.state.model_ready = False


@app.get("/")
def health_check(request: Request):
    """Health check. Reports whether the model finished loading."""
    return {
        "status": "ok",
        "message": "Sentiment Analysis API is running",
        "model_ready": request.app.state.model_ready,
    }


@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: SentimentRequest):
    """
    Receives text, runs the model and returns the sentiment with its confidence.
    """
    try:
        return analyze_text(request.text)
    except Exception:
        # Logged in full server-side; the caller gets no internals.
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=500, detail="Internal server error during prediction"
        ) from None


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
