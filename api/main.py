import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from api.schemas import SentimentRequest, SentimentResponse
from core.model_loader import get_irony_model, get_model
from core.service import analyze_text

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pre-loads both heads so the first caller does not pay for them.

    A failure is recorded rather than raised: the service still answers the
    health check, which is what lets an orchestrator report a degraded state
    instead of restarting the container in a loop. Every response carries an
    irony verdict, so a half-loaded service is not a usable one and readiness
    covers both models rather than reporting them separately.
    """
    logger.info("Server starting up: pre-loading models.")
    try:
        get_model()
        get_irony_model()
        app.state.model_ready = True
        logger.info("Models loaded successfully.")
    except Exception:
        logger.exception("Failed to load models on startup.")
        app.state.model_ready = False

    yield

    logger.info("Server shutting down.")


app = FastAPI(
    title="NLP Sentiment Analysis API",
    description=(
        "Real-time text classification with two DistilBERT heads: sentiment from a "
        "pre-trained checkpoint, irony from a head fine-tuned on TweetEval and served "
        "as a quantised ONNX graph."
    ),
    version="2.0.0",
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
    Receives text, runs both heads and returns the sentiment with its irony verdict.
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
