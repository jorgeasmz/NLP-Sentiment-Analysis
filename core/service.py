from core.config import MAX_LENGTH, TRUNCATION
from core.model_loader import get_model


def analyze_text(text: str) -> dict:
    """
    Classifies the sentiment of one string.

    Truncation is passed explicitly: without it the tokenizer raises on input
    longer than the model's 512-token window instead of trimming it, so a
    pasted article would surface as a 500 rather than a result.

    Returns:
        dict: 'label' (POSITIVE/NEGATIVE) and 'score' (confidence in that label).
    """
    model = get_model()

    results = model(text, truncation=TRUNCATION, max_length=MAX_LENGTH)
    prediction = results[0]

    return {
        "label": prediction["label"],
        "score": float(prediction["score"]),
    }
