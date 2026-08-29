from core.config import MAX_LENGTH, TRUNCATION
from core.model_loader import get_irony_model, get_model


def analyze_text(text: str) -> dict:
    """
    Classifies one string for sentiment and for irony.

    Truncation is passed explicitly: without it the tokenizer raises on input
    longer than the model's 512-token window instead of trimming it, so a
    pasted article would surface as a 500 rather than a result.

    The two heads answer independently. A sentence-level sentiment classifier
    reads surface lexical cues, so ironic text inverts its label while leaving
    the confidence untouched; the irony flag is what tells a caller that the
    sentiment label was produced under exactly those conditions.

    Returns:
        dict: 'label' (POSITIVE/NEGATIVE), 'score' (confidence in that label)
        and 'irony' ('detected' and its probability).
    """
    model = get_model()

    results = model(text, truncation=TRUNCATION, max_length=MAX_LENGTH)
    prediction = results[0]

    irony = get_irony_model().predict([text])[0]

    return {
        "label": prediction["label"],
        "score": float(prediction["score"]),
        "irony": {"detected": irony["ironic"], "score": irony["score"]},
    }
