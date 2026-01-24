from core.model_loader import get_model

def analyze_text(text: str) -> dict:
    """
    Analyzes the sentiment of the input text.

    Args:
        text (str): The string to analyze.

    Returns:
        dict: A dictionary with 'label' (POSITIVE/NEGATIVE) and 'score' (float).
    """
    # 1. Get the model
    pipeline = get_model()

    # 2. Perform inference
    
    # The pipeline returns a list of dicts: [{'label': 'POSITIVE', 'score': 0.99}]
    results = pipeline(text)

    # 3. Extract the first result
    prediction = results[0]

    return {
        "label": prediction['label'],
        "score": prediction['score']
    }