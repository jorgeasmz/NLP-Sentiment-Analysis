from pydantic import BaseModel, Field

class SentimentRequest(BaseModel):
    """
    Defines the structure of the input data expected by the API.
    """
    text: str = Field(..., min_length=1, description="The text to analyze")

    # This example appears in the auto-generated documentation (/docs)
    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "I absolutely love this new way of learning AI!"
            }
        }
    }

class SentimentResponse(BaseModel):
    """
    Defines the structure of the output data returned by the API.
    """
    label: str = Field(..., description="The sentiment label (POSITIVE or NEGATIVE)")
    score: float = Field(..., description="The confidence score (0.0 to 1.0)")