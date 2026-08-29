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


class IronyAssessment(BaseModel):
    """
    Defines the irony verdict that accompanies every sentiment label.
    """
    detected: bool = Field(
        ...,
        description="True when the irony head scores above its threshold, in which "
        "case the sentiment label describes the surface wording rather than the intent",
    )
    score: float = Field(..., description="Probability that the text is ironic (0.0 to 1.0)")


class SentimentResponse(BaseModel):
    """
    Defines the structure of the output data returned by the API.
    """
    label: str = Field(..., description="The sentiment label (POSITIVE or NEGATIVE)")
    score: float = Field(..., description="The confidence score (0.0 to 1.0)")
    irony: IronyAssessment
