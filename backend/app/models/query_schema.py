# backend/app/models/query_schema.py

from pydantic import BaseModel
from typing import Optional

# For the /query endpoint
class QueryRequest(BaseModel):
    audio_base64: str  # Base64 encoded audio string
    language: Optional[str] = "en"

class QueryResponse(BaseModel):
    original_text: str
    simplified_explanation: str

# For the /feedback endpoint
class FeedbackRequest(BaseModel):
    audio_base64: str
    expected_text: str

class FeedbackResponse(BaseModel):
    accuracy_score: float
    hesitation_detected: bool
    suggestions: Optional[str]
