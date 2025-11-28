from pydantic import BaseModel
from typing import List, Optional, Literal, Dict

Severity = Literal['low', 'medium', 'high']
AccuracyLevel = Literal['Low', 'Moderate', 'High']

class AnalyzeRequest(BaseModel):
    symptoms: List[str]
    age: Optional[int] = None
    gender: Optional[str] = None

class AnalyzeResponse(BaseModel):
    severity: Severity
    insights: List[str]
    advice: str
    redFlags: List[str] = []
    probabilities: Optional[Dict[str, float]] = None
    topCondition: Optional[str] = None
    conditionDetails: Optional[str] = None
    treatment: Optional[str] = None
    accuracyLevel: AccuracyLevel