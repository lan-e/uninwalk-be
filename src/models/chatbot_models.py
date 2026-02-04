"""Pydantic models for chatbot API requests and responses."""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    message: str
    timestamp: datetime


class InitializationStatus(BaseModel):
    """Status model for chatbot initialization."""

    is_initialized: bool
    status: str
    error: Optional[str] = None


class EvaluationSample(BaseModel):
    """Single sample for RAGAS evaluation."""

    question: str


class EvaluationRequest(BaseModel):
    """Request model for RAGAS evaluation endpoint."""

    samples: List[EvaluationSample]


class SampleResult(BaseModel):
    """Result for a single evaluated sample."""

    question: str
    answer: str
    contexts: List[str]
    metrics: Dict[str, Any]


class EvaluationResponse(BaseModel):
    """Response model for RAGAS evaluation endpoint."""

    results: List[SampleResult]
    aggregate_scores: Dict[str, float]
