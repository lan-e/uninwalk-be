"""Chatbot API endpoints."""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from src.models.chatbot_models import (
    ChatRequest,
    ChatResponse,
    InitializationStatus,
    EvaluationRequest,
    EvaluationResponse,
    SampleResult,
)
from src.dependencies.chatbot import get_chatbot_service
from src.services.chatbot_service import ChatbotService


router = APIRouter()


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, chatbot_service: ChatbotService = Depends(get_chatbot_service)):
    """
    Send a message to the chatbot and get a response.

    Args:
        request: ChatRequest with message and optional session_id

    Returns:
        ChatResponse with the bot's answer and timestamp
    """
    try:
        response = await chatbot_service.get_chat_response(request.message)
        return ChatResponse(
            message=response,
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting chatbot response: {str(e)}"
        )


@router.post("/stream")
async def chat_stream(request: ChatRequest, chatbot_service: ChatbotService = Depends(get_chatbot_service)):
    """
    Send a message to the chatbot and stream the response tokens.

    Args:
        request: ChatRequest with message and optional session_id

    Returns:
        StreamingResponse with Server-Sent Events containing response tokens
    """
    async def generate():
        try:
            async for token in chatbot_service.stream_chat_response(request.message):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.get("/status", response_model=InitializationStatus)
async def get_status(chatbot_service: ChatbotService = Depends(get_chatbot_service)):
    """
    Get the current initialization status of the chatbot.

    Returns:
        InitializationStatus with current state
    """
    status = chatbot_service.get_initialization_status()
    return InitializationStatus(**status)


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_rag(
    request: EvaluationRequest,
    chatbot_service: ChatbotService = Depends(get_chatbot_service),
):
    """
    Evaluate the RAG system using RAGAS metrics.

    Args:
        request: EvaluationRequest with list of samples (question + optional ground_truth)

    Returns:
        EvaluationResponse with per-sample results and aggregate scores
    """
    try:
        samples = [sample.model_dump() for sample in request.samples]
        eval_results = await chatbot_service.evaluate_rag(samples)
        return EvaluationResponse(
            results=[SampleResult(**r) for r in eval_results["results"]],
            aggregate_scores=eval_results["aggregate_scores"],
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error running RAGAS evaluation: {str(e)}"
        )
