"""
Ask Endpoint
Handles POST /ask requests for querying the AI agent.
"""

import logging
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class QueryRequest(BaseModel):
    """Request model for /ask endpoint"""
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to ask the AI agent",
        examples=["What is the leave policy?"]
    )
    session_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Optional session ID for conversation memory",
        examples=["user-123-session-1"]
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the leave policy?",
                "session_id": "user-123"
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str = Field(..., description="Error message")


class ChunkResult(BaseModel):
    """Retrieved chunk with metadata and scores"""
    chunk: str = Field(..., description="Chunk text")
    source: str = Field(..., description="Source filename")
    chunk_index: int = Field(..., description="Chunk index within the source")
    score: float = Field(..., description="Similarity score (lower is better)")
    confidence: float = Field(..., description="Confidence score")

class QueryResponse(BaseModel):
    """Response model for /ask endpoint"""
    answer: str = Field(
        ...,
        description="The AI agent's response"
    )
    sources: List[str] = Field(
        default=[],
        description="List of source documents used (if any)"
    )
    chunks: List[ChunkResult] = Field(
        default=[],
        description="Retrieved chunks with metadata"
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Similarity-based confidence score"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "According to the Leave Policy, full-time employees are entitled to 20 days of annual leave per year.",
                "sources": ["leave_policy.txt"]
            }
        }


# =============================================================================
# Endpoints
# =============================================================================

@router.post(
    "/ask",
    response_model=QueryResponse,
    responses={
        200: {"description": "Successful response", "model": QueryResponse},
        400: {"description": "Bad request - Invalid query", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
    summary="Ask the AI Agent",
    description="""
Ask the AI agent a question. The agent will:
1. Determine if it needs to search documents
2. If needed, retrieve relevant information from company policies
3. Generate an appropriate response

**Session Memory**: Provide a `session_id` to maintain conversation context across multiple requests.
    """
)
async def ask_question(request: QueryRequest) -> QueryResponse:
    """
    Process user query through AI agent with RAG.
    
    Args:
        request: QueryRequest containing query and optional session_id
        
    Returns:
        QueryResponse with answer and sources
        
    Raises:
        HTTPException: For validation errors or internal errors
    """
    # Validate query is not empty/whitespace
    query = request.query.strip()
    if not query:
        logger.warning("Received empty query")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty or contain only whitespace"
        )
    
    logger.info(f"Received query: {query[:50]}..." if len(query) > 50 else f"Received query: {query}")
    
    if request.session_id:
        logger.info(f"Session ID: {request.session_id}")
    
    try:
        # Import agent here to avoid circular imports
        from app.agent.agent import ask
        
        # Process query through agent
        answer, sources, chunks, confidence = ask(query, session_id=request.session_id)
        
        logger.info(f"Generated answer ({len(answer)} chars), sources: {sources}")
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            chunks=chunks,
            confidence=confidence
        )
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent module not properly configured"
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your request: {str(e)}"
        )


@router.get(
    "/session/{session_id}/clear",
    summary="Clear Session Memory",
    description="Clear the conversation history for a specific session.",
    responses={
        200: {"description": "Session cleared successfully"},
        404: {"description": "Session not found"},
    }
)
async def clear_session(session_id: str):
    """
    Clear conversation history for a session.
    
    Args:
        session_id: The session identifier to clear
        
    Returns:
        Success message
    """
    try:
        from app.agent.memory import memory
        
        memory.clear_session(session_id)
        logger.info(f"Cleared session: {session_id}")
        
        return {"message": f"Session '{session_id}' cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear session: {str(e)}"
        )
