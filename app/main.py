"""
FastAPI Application - API endpoint for your system
Deploy as web service
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from app.services.pipeline import pipeline
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hallucination Reduction LLM System",
    description="RAG + Verification system for trustworthy answers",
    version="1.0.0"
)

# CORS middleware (for web frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models


class QueryRequest(BaseModel):
    query: str
    return_sources: Optional[bool] = True


class QueryResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    confidence_level: str
    verification: Dict[str, Any]
    sources: Optional[List[Dict]]
    processing_time: float


@app.get("/")
async def root():
    return {
        "message": "Hallucination Reduction LLM System",
        "status": "running",
        "endpoints": ["/ask", "/health", "/verify"]
    }


@app.get("/health")
async def health_check():
    import psutil
    return {
        "status": "healthy",
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent
    }


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Main endpoint: Ask a question and get verified answer
    """
    start_time = time.time()

    try:
        # Process through pipeline
        result = pipeline.process_query(
            query=request.query,
            return_sources=request.return_sources
        )

        processing_time = time.time() - start_time

        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            confidence=result["confidence"],
            confidence_level=result["confidence_level"],
            verification=result["verification"],
            sources=result.get("sources"),
            processing_time=round(processing_time, 3)
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verify-only")
async def verify_only(query: str, answer: str, context: str):
    """
    Endpoint for verification only (for research/ablation studies)
    """
    from app.services.verifier import verifier_service

    verification = verifier_service.verify(context, answer)

    return {
        "verification": verification,
        "is_hallucination": verification['is_hallucination']
    }

# Run with: uvicorn app.main:app --reload
