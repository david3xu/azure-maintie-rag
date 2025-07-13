"""
Structured RAG query processing endpoint
Handles optimized structured retrieval approach (1 API call + graph operations)
Used for production performance and enhanced domain understanding
"""

import logging
import time
import sys
import os
from typing import Dict, List, Any

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from src.pipeline.enhanced_rag import get_rag_instance
from src.models.maintenance_models import RAGResponse
from api.models.query_models import QueryRequest, QueryResponse


logger = logging.getLogger(__name__)

# Create router for structured endpoint
router = APIRouter()


def get_rag_system():
    """Dependency to get RAG system instance"""
    rag_system = get_rag_instance()
    # Check if the structured RAG components are initialized
    if not hasattr(rag_system, 'structured_rag') or not rag_system.structured_rag.components_initialized:
        raise HTTPException(
            status_code=503,
            detail="RAG system components not initialized. Please wait for system startup to complete."
        )
    return rag_system


@router.post("/", response_model=QueryResponse)
async def process_structured_query(
    request: QueryRequest,
    rag_system=Depends(get_rag_system)
) -> QueryResponse:
    """
    Process maintenance query using optimized structured RAG approach

    This endpoint uses the new optimized retrieval method that combines:
    1. Enhanced domain understanding with structured knowledge
    2. Single API call optimization (vs 3 calls in original)
    3. Graph-enhanced ranking for better relevance
    4. Maintains same response quality with improved performance
    """

    start_time = time.time()

    try:
        logger.info(f"Processing structured query: {request.query}")

        # Use structured method (async)
        rag_response = await rag_system.process_query_structured(
            query=request.query,
            max_results=request.max_results,
            include_explanations=request.include_explanations,
            enable_safety_warnings=request.enable_safety_warnings
        )

        # Format response based on requested format
        formatted_response = _format_response(rag_response, request.response_format)

        # Build quality indicators
        quality_indicators = _build_quality_indicators(rag_response)

        # Build model information with method indicator
        from config.settings import settings
        model_info = {
            "rag_version": "1.0.0",
            "embedding_model": settings.embedding_model,
            "llm_model": settings.openai_model,
            "knowledge_base": "MaintIE",
            "pipeline_components": [
                "query_enhancement",
                "structured_retrieval",  # Different from multi_modal_retrieval
                "domain_generation"
            ],
            "retrieval_method": "optimized_structured_rag",  # Method indicator
            "api_calls": 1  # New method uses 1 vector search
        }

        response = QueryResponse(
            query=request.query,
            generated_response=formatted_response,
            confidence_score=rag_response.confidence_score,
            processing_time=rag_response.processing_time,
            query_analysis=rag_response.enhanced_query.analysis.to_dict(),
            expanded_concepts=rag_response.enhanced_query.expanded_concepts,
            sources=rag_response.sources,
            citations=rag_response.citations,
            search_results_count=len(rag_response.search_results),
            safety_warnings=rag_response.safety_warnings,
            quality_indicators=quality_indicators,
            timestamp=time.time(),
            model_info=model_info
        )

        logger.info(f"Structured query processed successfully in {rag_response.processing_time:.2f}s")
        return response

    except Exception as e:
        logger.error(f"Error processing structured query '{request.query}': {e}", exc_info=True)
        processing_time = time.time() - start_time

        # Return error response with helpful information
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Structured query processing failed",
                "message": str(e),
                "query": request.query,
                "processing_time": processing_time,
                "method": "optimized_structured_rag",
                "suggestions": [
                    "Try the original endpoint: POST /api/v1/query/multi-modal",
                    "Check if the system is fully initialized",
                    "Verify structured knowledge components are loaded"
                ]
            }
        )


def _format_response(rag_response: RAGResponse, format_type: str) -> str:
    """Format response based on requested format"""

    response = rag_response.generated_response

    if format_type == "minimal":
        # Return just the main response without additional formatting
        return response

    elif format_type == "summary":
        # Return a condensed version
        lines = response.split('\n')
        # Take first paragraph and any bullet points
        summary_lines = []
        for line in lines:
            if line.strip():
                summary_lines.append(line)
                if len(summary_lines) >= 5:  # Limit to 5 lines
                    break

        summary = '\n'.join(summary_lines)
        if len(rag_response.safety_warnings) > 0:
            summary += f"\n\n⚠️ Safety: {rag_response.safety_warnings[0]}"

        return summary

    else:  # detailed format
        # Return full response with additional formatting
        formatted = response

        # Add confidence indicator
        confidence_text = "High" if rag_response.confidence_score > 0.8 else "Medium" if rag_response.confidence_score > 0.6 else "Low"
        formatted += f"\n\n**Response Confidence:** {confidence_text} ({rag_response.confidence_score:.2f})"

        return formatted


def _build_quality_indicators(rag_response: RAGResponse) -> Dict[str, Any]:
    """Build quality indicators for the response"""

    indicators = {
        "confidence_level": "high" if rag_response.confidence_score > 0.8 else "medium" if rag_response.confidence_score > 0.6 else "low",
        "sources_used": len(rag_response.sources),
        "safety_warnings_included": len(rag_response.safety_warnings) > 0,
        "citations_provided": len(rag_response.citations) > 0,
        "response_length": len(rag_response.generated_response),
        "processing_efficiency": "fast" if rag_response.processing_time < 2.0 else "normal" if rag_response.processing_time < 5.0 else "slow",
        "concept_expansion": len(rag_response.enhanced_query.expanded_concepts),
        "retrieval_method": "optimized_structured_rag"
    }

    return indicators