"""
Query processing endpoints for MaintIE Enhanced RAG API
Handles maintenance query requests with validation and response formatting
"""

import logging
import time
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, HTTPException, Depends, Query as QueryParam
from pydantic import BaseModel, Field, validator

from src.pipeline.enhanced_rag import get_rag_instance
from src.models.maintenance_models import RAGResponse


logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


class QueryRequest(BaseModel):
    """Request model for maintenance queries"""

    query: str = Field(
        ...,
        min_length=3,  # Will be overridden by config
        max_length=500,  # Will be overridden by config
        description="Maintenance query in natural language"
    )
    max_results: Optional[int] = Field(
        default=10,
        ge=1,
        le=50,  # Will be overridden by config
        description="Maximum number of search results to return"
    )
    include_explanations: Optional[bool] = Field(
        default=True,
        description="Include explanations and reasoning in response"
    )
    enable_safety_warnings: Optional[bool] = Field(
        default=True,
        description="Include safety warnings in response"
    )
    response_format: Optional[str] = Field(
        default="detailed",
        description="Response format: 'detailed', 'summary', or 'minimal'"
    )

    def __init__(self, **data):
        """Initialize with configurable validation limits"""
        from config.settings import settings

        super().__init__(**data) # Call super().__init__ first

        # Override validation limits with configurable values
        # Access fields using self.model_fields (Pydantic v2 way)
        if 'query' in self.model_fields:
            if self.model_fields['query'].json_schema_extra is None:
                self.model_fields['query'].json_schema_extra = {}
            self.model_fields['query'].json_schema_extra['min_length'] = settings.query_min_length
            self.model_fields['query'].json_schema_extra['max_length'] = settings.query_max_length

        if 'max_results' in self.model_fields:
            if self.model_fields['max_results'].json_schema_extra is None:
                self.model_fields['max_results'].json_schema_extra = {}
            self.model_fields['max_results'].json_schema_extra['le'] = settings.max_results_limit

    @validator('query')
    def validate_query(cls, v):
        """Validate query content"""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

    @validator('response_format')
    def validate_response_format(cls, v):
        """Validate response format"""
        valid_formats = ['detailed', 'summary', 'minimal']
        if v not in valid_formats:
            raise ValueError(f"response_format must be one of: {valid_formats}")
        return v


class QueryResponse(BaseModel):
    """Response model for maintenance queries"""

    query: str = Field(description="Original query")
    response: str = Field(description="Generated maintenance response")
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the response"
    )
    processing_time: float = Field(description="Processing time in seconds")

    # Enhanced query information
    query_analysis: Dict[str, Any] = Field(description="Query analysis results")
    expanded_concepts: List[str] = Field(description="Expanded maintenance concepts")

    # Search and sources
    sources: List[str] = Field(description="Source document IDs")
    citations: List[str] = Field(description="Formatted citations")
    search_results_count: int = Field(description="Number of search results used")

    # Safety and quality
    safety_warnings: List[str] = Field(description="Safety warnings and considerations")
    quality_indicators: Dict[str, Any] = Field(description="Response quality indicators")

    # Metadata
    timestamp: float = Field(description="Response timestamp")
    model_info: Dict[str, Any] = Field(description="Model and system information")


class QuerySuggestionResponse(BaseModel):
    """Response model for query suggestions"""

    suggestions: List[str] = Field(description="Suggested maintenance queries")
    categories: Dict[str, List[str]] = Field(description="Suggestions by category")


def get_rag_system():
    """Dependency to get RAG system instance"""
    rag_system = get_rag_instance()
    if not rag_system.components_initialized:
        raise HTTPException(
            status_code=503,
            detail="RAG system components not initialized. Please wait for system startup to complete."
        )
    return rag_system


@router.post("/query", response_model=QueryResponse)
async def process_maintenance_query(
    request: QueryRequest,
    rag_system=Depends(get_rag_system)
) -> QueryResponse:
    """
    Process maintenance query and return enhanced response

    This endpoint processes natural language maintenance queries using the enhanced RAG pipeline:
    1. Analyzes and understands the maintenance query
    2. Expands concepts using domain knowledge
    3. Retrieves relevant documentation using multi-modal search
    4. Generates contextually appropriate responses
    5. Adds safety warnings and citations
    """

    start_time = time.time()

    try:
        logger.info(f"Processing maintenance query: {request.query}")

        # Process query through RAG pipeline
        rag_response = rag_system.process_query(
            query=request.query,
            max_results=request.max_results,
            include_explanations=request.include_explanations,
            enable_safety_warnings=request.enable_safety_warnings
        )

        # Format response based on requested format
        formatted_response = _format_response(rag_response, request.response_format)

        # Build quality indicators
        quality_indicators = _build_quality_indicators(rag_response)

        # Build model information
        from config.settings import settings
        model_info = {
            "rag_version": "1.0.0",
            "embedding_model": settings.embedding_model,
            "llm_model": settings.openai_model,
            "knowledge_base": "MaintIE",
            "pipeline_components": [
                "query_enhancement",
                "multi_modal_retrieval",
                "domain_generation"
            ]
        }

        response = QueryResponse(
            query=request.query,
            response=formatted_response,
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

        logger.info(f"Query processed successfully in {rag_response.processing_time:.2f}s")
        return response

    except Exception as e:
        logger.error(f"Error processing query '{request.query}': {e}", exc_info=True)
        processing_time = time.time() - start_time

        # Return error response with helpful information
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Query processing failed",
                "message": str(e),
                "query": request.query,
                "processing_time": processing_time,
                "suggestions": [
                    "Try rephrasing your query",
                    "Use more specific maintenance terminology",
                    "Check if the system is fully initialized"
                ]
            }
        )


@router.get("/query/suggestions", response_model=QuerySuggestionResponse)
async def get_query_suggestions(
    category: Optional[str] = QueryParam(
        None,
        description="Query category: 'troubleshooting', 'preventive', 'procedural', 'safety'"
    ),
    equipment: Optional[str] = QueryParam(
        None,
        description="Equipment type: 'pump', 'motor', 'compressor', 'valve', etc."
    )
) -> QuerySuggestionResponse:
    """
    Get maintenance query suggestions

    Returns commonly used maintenance queries organized by category and equipment type.
    Useful for guiding users on how to formulate effective maintenance queries.
    """

    try:
        # Build suggestions based on category and equipment
        suggestions = _build_query_suggestions(category, equipment)

        # Organize by categories
        categories = {
            "troubleshooting": [
                "How to diagnose pump seal failure?",
                "Troubleshooting motor overheating issues",
                "Compressor vibration analysis procedure",
                "Valve leakage root cause analysis"
            ],
            "preventive": [
                "Preventive maintenance schedule for centrifugal pumps",
                "Motor bearing lubrication intervals",
                "Heat exchanger cleaning procedures",
                "Valve inspection checklist"
            ],
            "procedural": [
                "Step-by-step pump impeller replacement",
                "Motor alignment procedure",
                "Pressure relief valve testing steps",
                "Bearing installation best practices"
            ],
            "safety": [
                "Electrical motor safety procedures",
                "Pressure system isolation steps",
                "Chemical handling safety for maintenance",
                "Lockout/tagout procedures for pumps"
            ]
        }

        # Filter categories if specified
        if category and category in categories:
            filtered_categories = {category: categories[category]}
        else:
            filtered_categories = categories

        return QuerySuggestionResponse(
            suggestions=suggestions,
            categories=filtered_categories
        )

    except Exception as e:
        logger.error(f"Error generating query suggestions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating suggestions: {str(e)}"
        )


@router.get("/query/examples")
async def get_query_examples():
    """
    Get example maintenance queries with expected outcomes

    Provides example queries that demonstrate the system's capabilities
    across different maintenance scenarios.
    """

    examples = [
        {
            "query": "How to troubleshoot centrifugal pump seal failure?",
            "type": "troubleshooting",
            "equipment": "pump",
            "expected_features": [
                "Step-by-step diagnostic procedure",
                "Common failure causes",
                "Required tools and safety equipment",
                "Safety warnings for pressure systems"
            ]
        },
        {
            "query": "Preventive maintenance schedule for electric motors",
            "type": "preventive",
            "equipment": "motor",
            "expected_features": [
                "Maintenance frequency recommendations",
                "Inspection checklist",
                "Lubrication requirements",
                "Performance monitoring parameters"
            ]
        },
        {
            "query": "Safety procedures for high-pressure system maintenance",
            "type": "safety",
            "equipment": "pressure_system",
            "expected_features": [
                "Comprehensive safety protocols",
                "PPE requirements",
                "Isolation procedures",
                "Emergency response guidance"
            ]
        }
    ]

    return {
        "examples": examples,
        "usage_tips": [
            "Be specific about equipment type and issue",
            "Include context about urgency or criticality",
            "Mention specific failure symptoms or observations",
            "Ask for specific information (procedures, schedules, safety)"
        ]
    }


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
        "concept_expansion": len(rag_response.enhanced_query.expanded_concepts)
    }

    return indicators


def _build_query_suggestions(category: Optional[str], equipment: Optional[str]) -> List[str]:
    """Build query suggestions based on filters"""

    base_suggestions = [
        "How to troubleshoot equipment failure?",
        "Preventive maintenance schedule recommendations",
        "Safety procedures for maintenance tasks",
        "Step-by-step repair procedures",
        "Root cause analysis methods"
    ]

    if equipment:
        equipment_suggestions = {
            "pump": [
                "Pump seal replacement procedure",
                "Centrifugal pump troubleshooting guide",
                "Pump performance monitoring",
                "Pump cavitation prevention"
            ],
            "motor": [
                "Motor bearing maintenance schedule",
                "Electric motor troubleshooting",
                "Motor alignment procedures",
                "Motor insulation testing"
            ],
            "compressor": [
                "Compressor vibration analysis",
                "Air compressor maintenance checklist",
                "Compressor safety procedures",
                "Compressor efficiency optimization"
            ]
        }

        if equipment.lower() in equipment_suggestions:
            return equipment_suggestions[equipment.lower()]

    if category:
        category_suggestions = {
            "troubleshooting": [
                "Equipment failure diagnosis steps",
                "Common failure modes analysis",
                "Diagnostic tools and procedures",
                "Troubleshooting decision trees"
            ],
            "preventive": [
                "Maintenance scheduling best practices",
                "Inspection frequency guidelines",
                "Condition monitoring techniques",
                "Replacement interval optimization"
            ],
            "procedural": [
                "Standard operating procedures",
                "Work instruction templates",
                "Quality control checkpoints",
                "Tool and material requirements"
            ],
            "safety": [
                "Hazard identification methods",
                "Personal protective equipment requirements",
                "Emergency response procedures",
                "Risk assessment techniques"
            ]
        }

        if category.lower() in category_suggestions:
            return category_suggestions[category.lower()]

    return base_suggestions


# Health check for query processing
@router.get("/query/health")
async def query_processing_health(rag_system=Depends(get_rag_system)):
    """Check health of query processing components"""

    try:
        # Test with a simple query
        test_query = "system test"
        start_time = time.time()

        # Quick health check processing
        health_result = {
            "query_processing": "healthy",
            "components": {
                "query_analyzer": rag_system.query_analyzer is not None,
                "vector_search": rag_system.vector_search is not None,
                "llm_interface": rag_system.llm_interface is not None
            },
            "response_time_target": "< 2.0s",
            "system_ready": rag_system.components_initialized
        }

        # Optional: Run actual test query if system is ready
        if rag_system.components_initialized:
            try:
                test_response = rag_system.process_query(test_query, max_results=1)
                health_result["test_query_processing"] = "successful"
                health_result["test_response_time"] = f"{test_response.processing_time:.2f}s"
            except:
                health_result["test_query_processing"] = "failed"

        return health_result

    except Exception as e:
        logger.error(f"Query processing health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Query processing health check failed: {str(e)}"
        )
