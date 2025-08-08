"""
PydanticAI Universal RAG API - Modern Agent Architecture
=======================================================

FastAPI endpoints using proper PydanticAI agent delegation:
- Domain Intelligence Agent for query analysis
- Universal Search Agent for multi-modal search
- Universal Orchestrator for agent coordination
- Zero hardcoded domain assumptions
"""

import asyncio

# Use orchestrator pattern for proper agent communication
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.core.universal_deps import get_universal_deps
from agents.knowledge_extraction.agent import run_knowledge_extraction
from agents.orchestrator import UniversalOrchestrator

# Create router
router = APIRouter(prefix="/api/v1", tags=["search"])


# PydanticAI API Models
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    max_results: int = Field(10, ge=1, le=50, description="Maximum results to return")
    use_domain_analysis: bool = Field(
        True, description="Enable domain intelligence for query optimization"
    )
    include_agent_metrics: bool = Field(
        False, description="Include agent performance metrics in response"
    )


class SearchResult(BaseModel):
    title: str
    content: str
    score: float
    source: str
    metadata: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    success: bool
    query: str
    results: List[SearchResult]
    total_results_found: int
    search_confidence: float
    strategy_used: str
    execution_time: float
    timestamp: str
    agent_metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class KnowledgeExtractionRequest(BaseModel):
    content: str = Field(..., description="Content to extract knowledge from")
    use_domain_analysis: bool = Field(
        True, description="Enable domain intelligence for extraction optimization"
    )


class KnowledgeExtractionResponse(BaseModel):
    success: bool
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    extraction_confidence: float
    processing_signature: str
    execution_time: float
    timestamp: str
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    services_available: List[str]
    total_services: int
    agent_status: Dict[str, str]
    timestamp: str


@router.post("/search", response_model=SearchResponse)
async def search_content(request: SearchRequest) -> SearchResponse:
    """Universal search endpoint using PydanticAI agents"""
    start_time = time.time()

    try:
        # Use orchestrator for proper agent coordination
        orchestrator = UniversalOrchestrator()
        workflow_result = await orchestrator.process_full_search_workflow(
            request.query,
            max_results=request.max_results,
            use_domain_analysis=request.use_domain_analysis,
        )

        if not workflow_result.success:
            raise HTTPException(
                status_code=500,
                detail=f"Search workflow failed: {'; '.join(workflow_result.errors)}",
            )

        search_result = (
            workflow_result  # Use workflow result instead of direct agent result
        )

        execution_time = time.time() - start_time

        # Convert orchestrator workflow result to API response format
        search_results = []
        if workflow_result.search_results:
            for result in workflow_result.search_results:
                search_results.append(
                    SearchResult(
                        title=result.title,
                        content=result.content,
                        score=result.score,
                        source=result.source,
                        metadata=getattr(result, "metadata", None),
                    )
                )

        # Prepare agent metrics if requested
        agent_metrics = None
        if request.include_agent_metrics:
            agent_metrics = {
                "total_processing_time": workflow_result.total_processing_time,
                "agent_metrics": workflow_result.agent_metrics,
                "search_results_count": len(search_results),
            }

        return SearchResponse(
            success=True,
            query=request.query,
            results=search_results,
            total_results_found=len(search_results),
            search_confidence=workflow_result.agent_metrics.get(
                "universal_search", {}
            ).get("search_confidence", 0.8),
            strategy_used=workflow_result.agent_metrics.get("universal_search", {}).get(
                "strategy_used", "multi-modal"
            ),
            execution_time=execution_time,
            timestamp=datetime.now().isoformat(),
            agent_metrics=agent_metrics,
        )

    except Exception as e:
        execution_time = time.time() - start_time
        return SearchResponse(
            success=False,
            query=request.query,
            results=[],
            total_results_found=0,
            search_confidence=0.0,
            strategy_used="failed",
            execution_time=execution_time,
            timestamp=datetime.now().isoformat(),
            error=str(e),
        )


@router.post("/extract", response_model=KnowledgeExtractionResponse)
async def extract_knowledge(
    request: KnowledgeExtractionRequest,
) -> KnowledgeExtractionResponse:
    """Knowledge extraction endpoint using PydanticAI agents"""
    start_time = time.time()

    try:
        # Run knowledge extraction using proper PydanticAI agent
        extraction_result = await run_knowledge_extraction(
            request.content, use_domain_analysis=request.use_domain_analysis
        )

        execution_time = time.time() - start_time

        # Convert entities and relationships to API format
        entities_data = []
        for entity in extraction_result.entities:
            entities_data.append(
                {
                    "text": entity.text,
                    "type": entity.type,
                    "confidence": entity.confidence,
                    "context": entity.context,
                    "properties": entity.properties or {},
                }
            )

        relationships_data = []
        for rel in extraction_result.relationships:
            relationships_data.append(
                {
                    "source_entity": rel.source_entity,
                    "target_entity": rel.target_entity,
                    "relationship_type": rel.relationship_type,
                    "confidence": rel.confidence,
                    "context": rel.context,
                    "properties": rel.properties or {},
                }
            )

        return KnowledgeExtractionResponse(
            success=True,
            entities=entities_data,
            relationships=relationships_data,
            extraction_confidence=extraction_result.extraction_confidence,
            processing_signature=extraction_result.processing_signature,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        execution_time = time.time() - start_time
        return KnowledgeExtractionResponse(
            success=False,
            entities=[],
            relationships=[],
            extraction_confidence=0.0,
            processing_signature="failed",
            execution_time=execution_time,
            timestamp=datetime.now().isoformat(),
            error=str(e),
        )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Comprehensive health check using Universal Dependencies"""
    try:
        # Get universal dependencies to check service availability
        deps = await get_universal_deps()
        available_services = deps.get_available_services()

        # Check agent status using orchestrator
        agent_status = {}
        try:
            # Test agents through orchestrator (proper pattern)
            orchestrator = UniversalOrchestrator()

            # Test Domain Intelligence
            domain_result = await orchestrator.process_content_with_domain_analysis(
                "health check"
            )
            agent_status["domain_intelligence"] = (
                "healthy" if domain_result.success else "degraded"
            )

            # Test Knowledge Extraction
            extraction_result = (
                await orchestrator.process_knowledge_extraction_workflow(
                    "health check", use_domain_analysis=False
                )
            )
            agent_status["knowledge_extraction"] = (
                "healthy" if extraction_result.success else "degraded"
            )

            # Test Universal Search
            search_result = await orchestrator.process_full_search_workflow(
                "health check", max_results=1, use_domain_analysis=False
            )
            agent_status["universal_search"] = (
                "healthy" if search_result.success else "degraded"
            )
        except Exception as e:
            # Fallback status if orchestrator fails
            agent_status = {
                "domain_intelligence": "unknown",
                "knowledge_extraction": "unknown",
                "universal_search": "unknown",
                "orchestrator_error": str(e),
            }

        overall_status = (
            "healthy"
            if all(status == "healthy" for status in agent_status.values())
            else "degraded"
        )

        return HealthResponse(
            status=overall_status,
            services_available=available_services,
            total_services=len(available_services),
            agent_status=agent_status,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        return HealthResponse(
            status="error",
            services_available=[],
            total_services=0,
            agent_status={"error": str(e)},
            timestamp=datetime.now().isoformat(),
        )
