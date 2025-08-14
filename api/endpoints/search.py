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
import json

# Use orchestrator pattern for proper agent communication
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.core.universal_deps import get_universal_deps
from agents.knowledge_extraction.agent import run_knowledge_extraction
from agents.orchestrator import UniversalOrchestrator

# Global orchestrator instance to preserve domain analysis cache across requests
_global_orchestrator: Optional[UniversalOrchestrator] = None

def get_orchestrator() -> UniversalOrchestrator:
    """Get or create global orchestrator instance with persistent domain analysis cache."""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = UniversalOrchestrator()
    return _global_orchestrator

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
    use_generated_prompts: bool = Field(
        False, description="Enable auto-generated prompts (requires Azure CLI authentication)"
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
        # Use GLOBAL orchestrator for proper agent coordination and domain analysis caching
        orchestrator = get_orchestrator()
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
                    "properties": entity.metadata or {},
                }
            )

        relationships_data = []
        for rel in extraction_result.relationships:
            relationships_data.append(
                {
                    "source_entity": rel.source,
                    "target_entity": rel.target,
                    "relationship_type": rel.relation,
                    "confidence": rel.confidence,
                    "context": rel.context,
                    "properties": rel.metadata or {},
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


@router.get("/stream/workflow/{query_id}")
async def stream_workflow_progress(query_id: str):
    """
    REAL Azure streaming endpoint - NO FAKE CODE

    Streams real workflow progress from Azure Universal RAG agents:
    - Domain Intelligence Agent analysis from REAL Azure OpenAI
    - Knowledge Extraction from REAL Cosmos DB
    - Universal Search from REAL Azure Cognitive Search
    - Uses REAL data from data/raw/azure-ai-services-language-service_output/
    """

    async def generate_real_workflow_events():
        """Stream REAL workflow events from Azure services - NO SIMULATION"""
        try:
            # Initialize REAL Azure dependencies
            deps = await get_universal_deps()
            orchestrator = UniversalOrchestrator()

            # Send connection established event
            yield f"data: {json.dumps({'event_type': 'connection_established', 'query_id': query_id, 'timestamp': datetime.now().isoformat()})}\n\n"

            # Get REAL data from data/raw directory
            data_dir = (
                Path(__file__).parent.parent.parent
                / "data"
                / "raw"
                / "azure-ai-services-language-service_output"
            )
            azure_files = list(data_dir.glob("*.md"))

            if not azure_files:
                yield f"data: {json.dumps({'event_type': 'error', 'error': 'No REAL Azure data files found in data/raw', 'query_id': query_id})}\n\n"
                return

            # Use REAL Azure data file for processing
            real_azure_file = azure_files[0]  # Use first available real file
            real_content = real_azure_file.read_text(encoding="utf-8", errors="ignore")[
                :1000
            ]

            # Step 1: REAL Domain Intelligence Agent
            yield f"data: {json.dumps({'event_type': 'progress', 'query_id': query_id, 'step_number': 1, 'step_name': 'domain_intelligence', 'user_friendly_name': 'Analyzing with REAL Azure OpenAI', 'status': 'in_progress', 'technology': 'Azure OpenAI GPT-4', 'details': f'Processing REAL file: {real_azure_file.name}', 'progress_percentage': 25})}\n\n"

            domain_result = await orchestrator.process_content_with_domain_analysis(
                real_content
            )

            if not domain_result.success:
                yield f"data: {json.dumps({'event_type': 'error', 'error': f'REAL Domain Intelligence FAILED: {domain_result.errors}', 'query_id': query_id})}\n\n"
                return

            yield f"data: {json.dumps({'event_type': 'progress', 'query_id': query_id, 'step_number': 1, 'step_name': 'domain_intelligence', 'user_friendly_name': 'Azure OpenAI Analysis Complete', 'status': 'completed', 'technology': 'Azure OpenAI GPT-4', 'details': f'Domain signature: {domain_result.domain_analysis.domain_signature}', 'progress_percentage': 25, 'processing_time_ms': domain_result.total_processing_time * 1000})}\n\n"

            # Step 2: REAL Knowledge Extraction with Cosmos DB
            yield f"data: {json.dumps({'event_type': 'progress', 'query_id': query_id, 'step_number': 2, 'step_name': 'knowledge_extraction', 'user_friendly_name': 'Extracting with REAL Cosmos DB', 'status': 'in_progress', 'technology': 'Azure Cosmos DB Gremlin', 'details': 'Extracting entities from REAL Azure data', 'progress_percentage': 50})}\n\n"

            extraction_result = (
                await orchestrator.process_knowledge_extraction_workflow(
                    real_content, use_domain_analysis=True
                )
            )

            if not extraction_result.success:
                yield f"data: {json.dumps({'event_type': 'error', 'error': f'REAL Knowledge Extraction FAILED: {extraction_result.errors}', 'query_id': query_id})}\n\n"
                return

            # Get extraction summary from workflow result
            extraction_summary = extraction_result.extraction_summary or {}
            entity_count = extraction_summary.get("entity_count", 0)
            relationship_count = extraction_summary.get("relationship_count", 0)

            yield f"data: {json.dumps({'event_type': 'progress', 'query_id': query_id, 'step_number': 2, 'step_name': 'knowledge_extraction', 'user_friendly_name': 'Cosmos DB Extraction Complete', 'status': 'completed', 'technology': 'Azure Cosmos DB Gremlin', 'details': f'Extracted {entity_count} entities, {relationship_count} relationships', 'progress_percentage': 50, 'processing_time_ms': extraction_result.total_processing_time * 1000})}\n\n"

            # Step 3: REAL Universal Search with Azure Cognitive Search
            yield f"data: {json.dumps({'event_type': 'progress', 'query_id': query_id, 'step_number': 3, 'step_name': 'universal_search', 'user_friendly_name': 'Searching with REAL Azure Cognitive Search', 'status': 'in_progress', 'technology': 'Azure Cognitive Search + Vector Search', 'details': f'Searching across {len(azure_files)} REAL Azure documents', 'progress_percentage': 75})}\n\n"

            search_result = await orchestrator.process_full_search_workflow(
                query_id, max_results=10, use_domain_analysis=True
            )

            if not search_result.success:
                yield f"data: {json.dumps({'event_type': 'error', 'error': f'REAL Universal Search FAILED: {search_result.errors}', 'query_id': query_id})}\n\n"
                return

            yield f"data: {json.dumps({'event_type': 'progress', 'query_id': query_id, 'step_number': 3, 'step_name': 'universal_search', 'user_friendly_name': 'Azure Cognitive Search Complete', 'status': 'completed', 'technology': 'Azure Cognitive Search + Vector Search', 'details': f'Found {len(search_result.search_results)} REAL results', 'progress_percentage': 100, 'processing_time_ms': search_result.total_processing_time * 1000})}\n\n"

            # Final completion with REAL results
            total_processing_time = (
                domain_result.total_processing_time
                + extraction_result.total_processing_time
                + search_result.total_processing_time
            )
            search_confidence = search_result.agent_metrics.get(
                "universal_search", {}
            ).get("search_confidence", 0.0)

            yield f"data: {json.dumps({'event_type': 'workflow_completed', 'query_id': query_id, 'query': query_id, 'generated_response': f'REAL Azure workflow completed: processed {real_azure_file.name}, extracted {entity_count} entities, found {len(search_result.search_results)} search results', 'confidence_score': search_confidence, 'processing_time': total_processing_time, 'safety_warnings': [], 'sources': [str(real_azure_file)], 'citations': [real_content[:200] + '...']})}\n\n"

        except Exception as e:
            # QUICK FAIL - Real error from Azure services
            yield f"data: {json.dumps({'event_type': 'workflow_failed', 'query_id': query_id, 'error': f'REAL Azure services FAILED: {str(e)}'})}\n\n"

    return StreamingResponse(
        generate_real_workflow_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )
