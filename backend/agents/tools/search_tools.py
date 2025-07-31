"""
PydanticAI Search Tools

This module converts our unique tri-modal search capabilities into PydanticAI tools.
Preserves 100% of our competitive advantage while leveraging PydanticAI's framework
for validation, execution, and orchestration.

Our Competitive Advantages Preserved:
- Tri-modal unity principle (Vector + Graph + GNN simultaneously)
- Unified result synthesis 
- Performance optimization (<3s response time)
- Advanced correlation and logging
"""

import asyncio
import time
import uuid
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import RunContext

# Import our existing tri-modal orchestrator - production only
from ..search.tri_modal_orchestrator import TriModalOrchestrator, SearchResult

# Import our Azure service container
try:
    from ..azure_integration import AzureServiceContainer
except ImportError:
    from typing import Any as AzureServiceContainer


logger = logging.getLogger(__name__)


class TriModalSearchRequest(BaseModel):
    """Request model for tri-modal search with full parameter validation"""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query text")
    search_types: List[str] = Field(
        default=["vector", "graph", "gnn"], 
        description="Search modalities to use (vector, graph, gnn)"
    )
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum results per modality")
    domain: Optional[str] = Field(default=None, description="Domain context for search optimization")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional search context")
    performance_requirements: Dict[str, float] = Field(
        default_factory=lambda: {"max_response_time": 3.0, "min_confidence": 0.7},
        description="Performance and quality requirements"
    )


class TriModalSearchResponse(BaseModel):
    """Response model for tri-modal search results"""
    results: str = Field(..., description="Unified search results")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    modality_contributions: Dict[str, Any] = Field(..., description="Individual modality contributions")
    execution_time: float = Field(..., ge=0.0, description="Total execution time in seconds")
    correlation_id: str = Field(..., description="Request correlation ID for tracking")
    performance_met: bool = Field(..., description="Whether performance requirements were met")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional result metadata")


async def execute_tri_modal_search(
    ctx: RunContext[AzureServiceContainer],
    request: TriModalSearchRequest
) -> TriModalSearchResponse:
    """
    Execute our proprietary tri-modal search (Vector + Graph + GNN) as PydanticAI tool.
    
    This tool preserves our core competitive advantage while leveraging PydanticAI's
    validation and execution framework.
    
    Features:
    - Simultaneous Vector + Graph + GNN search (tri-modal unity principle)
    - Unified result synthesis strengthening all modalities
    - Performance optimization for <3s response times
    - Advanced correlation tracking and observability
    """
    
    start_time = time.time()
    correlation_id = str(uuid.uuid4())
    
    logger.info(
        "PydanticAI tri-modal search initiated",
        extra={
            'correlation_id': correlation_id,
            'query': request.query[:100],  # Truncate for logging
            'search_types': request.search_types,
            'domain': request.domain,
            'max_results': request.max_results
        }
    )
    
    try:
        # Get tri-modal orchestrator from Azure service container
        orchestrator = ctx.deps.tri_modal_orchestrator
        
        if not orchestrator:
            raise RuntimeError("Tri-modal orchestrator not available in Azure service container")
        
        # Prepare search context
        search_context = {
            "domain": request.domain,
            "max_results": request.max_results,
            "search_types": request.search_types,
            **request.context
        }
        
        # Execute our tri-modal unified search (preserves competitive advantage)
        search_result = await orchestrator.execute_unified_search(
            query=request.query,
            context=search_context,
            correlation_id=correlation_id
        )
        
        execution_time = time.time() - start_time
        
        # Check performance requirements
        performance_met = (
            execution_time <= request.performance_requirements.get("max_response_time", 3.0) and
            search_result.confidence >= request.performance_requirements.get("min_confidence", 0.7)
        )
        
        # Format response with real search result data only
        response = TriModalSearchResponse(
            results=search_result.content,
            confidence=search_result.confidence,
            modality_contributions=search_result.modality_contributions,
            execution_time=execution_time,
            correlation_id=correlation_id,
            performance_met=performance_met,
            metadata={
                "search_types_used": request.search_types,
                "domain": request.domain,
                "azure_services_used": ["cognitive_search", "cosmos_db", "azure_ml"],
                "tri_modal_unity": True,  # Our unique value proposition
                "competitive_advantage": "simultaneous_vector_graph_gnn"
            }
        )
        
        logger.info(
            "PydanticAI tri-modal search completed",
            extra={
                'correlation_id': correlation_id,
                'execution_time': execution_time,
                'confidence': response.confidence,
                'performance_met': performance_met,
                'modalities_used': list(response.modality_contributions.keys())
            }
        )
        
        return response
        
    except Exception as e:
        execution_time = time.time() - start_time
        
        logger.error(
            "PydanticAI tri-modal search failed",
            extra={
                'correlation_id': correlation_id,
                'error': str(e),
                'execution_time': execution_time
            }
        )
        
        # Re-raise the exception - no fallbacks allowed per coding rules
        raise RuntimeError(f"Tri-modal search failed: {str(e)}") from e


class VectorSearchRequest(BaseModel):
    """Request model for pure vector search"""
    query: str = Field(..., min_length=1, description="Vector search query")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold")
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum results")


async def execute_vector_search(
    ctx: RunContext[AzureServiceContainer],
    request: VectorSearchRequest
) -> Dict[str, Any]:
    """
    Execute pure vector search using Azure Cognitive Search.
    
    This tool provides access to just the vector modality when
    full tri-modal search is not needed.
    """
    
    correlation_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(f"Vector search initiated: {request.query[:50]}...", 
                extra={'correlation_id': correlation_id})
    
    try:
        # Access Azure Cognitive Search from service container
        search_client = ctx.deps.azure_search
        
        if not search_client:
            raise RuntimeError("Azure Search client not available in service container")
        
        # Execute real vector search using Azure Cognitive Search
        search_results = await search_client.vector_search(
            query=request.query,
            similarity_threshold=request.similarity_threshold,
            max_results=request.max_results
        )
        
        execution_time = time.time() - start_time
        
        return {
            "results": search_results.get("results", []),
            "similarity_scores": search_results.get("similarity_scores", []),
            "execution_time": execution_time,
            "correlation_id": correlation_id,
            "modality": "vector_only"
        }
        
    except Exception as e:
        logger.error(f"Vector search failed: {e}", extra={'correlation_id': correlation_id})
        raise RuntimeError(f"Vector search failed: {str(e)}") from e


class GraphSearchRequest(BaseModel):
    """Request model for pure graph search"""
    query: str = Field(..., min_length=1, description="Graph search query")
    max_depth: int = Field(default=3, ge=1, le=5, description="Maximum traversal depth")
    relationship_types: List[str] = Field(default_factory=list, description="Specific relationship types to follow")


async def execute_graph_search(
    ctx: RunContext[AzureServiceContainer],  
    request: GraphSearchRequest
) -> Dict[str, Any]:
    """
    Execute pure graph search using Azure Cosmos DB Gremlin.
    
    This tool provides access to just the graph modality for
    relationship-focused queries.
    """
    
    correlation_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(f"Graph search initiated: {request.query[:50]}...",
                extra={'correlation_id': correlation_id})
    
    try:
        # Access Azure Cosmos DB from service container
        cosmos_client = ctx.deps.azure_cosmos
        
        if not cosmos_client:
            raise RuntimeError("Azure Cosmos DB client not available in service container")
        
        # Execute real graph search using Azure Cosmos DB Gremlin
        graph_results = await cosmos_client.graph_search(
            query=request.query,
            max_depth=request.max_depth,
            relationship_types=request.relationship_types
        )
        
        execution_time = time.time() - start_time
        
        return {
            "results": graph_results.get("results", []),
            "entities_found": graph_results.get("entities_found", 0),
            "relationships_traversed": graph_results.get("relationships_traversed", 0),
            "max_depth_reached": graph_results.get("max_depth_reached", 0),
            "execution_time": execution_time,
            "correlation_id": correlation_id,
            "modality": "graph_only"
        }
        
    except Exception as e:
        logger.error(f"Graph search failed: {e}", extra={'correlation_id': correlation_id})
        raise RuntimeError(f"Graph search failed: {str(e)}") from e


# Export functions for PydanticAI agent registration
__all__ = [
    "execute_tri_modal_search",
    "execute_vector_search", 
    "execute_graph_search",
    "TriModalSearchRequest",
    "TriModalSearchResponse",
    "VectorSearchRequest",
    "GraphSearchRequest"
]


# Test function for development
async def test_search_tools():
    """Test search tools functionality"""
    print("Testing PydanticAI Search Tools...")
    
    # Create mock context with all required services
    class MockContext:
        class MockDeps:
            tri_modal_orchestrator = TriModalOrchestrator()
            azure_search = None
            azure_cosmos = None
        deps = MockDeps()
    
    # Test tri-modal search
    request = TriModalSearchRequest(
        query="test machine learning algorithms",
        domain="technical"
    )
    
    result = await execute_tri_modal_search(MockContext(), request)
    print(f"✅ Tri-modal search: {result.confidence} confidence, {result.execution_time:.2f}s")
    
    # Test vector search
    vector_request = VectorSearchRequest(query="neural networks")
    vector_result = await execute_vector_search(MockContext(), vector_request)
    print(f"✅ Vector search: {vector_result['execution_time']:.2f}s")
    
    print("All search tools working! 🎯")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_search_tools())