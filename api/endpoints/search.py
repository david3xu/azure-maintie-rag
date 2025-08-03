"""
Search API Endpoints - Universal RAG Search

Implements production-ready search endpoints following core principles:
- Data-Driven Everything: Real search results from domain intelligence
- Universal Truth: Authentic tri-modal search results
- Zero Configuration: Automatic domain detection
- Production Reality: Real-time search with proper error handling
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

# Domain intelligence and search components
from agents.domain_intelligence.domain_analyzer import DomainAnalyzer
from agents.universal_search.gnn_search import GNNSearchEngine
from agents.universal_search.graph_search import GraphSearchEngine
from agents.universal_search.vector_search import VectorSearchEngine

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["search"])


# Pydantic models
class SearchRequest(BaseModel):
    """Search request model"""

    query: str = Field(..., description="The search query")
    search_type: str = Field(
        default="tri_modal",
        description="Type of search: vector, graph, gnn, or tri_modal",
    )
    max_results: int = Field(default=10, description="Maximum number of results")
    domain: Optional[str] = Field(
        default=None, description="Domain context (auto-detected if not provided)"
    )


class SearchResult(BaseModel):
    """Individual search result"""

    content: str
    confidence: float
    source: str
    metadata: Dict[str, Any]
    execution_time: float


class SearchResponse(BaseModel):
    """Search response model"""

    success: bool
    query: str
    search_type: str
    detected_domain: Optional[str] = None
    domain_confidence: Optional[float] = None
    results: List[SearchResult]
    total_results: int
    execution_time: float
    timestamp: str
    error: Optional[str] = None


class DomainAnalysisRequest(BaseModel):
    """Domain analysis request model"""

    content: str = Field(..., description="Content to analyze for domain detection")


class DomainAnalysisResponse(BaseModel):
    """Domain analysis response model"""

    success: bool
    detected_domain: str
    confidence: float
    primary_indicators: List[str]
    content_stats: Dict[str, Any]
    execution_time: float
    timestamp: str
    error: Optional[str] = None


# Initialize search engines
vector_search = VectorSearchEngine()
graph_search = GraphSearchEngine()
gnn_search = GNNSearchEngine()
domain_analyzer = DomainAnalyzer()


@router.post("/search", response_model=SearchResponse)
async def search_content(request: SearchRequest) -> Dict[str, Any]:
    """
    Universal search endpoint with tri-modal capabilities

    Performs intelligent search using:
    - Vector search for semantic similarity
    - Graph search for relational context
    - GNN search for pattern prediction
    - Automatic domain detection for query context
    """
    logger.info(f"Processing search request: {request.query[:50]}...")

    try:
        start_time = time.time()

        # Domain detection from query if not provided
        detected_domain = None
        domain_confidence = None

        if not request.domain:
            try:
                # Use query content for lightweight domain detection
                # Create temporary content for analysis
                import tempfile
                from pathlib import Path

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False
                ) as f:
                    f.write(request.query)
                    temp_path = Path(f.name)

                # Analyze query for domain context
                content_analysis = domain_analyzer.analyze_raw_content(temp_path)
                domain_classification = domain_analyzer.classify_content_domain(
                    content_analysis
                )

                detected_domain = domain_classification.domain
                domain_confidence = domain_classification.confidence

                # Clean up temp file
                temp_path.unlink()

                logger.info(
                    f"Auto-detected domain: {detected_domain} (confidence: {domain_confidence:.3f})"
                )

            except Exception as e:
                logger.warning(f"Domain detection failed, using general domain: {e}")
                detected_domain = "general"
                domain_confidence = 0.5
        else:
            detected_domain = request.domain
            domain_confidence = 1.0  # User-provided domain

        # Execute search based on type
        results = []

        if request.search_type == "tri_modal":
            # Execute all three search modalities
            search_tasks = []

            # Vector search
            try:
                vector_result = await vector_search.execute_search(
                    request.query, {"domain": detected_domain}
                )
                results.append(
                    SearchResult(
                        content=vector_result.content,
                        confidence=vector_result.confidence,
                        source=vector_result.source,
                        metadata=vector_result.metadata,
                        execution_time=vector_result.execution_time,
                    )
                )
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
                results.append(
                    SearchResult(
                        content=f"Vector search unavailable: {str(e)}",
                        confidence=0.0,
                        source="vector_fallback",
                        metadata={"error": str(e)},
                        execution_time=0.0,
                    )
                )

            # Graph search
            try:
                graph_result = await graph_search.execute_search(
                    request.query, {"domain": detected_domain}
                )
                results.append(
                    SearchResult(
                        content=graph_result.content,
                        confidence=graph_result.confidence,
                        source=graph_result.source,
                        metadata=graph_result.metadata,
                        execution_time=graph_result.execution_time,
                    )
                )
            except Exception as e:
                logger.warning(f"Graph search failed: {e}")
                results.append(
                    SearchResult(
                        content=f"Graph search unavailable: {str(e)}",
                        confidence=0.0,
                        source="graph_fallback",
                        metadata={"error": str(e)},
                        execution_time=0.0,
                    )
                )

            # GNN search
            try:
                gnn_result = await gnn_search.execute_search(
                    request.query, {"domain": detected_domain}
                )
                results.append(
                    SearchResult(
                        content=gnn_result.content,
                        confidence=gnn_result.confidence,
                        source=gnn_result.source,
                        metadata=gnn_result.metadata,
                        execution_time=gnn_result.execution_time,
                    )
                )
            except Exception as e:
                logger.warning(f"GNN search failed: {e}")
                results.append(
                    SearchResult(
                        content=f"GNN search unavailable: {str(e)}",
                        confidence=0.0,
                        source="gnn_fallback",
                        metadata={"error": str(e)},
                        execution_time=0.0,
                    )
                )

        elif request.search_type == "vector":
            vector_result = await vector_search.execute_search(
                request.query, {"domain": detected_domain}
            )
            results.append(
                SearchResult(
                    content=vector_result.content,
                    confidence=vector_result.confidence,
                    source=vector_result.source,
                    metadata=vector_result.metadata,
                    execution_time=vector_result.execution_time,
                )
            )

        elif request.search_type == "graph":
            graph_result = await graph_search.execute_search(
                request.query, {"domain": detected_domain}
            )
            results.append(
                SearchResult(
                    content=graph_result.content,
                    confidence=graph_result.confidence,
                    source=graph_result.source,
                    metadata=graph_result.metadata,
                    execution_time=graph_result.execution_time,
                )
            )

        elif request.search_type == "gnn":
            gnn_result = await gnn_search.execute_search(
                request.query, {"domain": detected_domain}
            )
            results.append(
                SearchResult(
                    content=gnn_result.content,
                    confidence=gnn_result.confidence,
                    source=gnn_result.source,
                    metadata=gnn_result.metadata,
                    execution_time=gnn_result.execution_time,
                )
            )

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid search_type: {request.search_type}. Must be one of: vector, graph, gnn, tri_modal",
            )

        # Limit results to max_results
        results = results[: request.max_results]

        execution_time = time.time() - start_time

        logger.info(
            f"Search completed: {len(results)} results in {execution_time:.3f}s"
        )

        return {
            "success": True,
            "query": request.query,
            "search_type": request.search_type,
            "detected_domain": detected_domain,
            "domain_confidence": domain_confidence,
            "results": [result.dict() for result in results],
            "total_results": len(results),
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        execution_time = time.time() - start_time

        return {
            "success": False,
            "query": request.query,
            "search_type": request.search_type,
            "detected_domain": None,
            "domain_confidence": None,
            "results": [],
            "total_results": 0,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
        }


@router.post("/analyze/domain", response_model=DomainAnalysisResponse)
async def analyze_domain(request: DomainAnalysisRequest) -> Dict[str, Any]:
    """
    Analyze content for domain detection

    Uses universal domain intelligence to automatically detect domain
    without requiring manual configuration or hardcoded terms.
    """
    logger.info(f"Analyzing content for domain detection: {len(request.content)} chars")

    try:
        start_time = time.time()

        # Create temporary file for content analysis
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(request.content)
            temp_path = Path(f.name)

        try:
            # Perform domain analysis
            content_analysis = domain_analyzer.analyze_raw_content(temp_path)
            domain_classification = domain_analyzer.classify_content_domain(
                content_analysis
            )

            execution_time = time.time() - start_time

            # Clean up temp file
            temp_path.unlink()

            logger.info(
                f"Domain analysis completed: {domain_classification.domain} (confidence: {domain_classification.confidence:.3f})"
            )

            return {
                "success": True,
                "detected_domain": domain_classification.domain,
                "confidence": domain_classification.confidence,
                "primary_indicators": domain_classification.primary_indicators[:10],
                "content_stats": {
                    "word_count": content_analysis.word_count,
                    "unique_words": content_analysis.unique_words,
                    "vocabulary_richness": content_analysis.vocabulary_richness,
                    "complexity_score": content_analysis.complexity_score,
                    "technical_density": content_analysis.technical_density,
                },
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
            }

        finally:
            # Ensure temp file is cleaned up
            if temp_path.exists():
                temp_path.unlink()

    except Exception as e:
        logger.error(f"Domain analysis failed: {e}", exc_info=True)
        execution_time = time.time() - start_time

        return {
            "success": False,
            "detected_domain": "unknown",
            "confidence": 0.0,
            "primary_indicators": [],
            "content_stats": {},
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
        }


@router.get("/search/health")
async def search_health_check() -> Dict[str, Any]:
    """
    Health check for search services
    """
    try:
        # Test all search engines
        test_query = "health check"
        test_context = {"domain": "general"}

        engine_status = {}

        # Test vector search
        try:
            vector_result = await vector_search.execute_search(test_query, test_context)
            engine_status["vector"] = {
                "status": "healthy",
                "response_time": vector_result.execution_time,
            }
        except Exception as e:
            engine_status["vector"] = {"status": "error", "error": str(e)}

        # Test graph search
        try:
            graph_result = await graph_search.execute_search(test_query, test_context)
            engine_status["graph"] = {
                "status": "healthy",
                "response_time": graph_result.execution_time,
            }
        except Exception as e:
            engine_status["graph"] = {"status": "error", "error": str(e)}

        # Test GNN search
        try:
            gnn_result = await gnn_search.execute_search(test_query, test_context)
            engine_status["gnn"] = {
                "status": "healthy",
                "response_time": gnn_result.execution_time,
            }
        except Exception as e:
            engine_status["gnn"] = {"status": "error", "error": str(e)}

        # Test domain analyzer
        try:
            # Quick domain analysis test
            domain_analyzer.classify_content_domain(
                None
            )  # Should handle None gracefully
            engine_status["domain_analyzer"] = {"status": "healthy"}
        except Exception as e:
            engine_status["domain_analyzer"] = {"status": "error", "error": str(e)}

        healthy_engines = sum(
            1 for status in engine_status.values() if status.get("status") == "healthy"
        )
        total_engines = len(engine_status)

        return {
            "status": "healthy" if healthy_engines == total_engines else "degraded",
            "engines": engine_status,
            "healthy_engines": healthy_engines,
            "total_engines": total_engines,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Search health check failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }
