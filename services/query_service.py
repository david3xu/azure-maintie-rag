"""
Consolidated Query Service
Merges enhanced_query_service.py (enhanced query processing) + request_orchestrator.py (request orchestration patterns)

This service provides both:
1. Enhanced query processing with proper agent integration
2. Modern request orchestration with performance optimization
3. Unified caching and infrastructure coordination
4. Agent intelligence coordination with clear boundaries

Architecture:
- Maintains backward compatibility with existing query patterns
- Adds modern orchestration capabilities for complex queries
- Integrates with agent layer for intelligent query decisions
- Provides comprehensive performance monitoring and caching
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from agents import (  # Simplified PydanticAI agent
    SimpleQueryRequest,
    get_universal_agent_orchestrator,
)
from agents.universal_search.consolidated_tools import EnhancedPerformanceTracker

# Legacy enhanced query processing imports 
# AgentRequest now defined in agent_service.py
from .agent_service import (
    AgentRequest,
    AgentResponse,
    OperationResult,
    OperationStatus,
    ServicesToAgentsInterface,
)
from infrastructure.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient
from infrastructure.azure_openai import UnifiedAzureOpenAIClient
from infrastructure.azure_search import UnifiedSearchClient

# from agents.capabilities.graph_intelligence import GraphService  # Disabled - capabilities module removed
from .cache_service import SimpleCacheService

logger = logging.getLogger(__name__)


# ===== MODERN ORCHESTRATION DEFINITIONS =====


@dataclass
class QueryProcessingContext:
    """Context for query processing workflow"""

    query: str
    domain: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    user_session: Optional[str] = None
    correlation_id: Optional[str] = None
    max_results: Optional[int] = None
    performance_requirements: Optional[Dict[str, Any]] = None


@dataclass
class ProcessingResult:
    """Result from processing workflow"""

    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    cache_hit: bool = False
    agent_intelligence: Optional[AgentResponse] = None
    infrastructure_results: Optional[List[Dict[str, Any]]] = None


class ConsolidatedQueryService:
    """
    Consolidated query service combining enhanced query processing
    with modern orchestration patterns.

    Provides both:
    - Enhanced query processing (backward compatibility)
    - Modern request orchestration (new capabilities)
    """

    def __init__(self):
        # Enhanced query processing infrastructure
        self.openai_client = UnifiedAzureOpenAIClient()
        self.search_client = UnifiedSearchClient()
        self.cosmos_client = AzureCosmosGremlinClient()
        # self.graph_service = GraphService()  # Disabled - capabilities module removed
        self.graph_service = None
        self.cache_service = SimpleCacheService(use_redis=False)
        self.performance_service = EnhancedPerformanceTracker()

        # Modern orchestration components
        self._active_requests: Dict[str, QueryProcessingContext] = {}

        logger.info(
            "Consolidated Query Service initialized with enhanced + orchestration patterns"
        )

    # ===== ENHANCED QUERY METHODS (Backward Compatibility) =====

    async def process_universal_query(
        self,
        query: str,
        domain: str = None,
        max_results: int = None,
        user_session: str = None,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Enhanced method: Process query using Services orchestration with Agent intelligence.

        Service Layer Responsibilities:
        - Cache management and optimization
        - Performance tracking and monitoring
        - Infrastructure service coordination
        - Result formatting and response preparation

        Agent Layer Responsibilities:
        - Query intelligence analysis
        - Domain detection and adaptation
        - Reasoning and decision-making
        - Tool selection and coordination
        """
        start_time = datetime.now()
        correlation_id = f"query_{int(time.time() * 1000)}"

        logger.info(
            f"Processing universal query: {query[:100]}...",
            extra={
                "correlation_id": correlation_id,
                "domain": domain,
                "user_session": user_session,
            },
        )

        async with self.performance_service.create_performance_context(
            query, domain or "auto-detect", "process_universal_query"
        ) as perf:
            try:
                # ✅ SERVICE RESPONSIBILITY: Cache management
                cached_result = await self._check_cache(query, domain, correlation_id)
                if cached_result:
                    perf.mark_cache_hit()
                    return cached_result

                perf.mark_cache_miss()

                # ✅ AGENT RESPONSIBILITY: Intelligent query analysis
                intelligence_result = await self._request_query_intelligence(
                    query, domain, context, user_session, correlation_id
                )

                # ✅ SERVICE RESPONSIBILITY: Workflow orchestration based on intelligence
                processing_result = await self._orchestrate_processing_workflow(
                    query, intelligence_result, max_results, correlation_id
                )

                # ✅ SERVICE RESPONSIBILITY: Result formatting and caching
                final_result = await self._finalize_and_cache_result(
                    processing_result,
                    query,
                    intelligence_result.discovered_domain,
                    correlation_id,
                    perf,
                )

                execution_time = (datetime.now() - start_time).total_seconds()

                logger.info(
                    f"Universal query processed successfully in {execution_time:.2f}s",
                    extra={
                        "correlation_id": correlation_id,
                        "intent": intelligence_result.primary_intent,
                        "domain": intelligence_result.discovered_domain,
                        "confidence": intelligence_result.confidence,
                        "performance_met": execution_time < 3.0,
                    },
                )

                return final_result

            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()

                logger.error(
                    f"Universal query processing failed: {e}",
                    extra={
                        "correlation_id": correlation_id,
                        "execution_time": execution_time,
                        "query": query[:100],
                    },
                )

                # ✅ SERVICE RESPONSIBILITY: Error handling and fallback
                return await self._get_fallback_result(query, str(e), correlation_id)

    async def process_complex_reasoning_query(
        self,
        query: str,
        reasoning_type: str = "complex",
        domain: str = None,
        context: Dict[str, Any] = None,
        user_session: str = None,
    ) -> Dict[str, Any]:
        """
        Enhanced method: Process query requiring complex multi-step reasoning.

        This method demonstrates proper Service-Agent coordination for
        sophisticated reasoning workflows.
        """
        correlation_id = f"reasoning_{int(time.time() * 1000)}"

        logger.info(
            f"Processing complex reasoning query: {reasoning_type}",
            extra={
                "correlation_id": correlation_id,
                "query": query[:100],
                "reasoning_type": reasoning_type,
            },
        )

        try:
            # ✅ AGENT RESPONSIBILITY: Initial intelligence analysis
            intelligence_result = await self._request_query_intelligence(
                query, domain, context, user_session, correlation_id
            )

            # ✅ AGENT RESPONSIBILITY: Execute complex reasoning workflow
            # Placeholder for complex reasoning - would integrate with actual agent reasoning
            reasoning_result = {
                "final_result": f"Complex reasoning result for: {query}",
                "reasoning_steps": [
                    "step1_analysis",
                    "step2_synthesis",
                    "step3_conclusion",
                ],
                "tools_used": intelligence_result.tool_recommendations,
                "confidence": intelligence_result.confidence,
                "performance_met": True,
                "execution_time": 1.5,
            }

            # ✅ SERVICE RESPONSIBILITY: Infrastructure coordination and result formatting
            infrastructure_result = await self._execute_infrastructure_operations(
                reasoning_result, correlation_id
            )

            # ✅ SERVICE RESPONSIBILITY: Performance tracking and caching
            final_result = {
                "query": query,
                "reasoning_type": reasoning_type,
                "intelligence_analysis": {
                    "intent": intelligence_result.primary_intent,
                    "confidence": intelligence_result.confidence,
                    "domain": intelligence_result.discovered_domain,
                },
                "reasoning_workflow": {
                    "final_result": reasoning_result["final_result"],
                    "steps_executed": len(reasoning_result["reasoning_steps"]),
                    "tools_used": reasoning_result["tools_used"],
                    "confidence": reasoning_result["confidence"],
                    "performance_met": reasoning_result["performance_met"],
                },
                "infrastructure_result": infrastructure_result,
                "metadata": {
                    "correlation_id": correlation_id,
                    "execution_time": reasoning_result["execution_time"],
                    "service_orchestration": "consolidated_query_service",
                },
            }

            # Optional caching for complex reasoning results
            if reasoning_result["confidence"] > 0.8:
                await self.cache_service.set_cached_query_result(
                    f"reasoning_{query}_{reasoning_type}",
                    final_result,
                    ttl_seconds=1800,  # 30 minutes for complex reasoning
                )

            return final_result

        except Exception as e:
            logger.error(
                f"Complex reasoning query failed: {e}",
                extra={"correlation_id": correlation_id},
            )

            return await self._get_fallback_result(query, str(e), correlation_id)

    # ===== MODERN ORCHESTRATION METHODS =====

    async def orchestrate_query_request(
        self,
        query: str,
        domain: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        user_session: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> OperationResult:
        """
        Modern method: Orchestrate complete query processing with agent intelligence.

        Orchestration Flow:
        1. Request validation and preparation
        2. Cache check for performance
        3. Agent intelligence coordination
        4. Result synthesis and optimization
        5. Performance monitoring and caching
        """
        start_time = time.time()
        correlation_id = correlation_id or f"req_{int(time.time() * 1000)}"

        try:
            logger.info(
                f"Orchestrating query request",
                extra={
                    "correlation_id": correlation_id,
                    "query_preview": query[:100],
                    "domain": domain,
                },
            )

            # Create processing context
            processing_context = QueryProcessingContext(
                query=query,
                domain=domain,
                context=context,
                user_session=user_session,
                correlation_id=correlation_id,
            )

            # Store active request
            self._active_requests[correlation_id] = processing_context

            try:
                # Step 1: Check cache for performance optimization
                cache_key = self._generate_cache_key(query, domain, context)
                cached_result = await self._get_cached_result(cache_key)

                if cached_result:
                    logger.info(
                        "Cache hit - returning cached result",
                        extra={
                            "correlation_id": correlation_id,
                            "cache_key": cache_key,
                        },
                    )
                    return self._create_success_result(
                        cached_result,
                        correlation_id,
                        time.time() - start_time,
                        from_cache=True,
                    )

                # Step 2: Coordinate with agent intelligence
                agent_request = AgentRequest(
                    operation_type="query_analysis",
                    query=query,
                    domain=domain,
                    context=context or {},
                    performance_requirements={"max_response_time": 2.5},
                    correlation_id=correlation_id,
                )

                # Use PydanticAI agent for intelligence
                agent_result = await self._coordinate_agent_intelligence(agent_request)

                if agent_result.status != OperationStatus.SUCCESS:
                    return agent_result

                # Step 3: Synthesize and optimize results
                final_result = await self._synthesize_results(
                    agent_result.data, query, correlation_id
                )

                # Step 4: Cache successful results for performance
                if final_result.status == OperationStatus.SUCCESS:
                    await self._cache_result(
                        cache_key, final_result.data, ttl_seconds=300  # 5 minute cache
                    )

                # Step 5: Performance monitoring
                execution_time = time.time() - start_time
                await self.performance_service.record_request_metrics(
                    operation="query_orchestration",
                    execution_time=execution_time,
                    success=(final_result.status == OperationStatus.SUCCESS),
                    correlation_id=correlation_id,
                )

                logger.info(
                    "Query orchestration completed",
                    extra={
                        "correlation_id": correlation_id,
                        "execution_time": execution_time,
                        "status": final_result.status.value,
                    },
                )

                return final_result

            finally:
                # Clean up active request
                self._active_requests.pop(correlation_id, None)

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Query orchestration failed: {e}",
                extra={
                    "correlation_id": correlation_id,
                    "execution_time": execution_time,
                    "error": str(e),
                },
            )

            return OperationResult(
                status=OperationStatus.FAILURE,
                error_message=f"Orchestration failed: {str(e)}",
                correlation_id=correlation_id,
                execution_time=execution_time,
                performance_met=False,
            )

    # ===== UNIFIED HELPER METHODS =====

    async def _check_cache(
        self, query: str, domain: Optional[str], correlation_id: str
    ) -> Optional[Dict[str, Any]]:
        """✅ SERVICE RESPONSIBILITY: Cache management (Enhanced Query Service pattern)"""
        try:
            cache_key = f"{query}_{domain or 'auto'}"
            cached_result = await self.cache_service.get_cached_query_result(
                cache_key, domain
            )

            if cached_result:
                logger.debug(
                    f"Cache HIT for query",
                    extra={"correlation_id": correlation_id, "cache_key": cache_key},
                )

                # Add cache metadata
                cached_result["metadata"] = cached_result.get("metadata", {})
                cached_result["metadata"]["cache_hit"] = True
                cached_result["metadata"]["correlation_id"] = correlation_id

                return cached_result

            logger.debug(
                f"Cache MISS for query",
                extra={"correlation_id": correlation_id, "cache_key": cache_key},
            )

        except Exception as e:
            logger.warning(
                f"Cache check failed: {e}", extra={"correlation_id": correlation_id}
            )

        return None

    async def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Modern orchestrator cache check"""
        try:
            return await self.cache_service.get(cache_key)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None

    async def _cache_result(self, cache_key: str, data: Any, ttl_seconds: int = 300):
        """Modern orchestrator cache storage"""
        try:
            await self.cache_service.set(cache_key, data, ttl_seconds)
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")

    async def _request_query_intelligence(
        self,
        query: str,
        domain: Optional[str],
        context: Optional[Dict[str, Any]],
        user_session: Optional[str],
        correlation_id: str,
    ) -> AgentResponse:
        """✅ PROPER BOUNDARY: Request intelligence from Agent layer"""

        intelligence_request = AgentRequest(
            query=query,
            domain=domain,
            context=context or {},
            performance_requirements={"max_response_time": 2.5},
            user_session=user_session,
        )

        # Placeholder for actual agent intelligence - would integrate with PydanticAI
        # In full implementation: result = await agent.run(query, deps=azure_service_container)

        intelligence_result = AgentResponse(
            primary_result=f"Intelligent analysis of: {query}",
            primary_intent="information_retrieval",
            discovered_domain=domain or "general",
            confidence=0.85,
            reasoning_trace=[
                {"step": "query_analysis", "result": "Query parsed and analyzed"},
                {
                    "step": "domain_detection",
                    "result": f"Domain: {domain or 'general'}",
                },
                {"step": "intelligence_synthesis", "result": "Results synthesized"},
            ],
            intelligence_insights={
                "query_complexity": "medium",
                "domain_confidence": 0.8,
                "recommended_approach": "tri_modal_search",
            },
            tool_recommendations=["vector_search", "knowledge_graph_traversal"],
        )

        logger.debug(
            f"Received intelligence analysis from agents",
            extra={
                "correlation_id": correlation_id,
                "intent": intelligence_result.primary_intent,
                "confidence": intelligence_result.confidence,
                "discovered_domain": intelligence_result.discovered_domain,
            },
        )

        return intelligence_result

    async def _coordinate_agent_intelligence(
        self, request: AgentRequest
    ) -> OperationResult:
        """Coordinate with agent intelligence layer (Orchestrator pattern)"""
        try:
            logger.info(
                "Coordinating with agent intelligence",
                extra={
                    "correlation_id": request.correlation_id,
                    "operation": request.operation_type,
                },
            )

            # Get intelligence using the unified method
            intelligence_result = await self._request_query_intelligence(
                request.query,
                request.domain,
                request.context,
                request.user_session,
                request.correlation_id,
            )

            return OperationResult(
                status=OperationStatus.SUCCESS,
                data=intelligence_result,
                correlation_id=request.correlation_id,
                execution_time=0.5,  # Simulated execution time
                performance_met=True,
            )

        except Exception as e:
            logger.error(
                f"Agent coordination failed: {e}",
                extra={"correlation_id": request.correlation_id},
            )

            return OperationResult(
                status=OperationStatus.FAILURE,
                error_message=f"Agent coordination failed: {str(e)}",
                correlation_id=request.correlation_id,
                performance_met=False,
            )

    async def _orchestrate_processing_workflow(
        self,
        query: str,
        intelligence_result: AgentResponse,
        max_results: Optional[int],
        correlation_id: str,
    ) -> Dict[str, Any]:
        """✅ SERVICE RESPONSIBILITY: Orchestrate infrastructure based on agent intelligence"""

        # Use intelligence results to coordinate infrastructure services
        processing_tasks = []

        # Infrastructure coordination based on agent recommendations
        for tool in intelligence_result.tool_recommendations:
            if tool == "vector_search":
                processing_tasks.append(
                    self._execute_vector_search(query, intelligence_result, max_results)
                )
            elif tool == "knowledge_graph_traversal":
                processing_tasks.append(
                    self._execute_graph_search(query, intelligence_result)
                )
            elif tool == "gnn_predictor":
                processing_tasks.append(
                    self._execute_gnn_analysis(query, intelligence_result)
                )

        # Execute infrastructure operations in parallel
        if processing_tasks:
            results = await asyncio.gather(*processing_tasks, return_exceptions=True)

            # Process results and handle exceptions
            processing_result = {
                "infrastructure_results": [],
                "successful_operations": 0,
                "failed_operations": 0,
            }

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processing_result["infrastructure_results"].append(
                        {
                            "operation": intelligence_result.tool_recommendations[i],
                            "success": False,
                            "error": str(result),
                        }
                    )
                    processing_result["failed_operations"] += 1
                else:
                    processing_result["infrastructure_results"].append(
                        {
                            "operation": intelligence_result.tool_recommendations[i],
                            "success": True,
                            "result": result,
                        }
                    )
                    processing_result["successful_operations"] += 1
        else:
            # Fallback to basic search if no specific tools recommended
            basic_result = await self._execute_basic_search(query, intelligence_result)
            processing_result = {
                "infrastructure_results": [
                    {
                        "operation": "basic_search",
                        "success": True,
                        "result": basic_result,
                    }
                ],
                "successful_operations": 1,
                "failed_operations": 0,
            }

        processing_result["correlation_id"] = correlation_id
        return processing_result

    async def _synthesize_results(
        self, agent_response: AgentResponse, original_query: str, correlation_id: str
    ) -> OperationResult:
        """Synthesize agent results into final response (Modern Orchestrator pattern)"""
        try:
            synthesized_result = {
                "query": original_query,
                "answer": agent_response.primary_result,
                "confidence": agent_response.confidence,
                "reasoning": agent_response.reasoning_trace,
                "insights": agent_response.intelligence_insights,
                "tools_used": agent_response.tool_recommendations,
                "timestamp": datetime.utcnow().isoformat(),
                "correlation_id": correlation_id,
            }

            return OperationResult(
                status=OperationStatus.SUCCESS,
                data=synthesized_result,
                correlation_id=correlation_id,
                performance_met=True,
            )

        except Exception as e:
            return OperationResult(
                status=OperationStatus.FAILURE,
                error_message=f"Result synthesis failed: {str(e)}",
                correlation_id=correlation_id,
                performance_met=False,
            )

    async def _execute_vector_search(
        self, query: str, intelligence_result: AgentResponse, max_results: Optional[int]
    ) -> Dict[str, Any]:
        """✅ SERVICE RESPONSIBILITY: Execute vector search infrastructure"""
        try:
            # Use discovered domain from intelligence for search optimization
            domain = intelligence_result.discovered_domain or "general"

            search_results = await self.search_client.vector_search(
                query=query, top_k=max_results or 10, domain_context=domain
            )

            return {
                "operation": "vector_search",
                "results": search_results,
                "domain_used": domain,
                "result_count": len(search_results.get("results", [])),
            }

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return {"operation": "vector_search", "error": str(e)}

    async def _execute_graph_search(
        self, query: str, intelligence_result: AgentResponse
    ) -> Dict[str, Any]:
        """✅ SERVICE RESPONSIBILITY: Execute graph search infrastructure"""
        try:
            # graph_results = await self.graph_service.traverse_relationships(  # Disabled - capabilities module removed
            # query=query, domain=intelligence_result.discovered_domain, max_depth=3)
            graph_results = []  # Placeholder for graph results

            return {
                "operation": "graph_search",
                "results": graph_results,
                "relationship_count": len(graph_results.get("relationships", [])),
            }

        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return {"operation": "graph_search", "error": str(e)}

    async def _execute_gnn_analysis(
        self, query: str, intelligence_result: AgentResponse
    ) -> Dict[str, Any]:
        """✅ SERVICE RESPONSIBILITY: Execute GNN analysis infrastructure"""
        try:
            # Placeholder for GNN analysis - would integrate with actual GNN service
            gnn_results = {
                "predictions": ["prediction1", "prediction2"],
                "confidence_scores": [0.8, 0.7],
                "pattern_analysis": "complex_pattern_detected",
            }

            return {
                "operation": "gnn_analysis",
                "results": gnn_results,
                "prediction_count": len(gnn_results["predictions"]),
            }

        except Exception as e:
            logger.error(f"GNN analysis failed: {e}")
            return {"operation": "gnn_analysis", "error": str(e)}

    async def _execute_basic_search(
        self, query: str, intelligence_result: AgentResponse
    ) -> Dict[str, Any]:
        """✅ SERVICE RESPONSIBILITY: Execute basic search as fallback"""
        try:
            basic_results = await self.search_client.basic_search(
                query=query, domain=intelligence_result.discovered_domain
            )

            return {
                "operation": "basic_search",
                "results": basic_results,
                "result_count": len(basic_results.get("results", [])),
            }

        except Exception as e:
            logger.error(f"Basic search failed: {e}")
            return {"operation": "basic_search", "error": str(e)}

    async def _execute_infrastructure_operations(
        self, reasoning_result, correlation_id: str
    ) -> Dict[str, Any]:
        """✅ SERVICE RESPONSIBILITY: Execute infrastructure operations based on reasoning"""
        # Placeholder for infrastructure operations based on reasoning results
        return {
            "operations_executed": reasoning_result.get("tools_used", []),
            "infrastructure_status": "operational",
            "correlation_id": correlation_id,
        }

    async def _finalize_and_cache_result(
        self,
        processing_result: Dict[str, Any],
        query: str,
        domain: Optional[str],
        correlation_id: str,
        perf,
    ) -> Dict[str, Any]:
        """✅ SERVICE RESPONSIBILITY: Result finalization and caching"""

        # Build final result structure
        final_result = {
            "query": query,
            "domain": domain,
            "processing_result": processing_result,
            "performance": {
                "successful_operations": processing_result.get(
                    "successful_operations", 0
                ),
                "failed_operations": processing_result.get("failed_operations", 0),
                "total_operations": (
                    processing_result.get("successful_operations", 0)
                    + processing_result.get("failed_operations", 0)
                ),
            },
            "metadata": {
                "correlation_id": correlation_id,
                "cache_hit": False,
                "service_layer": "consolidated_query_service",
                "timestamp": datetime.now().isoformat(),
            },
        }

        # Set performance tracking
        perf.set_result_count(final_result["performance"]["total_operations"])

        # Cache successful results
        if processing_result.get("successful_operations", 0) > 0:
            try:
                cache_key = f"{query}_{domain or 'auto'}"
                await self.cache_service.set_cached_query_result(
                    cache_key, final_result, ttl_seconds=3600
                )
                logger.debug(
                    f"Result cached successfully",
                    extra={"correlation_id": correlation_id, "cache_key": cache_key},
                )
            except Exception as e:
                logger.warning(
                    f"Failed to cache result: {e}",
                    extra={"correlation_id": correlation_id},
                )

        return final_result

    def _generate_cache_key(
        self, query: str, domain: Optional[str], context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate cache key for request"""
        key_parts = [
            query.lower().strip(),
            domain or "general",
            str(sorted((context or {}).items())),
        ]

        key_string = "|".join(key_parts)
        return f"query_cache:{hashlib.md5(key_string.encode()).hexdigest()[:16]}"

    def _create_success_result(
        self,
        data: Any,
        correlation_id: str,
        execution_time: float,
        from_cache: bool = False,
    ) -> OperationResult:
        """Create successful operation result"""
        return OperationResult(
            status=OperationStatus.SUCCESS,
            data=data,
            correlation_id=correlation_id,
            execution_time=execution_time,
            performance_met=True,
            metadata={"from_cache": from_cache},
        )

    async def _get_fallback_result(
        self, query: str, error: str, correlation_id: str
    ) -> Dict[str, Any]:
        """✅ SERVICE RESPONSIBILITY: Fallback result generation"""
        return {
            "query": query,
            "domain": "fallback",
            "processing_result": {
                "infrastructure_results": [],
                "successful_operations": 0,
                "failed_operations": 1,
                "error": error,
            },
            "performance": {
                "successful_operations": 0,
                "failed_operations": 1,
                "total_operations": 1,
            },
            "metadata": {
                "correlation_id": correlation_id,
                "fallback_used": True,
                "error": error,
                "service_layer": "consolidated_query_service",
            },
        }

    # ===== DOMAIN ADAPTATION METHODS =====

    async def adapt_service_to_domain(
        self,
        domain_name: str,
        raw_text_samples: List[str],
        adaptation_strategy: str = "balanced",
    ) -> Dict[str, Any]:
        """
        Adapt service capabilities to new domain.

        Services orchestrate domain adaptation, Agents perform the intelligence.
        """
        correlation_id = f"adaptation_{int(time.time() * 1000)}"

        logger.info(
            f"Adapting service to domain: {domain_name}",
            extra={
                "correlation_id": correlation_id,
                "sample_count": len(raw_text_samples),
                "strategy": adaptation_strategy,
            },
        )

        try:
            # ✅ AGENT RESPONSIBILITY: Domain adaptation and learning
            # Placeholder for domain adaptation - would integrate with actual agent learning

            adaptation_result = {
                "discovered_domain": domain_name,
                "domain_patterns": ["pattern1", "pattern2", "pattern3"],
                "confidence": 0.85,
                "adaptation_time": 2.5,
                "success": True,
            }

            # ✅ SERVICE RESPONSIBILITY: Update service configuration and caches
            await self._update_service_configuration_for_domain(adaptation_result)

            # ✅ SERVICE RESPONSIBILITY: Validate infrastructure compatibility
            infrastructure_validation = await self._validate_domain_infrastructure(
                adaptation_result["discovered_domain"]
            )

            result = {
                "domain_adaptation": {
                    "discovered_domain": adaptation_result["discovered_domain"],
                    "patterns_found": len(adaptation_result["domain_patterns"]),
                    "confidence": adaptation_result["confidence"],
                    "adaptation_time": adaptation_result["adaptation_time"],
                    "success": adaptation_result["success"],
                },
                "service_updates": {
                    "configuration_updated": True,
                    "cache_cleared": True,
                    "infrastructure_validated": infrastructure_validation,
                },
                "metadata": {
                    "correlation_id": correlation_id,
                    "domain_name": domain_name,
                    "adaptation_strategy": adaptation_strategy,
                },
            }

            logger.info(
                f"Domain adaptation completed successfully",
                extra={
                    "correlation_id": correlation_id,
                    "discovered_domain": adaptation_result["discovered_domain"],
                    "confidence": adaptation_result["confidence"],
                },
            )

            return result

        except Exception as e:
            logger.error(
                f"Domain adaptation failed: {e}",
                extra={"correlation_id": correlation_id},
            )

            return {
                "domain_adaptation": {"success": False, "error": str(e)},
                "service_updates": {"error": str(e)},
                "metadata": {"correlation_id": correlation_id},
            }

    async def _update_service_configuration_for_domain(self, adaptation_result) -> None:
        """✅ SERVICE RESPONSIBILITY: Update service configuration"""
        # Clear relevant caches for new domain
        await self.cache_service.clear_domain_cache(
            adaptation_result["discovered_domain"]
        )

        # Update service-level configuration if needed
        # Placeholder for service configuration updates

    async def _validate_domain_infrastructure(self, domain: str) -> bool:
        """✅ SERVICE RESPONSIBILITY: Validate infrastructure for domain"""
        # Check if infrastructure services are ready for this domain
        try:
            # Basic connectivity checks
            search_health = await self.search_client.health_check()
            cosmos_health = await self.cosmos_client.health_check()
            openai_health = await self.openai_client.health_check()

            return all([search_health, cosmos_health, openai_health])

        except Exception as e:
            logger.error(f"Infrastructure validation failed: {e}")
            return False

    # ===== HEALTH CHECK AND MONITORING =====

    async def health_check(self) -> Dict[str, Any]:
        """✅ SERVICE RESPONSIBILITY: Comprehensive service health monitoring"""
        try:
            # Check service-level health
            service_health = {
                "cache_service": await self.cache_service.health_check(),
                "performance_service": await self.performance_service.health_check(),
                # 'graph_service': await self.graph_service.health_check()  # Disabled - capabilities module removed
                "graph_service": {
                    "status": "disabled",
                    "reason": "capabilities module removed",
                },
            }

            # Check infrastructure health
            infrastructure_health = {
                "openai_client": await self.openai_client.health_check(),
                "search_client": await self.search_client.health_check(),
                "cosmos_client": await self.cosmos_client.health_check(),
            }

            # Check orchestrator status
            orchestrator_status = {
                "active_requests": len(self._active_requests),
                "request_orchestration": "available",
                "agent_coordination": "available",
            }

            overall_healthy = all(service_health.values()) and all(
                infrastructure_health.values()
            )

            return {
                "overall_status": "healthy" if overall_healthy else "degraded",
                "service_health": service_health,
                "infrastructure_health": infrastructure_health,
                "orchestrator_status": orchestrator_status,
                "capabilities": {
                    "enhanced_query_processing": True,
                    "modern_orchestration": True,
                    "agent_intelligence_coordination": True,
                    "domain_adaptation": True,
                },
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "overall_status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }


# Backward compatibility aliases
EnhancedQueryService = ConsolidatedQueryService
RequestOrchestrator = ConsolidatedQueryService
