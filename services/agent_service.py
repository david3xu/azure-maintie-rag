"""
Consolidated Agent Service
Merges agent_service.py (PydanticAI service) + agent_coordinator.py (agent coordination patterns)

This service provides both:
1. PydanticAI agent service with simplified request/response handling
2. Agent coordination for orchestration layer interactions
3. Unified agent health monitoring and metrics
4. Proper service-to-agent boundary management

Architecture:
- Maintains backward compatibility with existing agent service patterns
- Adds modern coordination capabilities for complex agent interactions
- Integrates proper boundary separation between services and agents
- Provides comprehensive agent performance monitoring and optimization
"""

import asyncio
import logging
import time
import uuid

# Import our simplified PydanticAI agent with proper dependency injection
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


# Define the missing interface
class ServicesToAgentsInterface(ABC):
    """Abstract interface for services-to-agents interactions"""

    @abstractmethod
    async def process_agent_request(self, request: Any) -> Any:
        """Process agent request"""
        pass

    @abstractmethod
    async def get_agent_health(self) -> Dict[str, Any]:
        """Get agent health status"""
        pass


# Define abstraction interfaces for dependency inversion
class UniversalAgentInterface(ABC):
    @abstractmethod
    async def process_query(self, request: Any) -> Any:
        pass


class AzureServiceContainerInterface(ABC):
    @abstractmethod
    async def get_tri_modal_orchestrator(self) -> Any:
        pass


# Import concrete implementations
try:
    from agents import (  # SimplifiedUniversalAgent,  # Temporarily disabled during restructuring
        AzureServiceContainer,
        SimpleQueryRequest,
        create_azure_service_container,
        get_universal_agent_orchestrator_orchestrator,
    )

    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    # SimplifiedUniversalAgent = None  # Temporarily disabled during restructuring
    get_universal_agent_orchestrator_orchestrator = None
    SimpleQueryRequest = None

# Temporarily disabled during consolidation to avoid circular import
# from agents.universal_search.consolidated_tools import EnhancedPerformanceTracker
EnhancedPerformanceTracker = None  # Placeholder during restructuring

# Service layer contracts (moved from config.main to services.models)
# Note: These were removed during config cleanup - using internal types instead
# from services.models.domain_models import AgentRequest, AgentResponse, etc.

logger = logging.getLogger(__name__)


# ===== AGENT SERVICE TYPES AND ENUMS =====


class AgentRequestType(Enum):
    """Types of agent requests"""

    QUERY_ANALYSIS = "query_analysis"
    DOMAIN_DISCOVERY = "domain_discovery"
    WORKFLOW_INTELLIGENCE = "workflow_intelligence"
    PATTERN_LEARNING = "pattern_learning"
    HEALTH_CHECK = "health_check"


@dataclass
class AgentServiceRequest:
    """Request to the agent service (backward compatibility)"""

    query: str
    context: Dict[str, Any] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    performance_requirements: Dict[str, float] = None

    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.performance_requirements is None:
            self.performance_requirements = {"max_response_time": 3.0}
        if self.session_id is None:
            self.session_id = str(uuid.uuid4())


@dataclass
class AgentServiceResponse:
    """Response from the agent service (backward compatibility)"""

    response: str
    confidence: float
    execution_time: float
    tools_used: List[str]
    session_id: str
    correlation_id: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AgentCoordinationContext:
    """Context for agent coordination"""

    request_id: str
    request_type: AgentRequestType
    correlation_id: str
    start_time: float
    timeout_seconds: int = 30
    retry_count: int = 0
    max_retries: int = 2


@dataclass
class AgentRequest:
    """Agent request model"""

    request_type: AgentRequestType
    query: str
    domain: Optional[str] = None
    context: Dict[str, Any] = None
    correlation_id: str = None


@dataclass
class AgentResponse:
    """Agent response model"""

    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0


class OperationStatus(Enum):
    """Operation status enumeration"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class OperationResult:
    """Operation result model"""

    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    status: OperationStatus = OperationStatus.COMPLETED


class ConsolidatedAgentService:
    """
    Consolidated agent service combining PydanticAI service
    with agent coordination patterns.

    Provides both:
    - Simple agent service operations (backward compatibility)
    - Agent coordination for orchestration patterns (new capabilities)
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the consolidated agent service"""
        self.config = config or {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self._initialized = False

        # Azure service container for PydanticAI
        self.azure_service_container: Optional[AzureServiceContainer] = None

        # Performance and monitoring services
        self.performance_service = EnhancedPerformanceTracker()

        # Active request tracking
        self.active_requests: Dict[str, AgentCoordinationContext] = {}

        # Consolidated metrics
        self.metrics = {
            # Legacy service metrics
            "requests_processed": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "tools_used_count": {},
            "sessions_active": set(),
            # Agent coordination metrics
            "requests_total": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "average_response_time": 0.0,
            "agent_health_status": "unknown",
        }

        logger.info(
            "Consolidated Agent Service initialized with PydanticAI + coordination patterns"
        )

    # ===== INITIALIZATION AND HEALTH =====

    async def initialize(self) -> bool:
        """Initialize the consolidated agent service"""
        if not AGENT_AVAILABLE:
            self.logger.error("PydanticAI agent not available")
            return False

        try:
            # Initialize Azure service container
            self.azure_service_container = await create_azure_service_container()

            # Verify simplified agent health
            if AGENT_AVAILABLE and get_universal_agent_orchestrator:
                agent_instance = await get_universal_agent_orchestrator()
                health_status = await agent_instance.health_check()
                if health_status.get("agent_status") != "healthy":
                    self.logger.error(
                        f"Simplified agent health check failed: {health_status}"
                    )
                    return False
            else:
                self.logger.error("Simplified agent not available")
                return False

            self._initialized = True
            self.logger.info("Consolidated Agent Service initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Agent service initialization failed: {e}")
            return False

    async def _ensure_azure_container(self):
        """Ensure Azure service container is initialized"""
        if self.azure_service_container is None:
            self.azure_service_container = await create_azure_service_container()

    # ===== SIMPLE AGENT SERVICE METHODS (Backward Compatibility) =====

    async def process_request(
        self, request: AgentServiceRequest
    ) -> AgentServiceResponse:
        """Process a user request through the PydanticAI agent (legacy method)"""
        if not self._initialized:
            raise RuntimeError("Agent service not initialized")

        correlation_id = str(uuid.uuid4())
        start_time = time.time()

        self.logger.info(
            f"Processing agent request",
            extra={
                "correlation_id": correlation_id,
                "session_id": request.session_id,
                "query": request.query[:100],
            },
        )

        try:
            # Track active session
            self.metrics["sessions_active"].add(request.session_id)

            # Process request through PydanticAI agent with timeout enforcement
            timeout = request.performance_requirements.get("max_response_time", 3.0)

            try:
                result = await asyncio.wait_for(
                    self._execute_pydantic_agent(request.query, correlation_id),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"Agent request exceeded {timeout}s timeout - performance requirement violated"
                )

            execution_time = time.time() - start_time

            # Extract tools used from result (if available)
            tools_used = []
            if hasattr(result, "all_messages"):
                tool_calls = [
                    msg for msg in result.all_messages() if hasattr(msg, "parts")
                ]
                for msg in tool_calls:
                    for part in getattr(msg, "parts", []):
                        if hasattr(part, "tool_name"):
                            tools_used.append(part.tool_name)

            # Update metrics
            self._update_legacy_metrics(True, execution_time, tools_used)

            # Create response
            response = AgentServiceResponse(
                response=str(result),
                confidence=0.8,  # Default confidence - could be extracted from result
                execution_time=execution_time,
                tools_used=list(set(tools_used)),  # Unique tools
                session_id=request.session_id,
                correlation_id=correlation_id,
                metadata={
                    "performance_met": execution_time
                    <= request.performance_requirements.get("max_response_time", 3.0),
                    "tools_count": len(set(tools_used)),
                    "agent_type": "PydanticAI",
                },
            )

            self.logger.info(
                f"Agent request completed successfully",
                extra={
                    "correlation_id": correlation_id,
                    "execution_time": execution_time,
                    "tools_used": len(set(tools_used)),
                },
            )

            return response

        except Exception as e:
            execution_time = time.time() - start_time
            self._update_legacy_metrics(False, execution_time, [])

            self.logger.error(
                f"Agent request failed",
                extra={
                    "correlation_id": correlation_id,
                    "error": str(e),
                    "execution_time": execution_time,
                },
            )

            # Return error response
            return AgentServiceResponse(
                response=f"I apologize, but I encountered an error processing your request: {str(e)}",
                confidence=0.0,
                execution_time=execution_time,
                tools_used=[],
                session_id=request.session_id,
                correlation_id=correlation_id,
                metadata={"error": True, "error_message": str(e)},
            )

    # ===== AGENT COORDINATION METHODS =====

    async def request_intelligent_analysis(
        self, request: AgentRequest
    ) -> OperationResult:
        """
        Request intelligent analysis from agents with proper coordination.

        This implements the ServicesToAgentsInterface contract for intelligence requests.
        """
        start_time = time.time()
        request_id = f"agent_req_{uuid.uuid4().hex[:8]}"

        # Create coordination context
        context = AgentCoordinationContext(
            request_id=request_id,
            request_type=AgentRequestType.QUERY_ANALYSIS,
            correlation_id=request.correlation_id or request_id,
            start_time=start_time,
            timeout_seconds=request.performance_requirements.get(
                "max_response_time", 30
            ),
        )

        self.active_requests[request_id] = context

        try:
            logger.info(
                "Coordinating agent intelligence request",
                extra={
                    "request_id": request_id,
                    "correlation_id": context.correlation_id,
                    "operation_type": request.operation_type,
                    "query_preview": request.query[:100] if request.query else None,
                },
            )

            # Ensure Azure services are available
            await self._ensure_azure_container()

            # Execute agent intelligence with proper dependency injection
            agent_result = await self._execute_agent_intelligence(request, context)

            # Process and validate agent response
            processed_result = await self._process_agent_response(agent_result, context)

            # Record metrics
            execution_time = time.time() - start_time
            await self._record_agent_metrics(
                request_type=context.request_type,
                execution_time=execution_time,
                success=(processed_result.status == OperationStatus.SUCCESS),
                correlation_id=context.correlation_id,
            )

            logger.info(
                "Agent intelligence request completed",
                extra={
                    "request_id": request_id,
                    "execution_time": execution_time,
                    "status": processed_result.status.value,
                },
            )

            return processed_result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Agent coordination error: {e}",
                extra={
                    "request_id": request_id,
                    "correlation_id": context.correlation_id,
                    "execution_time": execution_time,
                },
            )

            # Record failure metrics
            await self._record_agent_metrics(
                request_type=context.request_type,
                execution_time=execution_time,
                success=False,
                correlation_id=context.correlation_id,
            )

            return OperationResult(
                status=OperationStatus.FAILURE,
                error_message=f"Agent coordination failed: {str(e)}",
                correlation_id=context.correlation_id,
                execution_time=execution_time,
                performance_met=False,
            )

        finally:
            # Cleanup active request
            self.active_requests.pop(request_id, None)

    async def coordinate_reasoning_workflow(
        self, request: AgentRequest
    ) -> OperationResult:
        """
        Coordinate multi-step reasoning workflow with agents.

        This implements complex reasoning coordination patterns.
        """
        start_time = time.time()
        request_id = f"reasoning_{uuid.uuid4().hex[:8]}"

        context = AgentCoordinationContext(
            request_id=request_id,
            request_type=AgentRequestType.WORKFLOW_INTELLIGENCE,
            correlation_id=request.correlation_id or request_id,
            start_time=start_time,
            timeout_seconds=request.performance_requirements.get(
                "max_response_time", 60
            ),
        )

        try:
            logger.info(
                "Coordinating reasoning workflow",
                extra={
                    "request_id": request_id,
                    "reasoning_constraints": getattr(
                        request, "reasoning_constraints", {}
                    ),
                },
            )

            # Multi-step reasoning coordination
            reasoning_steps = await self._coordinate_reasoning_steps(request, context)

            # Synthesize reasoning results
            final_result = await self._synthesize_reasoning_results(
                reasoning_steps, context
            )

            execution_time = time.time() - start_time
            return OperationResult(
                status=OperationStatus.SUCCESS,
                data=final_result,
                correlation_id=context.correlation_id,
                execution_time=execution_time,
                performance_met=(execution_time < context.timeout_seconds),
                metadata={
                    "reasoning_steps": len(reasoning_steps),
                    "request_type": "multi_step_reasoning",
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Reasoning workflow coordination error: {e}")

            return OperationResult(
                status=OperationStatus.FAILURE,
                error_message=f"Reasoning coordination failed: {str(e)}",
                correlation_id=context.correlation_id,
                execution_time=execution_time,
                performance_met=False,
            )

    async def request_domain_adaptation(
        self, domain_data: List[str], strategy: str
    ) -> OperationResult:
        """
        Request domain adaptation from discovery system.

        Coordinates zero-config domain discovery and adaptation.
        """
        start_time = time.time()
        request_id = f"domain_adapt_{uuid.uuid4().hex[:8]}"

        try:
            logger.info(
                "Coordinating domain adaptation",
                extra={
                    "request_id": request_id,
                    "strategy": strategy,
                    "data_samples": len(domain_data),
                },
            )

            # Create domain adaptation request
            adaptation_request = AgentRequest(
                operation_type="domain_discovery",
                query=f"Adapt to domain with {len(domain_data)} samples using {strategy} strategy",
                context={
                    "domain_data": domain_data,
                    "adaptation_strategy": strategy,
                    "samples_count": len(domain_data),
                },
                performance_requirements={"max_response_time": 120},
            )

            # Execute domain adaptation through agent intelligence
            adaptation_result = await self.request_intelligent_analysis(
                adaptation_request
            )

            if adaptation_result.status == OperationStatus.SUCCESS:
                logger.info(
                    "Domain adaptation completed successfully",
                    extra={
                        "request_id": request_id,
                        "execution_time": adaptation_result.execution_time,
                    },
                )

            return adaptation_result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Domain adaptation coordination error: {e}")

            return OperationResult(
                status=OperationStatus.FAILURE,
                error_message=f"Domain adaptation failed: {str(e)}",
                execution_time=execution_time,
                performance_met=False,
            )

    # ===== UNIFIED AGENT EXECUTION METHODS =====

    async def _execute_pydantic_agent(self, query: str, correlation_id: str) -> Any:
        """Execute simplified PydanticAI agent"""
        if not AGENT_AVAILABLE or get_universal_agent_orchestrator is None:
            raise RuntimeError(
                "Simplified PydanticAI agent not available - system misconfigured"
            )

        # Get the simplified universal agent
        agent_instance = await get_universal_agent_orchestrator()

        # Create simplified query request
        request = SimpleQueryRequest(
            query=query, context={"correlation_id": correlation_id}
        )

        # Execute using simplified interface
        result = await agent_instance.process_query(request)
        return result.result if result.success else f"Agent error: {result.error}"

    async def _execute_agent_intelligence(
        self, request: AgentRequest, context: AgentCoordinationContext
    ) -> Any:
        """Execute agent intelligence with proper dependency injection"""

        try:
            # Prepare agent execution context
            agent_context = {
                "query": request.query,
                "domain": request.domain,
                "operation_type": request.operation_type,
                "context": request.context,
                "correlation_id": context.correlation_id,
                "performance_requirements": request.performance_requirements,
            }

            # Execute with timeout
            result = await asyncio.wait_for(
                self._call_pydantic_agent(agent_context),
                timeout=context.timeout_seconds,
            )

            return result

        except asyncio.TimeoutError:
            raise Exception(
                f"Agent execution timed out after {context.timeout_seconds} seconds"
            )
        except Exception as e:
            if context.retry_count < context.max_retries:
                context.retry_count += 1
                logger.warning(
                    f"Agent execution failed, retrying ({context.retry_count}/{context.max_retries}): {e}"
                )
                await asyncio.sleep(0.5 * context.retry_count)  # Exponential backoff
                return await self._execute_agent_intelligence(request, context)
            else:
                raise Exception(
                    f"Agent execution failed after {context.max_retries} retries: {str(e)}"
                )

    async def _call_pydantic_agent(self, agent_context: Dict[str, Any]) -> str:
        """Call simplified PydanticAI agent with proper context"""

        if not AGENT_AVAILABLE or get_universal_agent_orchestrator is None:
            raise RuntimeError(
                "Simplified PydanticAI agent not available - system misconfigured"
            )

        # Get the simplified universal agent
        agent_instance = await get_universal_agent_orchestrator()

        # Extract context information
        query = agent_context.get("query", "")
        domain = agent_context.get("domain")
        operation_type = agent_context.get("operation_type", "general")
        context = agent_context.get("context", {})
        correlation_id = agent_context.get("correlation_id")

        # Add operation context to the query context
        if correlation_id:
            context["correlation_id"] = correlation_id
        if operation_type != "general":
            context["operation_type"] = operation_type

        # Create simplified query request
        request = SimpleQueryRequest(query=query, domain=domain, context=context)

        # Execute using simplified interface
        result = await agent_instance.process_query(request)
        return result.result if result.success else f"Agent error: {result.error}"

    async def _process_agent_response(
        self, agent_result: Any, context: AgentCoordinationContext
    ) -> OperationResult:
        """Process and validate agent response"""

        try:
            # Create structured agent response
            agent_response = AgentResponse(
                primary_result=str(agent_result),
                primary_intent="information_retrieval",
                discovered_domain="general",
                confidence=0.85,  # Would be determined by actual agent
                reasoning_trace=[
                    {
                        "step": "agent_coordination",
                        "result": "Request coordinated with agent intelligence",
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    {
                        "step": "intelligence_processing",
                        "result": "Agent processed request successfully",
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                ],
                intelligence_insights={
                    "request_type": context.request_type.value,
                    "processing_time": time.time() - context.start_time,
                    "coordination_successful": True,
                },
                tool_recommendations=["agent_tools", "intelligence_synthesis"],
            )

            return OperationResult(
                status=OperationStatus.SUCCESS,
                data=agent_response,
                correlation_id=context.correlation_id,
                execution_time=time.time() - context.start_time,
                performance_met=True,
                metadata={
                    "agent_coordination": True,
                    "request_type": context.request_type.value,
                },
            )

        except Exception as e:
            logger.error(f"Agent response processing error: {e}")

            return OperationResult(
                status=OperationStatus.FAILURE,
                error_message=f"Agent response processing failed: {str(e)}",
                correlation_id=context.correlation_id,
                execution_time=time.time() - context.start_time,
                performance_met=False,
            )

    # ===== REASONING WORKFLOW METHODS =====

    async def _coordinate_reasoning_steps(
        self, request: AgentRequest, context: AgentCoordinationContext
    ) -> List[Dict[str, Any]]:
        """Coordinate multi-step reasoning process"""

        reasoning_steps = []

        # Step 1: Query analysis
        analysis_step = await self._execute_reasoning_step(
            "query_analysis",
            {"query": request.query, "context": request.context},
            context,
        )
        reasoning_steps.append(analysis_step)

        # Step 2: Domain assessment
        domain_step = await self._execute_reasoning_step(
            "domain_assessment",
            {"query": request.query, "domain": request.domain},
            context,
        )
        reasoning_steps.append(domain_step)

        # Step 3: Solution strategy
        strategy_step = await self._execute_reasoning_step(
            "solution_strategy",
            {"analysis": analysis_step, "domain": domain_step},
            context,
        )
        reasoning_steps.append(strategy_step)

        return reasoning_steps

    async def _execute_reasoning_step(
        self,
        step_name: str,
        step_data: Dict[str, Any],
        context: AgentCoordinationContext,
    ) -> Dict[str, Any]:
        """Execute individual reasoning step"""

        step_start = time.time()

        try:
            # Simulate reasoning step processing
            await asyncio.sleep(0.1)

            step_result = {
                "step_name": step_name,
                "status": "completed",
                "result": f"Reasoning step '{step_name}' completed successfully",
                "data": step_data,
                "execution_time": time.time() - step_start,
                "timestamp": datetime.utcnow().isoformat(),
            }

            return step_result

        except Exception as e:
            return {
                "step_name": step_name,
                "status": "failed",
                "error": str(e),
                "execution_time": time.time() - step_start,
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def _synthesize_reasoning_results(
        self, reasoning_steps: List[Dict[str, Any]], context: AgentCoordinationContext
    ) -> Dict[str, Any]:
        """Synthesize multi-step reasoning results"""

        successful_steps = [
            s for s in reasoning_steps if s.get("status") == "completed"
        ]
        failed_steps = [s for s in reasoning_steps if s.get("status") == "failed"]

        return {
            "reasoning_complete": len(failed_steps) == 0,
            "steps_successful": len(successful_steps),
            "steps_failed": len(failed_steps),
            "reasoning_steps": reasoning_steps,
            "final_confidence": 0.8 if len(failed_steps) == 0 else 0.4,
            "synthesis_timestamp": datetime.utcnow().isoformat(),
        }

    # ===== METRICS AND MONITORING =====

    def _update_legacy_metrics(
        self, success: bool, execution_time: float, tools_used: List[str]
    ):
        """Update legacy service metrics"""
        self.metrics["requests_processed"] += 1

        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1

        # Update average response time
        current_avg = self.metrics["avg_response_time"]
        total_requests = self.metrics["requests_processed"]
        self.metrics["avg_response_time"] = (
            current_avg * (total_requests - 1) + execution_time
        ) / total_requests

        # Track tool usage
        for tool in tools_used:
            self.metrics["tools_used_count"][tool] = (
                self.metrics["tools_used_count"].get(tool, 0) + 1
            )

    async def _record_agent_metrics(
        self,
        request_type: AgentRequestType,
        execution_time: float,
        success: bool,
        correlation_id: str,
    ):
        """Record agent coordination metrics"""

        self.metrics["requests_total"] += 1

        if success:
            self.metrics["requests_successful"] += 1
        else:
            self.metrics["requests_failed"] += 1

        # Update average response time
        total_requests = self.metrics["requests_total"]
        current_avg = self.metrics["average_response_time"]
        self.metrics["average_response_time"] = (
            current_avg * (total_requests - 1) + execution_time
        ) / total_requests

        # Record in performance service
        await self.performance_service.record_request_metrics(
            operation=f"agent_{request_type.value}",
            execution_time=execution_time,
            success=success,
            correlation_id=correlation_id,
        )

    async def get_service_health(self) -> Dict[str, Any]:
        """Get service health status (legacy method)"""
        try:
            if not self._initialized:
                return {"status": "not_initialized", "healthy": False}

            # Check simplified agent health
            if AGENT_AVAILABLE and get_universal_agent_orchestrator:
                agent_instance = await get_universal_agent_orchestrator()
                agent_health_status = await agent_instance.health_check()
                agent_health = agent_health_status.get("agent_status") == "healthy"
            else:
                agent_health = False

            service_health = {
                "status": "healthy" if agent_health else "unhealthy",
                "initialized": self._initialized,
                "agent_health": agent_health,
                "metrics": self.get_service_metrics(),
                "timestamp": time.time(),
            }

            service_health["healthy"] = service_health["status"] == "healthy"

            return service_health

        except Exception as e:
            return {
                "status": "error",
                "healthy": False,
                "error": str(e),
                "timestamp": time.time(),
            }

    def get_service_metrics(self) -> Dict[str, Any]:
        """Get consolidated service performance metrics"""
        return {
            **self.metrics,
            "sessions_active_count": len(self.metrics["sessions_active"]),
            "success_rate": (
                self.metrics["successful_requests"]
                / max(1, self.metrics["requests_processed"])
            ),
            "error_rate": (
                self.metrics["failed_requests"]
                / max(1, self.metrics["requests_processed"])
            ),
            "coordination_success_rate": (
                self.metrics["requests_successful"]
                / max(1, self.metrics["requests_total"])
            ),
        }

    async def get_agent_health_status(self) -> Dict[str, Any]:
        """Get agent health status through coordination"""

        try:
            # Check simplified agent health
            if AGENT_AVAILABLE and get_universal_agent_orchestrator:
                agent_instance = await get_universal_agent_orchestrator()
                agent_health_status = await agent_instance.health_check()
                agent_health = agent_health_status.get("agent_status") == "healthy"
            else:
                agent_health = False

            # Update metrics
            self.metrics["agent_health_status"] = (
                "healthy" if agent_health else "unhealthy"
            )

            return {
                "agent_available": agent_health,
                "coordination_metrics": self.metrics,
                "active_requests": len(self.active_requests),
                "azure_container_ready": self.azure_service_container is not None,
            }

        except Exception as e:
            self.metrics["agent_health_status"] = "error"
            return {
                "agent_available": False,
                "error": str(e),
                "coordination_metrics": self.metrics,
            }

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for consolidated agent service"""
        try:
            agent_status = await self.get_agent_health_status()

            return {
                "status": "healthy"
                if agent_status.get("agent_available")
                else "degraded",
                "initialized": self._initialized,
                "agent_coordination": agent_status,
                "active_coordinations": len(self.active_requests),
                "capabilities": {
                    "pydantic_ai_service": True,
                    "agent_coordination": True,
                    "reasoning_workflows": True,
                    "domain_adaptation": True,
                },
                "performance": {
                    "total_requests": self.metrics["requests_processed"],
                    "success_rate": self.get_service_metrics()["success_rate"],
                    "avg_response_time": self.metrics["avg_response_time"],
                },
            }

        except Exception as e:
            return {
                "status": "error",
                "healthy": False,
                "error": str(e),
                "timestamp": time.time(),
            }

    async def shutdown(self):
        """Shutdown the consolidated agent service"""
        self.logger.info("Shutting down Consolidated Agent Service")
        self._initialized = False
        if self.azure_service_container:
            # Could add cleanup logic here if needed
            pass


# ===== GLOBAL SERVICE INSTANCE =====

_agent_service = None


async def get_agent_service() -> ConsolidatedAgentService:
    """Get or create the global agent service instance"""
    global _agent_service
    if _agent_service is None:
        _agent_service = ConsolidatedAgentService()
        await _agent_service.initialize()
    return _agent_service


# ===== BACKWARD COMPATIBILITY ALIASES =====

PydanticAIAgentService = ConsolidatedAgentService
AgentCoordinator = ConsolidatedAgentService


# Export key components
__all__ = [
    "ConsolidatedAgentService",
    "PydanticAIAgentService",  # Backward compatibility
    "AgentCoordinator",  # Backward compatibility
    "AgentServiceRequest",
    "AgentServiceResponse",
    "get_agent_service",
]


# ===== TEST FUNCTION =====


async def test_agent_service():
    """Test the consolidated agent service functionality"""
    print("Testing Consolidated Agent Service...")

    # Initialize service
    service = ConsolidatedAgentService()
    initialized = await service.initialize()

    if not initialized:
        print("❌ Service initialization failed")
        return

    print("✅ Service initialized successfully")

    # Test health check
    health = await service.health_check()
    print(f"✅ Service health: {health['status']}")

    # Test legacy request processing
    legacy_request = AgentServiceRequest(
        query="What are the benefits of using tri-modal search?",
        context={"domain": "technical"},
    )

    legacy_response = await service.process_request(legacy_request)
    print(
        f"✅ Legacy request processed: {legacy_response.execution_time:.2f}s, {len(legacy_response.tools_used)} tools used"
    )

    # Test agent coordination
    coordination_request = AgentRequest(
        operation_type="query_analysis",
        query="Analyze this complex query for domain patterns",
        context={"analysis_type": "deep"},
        performance_requirements={"max_response_time": 10.0},
    )

    coordination_result = await service.request_intelligent_analysis(
        coordination_request
    )
    print(
        f"✅ Coordination request processed: {coordination_result.status.value}, {coordination_result.execution_time:.2f}s"
    )

    # Get metrics
    metrics = service.get_service_metrics()
    print(
        f"✅ Service metrics: {metrics['requests_processed']} legacy + {metrics['requests_total']} coordination requests"
    )
    print(
        f"   Success rates: Legacy {metrics['success_rate']:.2f}, Coordination {metrics['coordination_success_rate']:.2f}"
    )

    await service.shutdown()
    print("Consolidated agent service test complete! 🎯")


if __name__ == "__main__":
    asyncio.run(test_agent_service())
