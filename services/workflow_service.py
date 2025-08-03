"""
Consolidated Workflow Service
Merges workflow_service.py (legacy Azure workflow tracking) + workflow_orchestrator.py (modern orchestration patterns)

This service provides both:
1. Legacy Azure workflow tracking and progress monitoring
2. Modern workflow orchestration with step dependency management
3. Performance optimization and error handling
4. Agent integration and coordination

Architecture:
- Maintains backward compatibility with existing Azure workflow patterns
- Adds modern orchestration capabilities for complex workflows
- Integrates with agent layer for intelligent workflow decisions
- Provides comprehensive monitoring and observability
"""

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from agents import SimpleQueryRequest, get_universal_agent_orchestrator
from agents.universal_search.consolidated_tools import EnhancedPerformanceTracker

# Legacy workflow tracking imports
from config.azure_settings import azure_settings
from infrastructure.utilities.azure_cost_tracker import AzureServiceCostTracker
from infrastructure.utilities.workflow_evidence_collector import (
    AzureDataWorkflowEvidenceCollector,
    DataWorkflowEvidence,
)

# Modern orchestration imports
from .agent_service import AgentRequest, AgentResponse, OperationResult, OperationStatus

logger = logging.getLogger(__name__)


# ===== LEGACY WORKFLOW DEFINITIONS (from workflow_service.py) =====


class WorkflowStep(Enum):
    """Legacy workflow step enumeration for Azure service workflows"""

    INITIALIZATION = "initialization"
    DATA_LOADING = "data_loading"
    BLOB_STORAGE = "blob_storage"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    SEARCH_INDEXING = "search_indexing"
    COSMOS_STORAGE = "cosmos_storage"
    VALIDATION = "validation"
    COMPLETION = "completion"


class AzureServiceType(Enum):
    """Azure service types for workflow management"""

    OPENAI = "azure_openai"
    SEARCH = "cognitive_search"
    COSMOS = "cosmos_db"
    STORAGE = "blob_storage"


@dataclass
class ProgressStatus:
    """Legacy progress status tracking for Azure workflows"""

    current_step: WorkflowStep
    step_progress: float
    total_progress: float
    start_time: datetime
    estimated_completion: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)


# ===== MODERN ORCHESTRATION DEFINITIONS (from workflow_orchestrator.py) =====


class WorkflowStatus(Enum):
    """Modern workflow execution status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StepStatus(Enum):
    """Individual step status for modern workflows"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ModernWorkflowStep:
    """Modern workflow step definition with dependencies"""

    id: str
    name: str
    description: str
    handler: Callable
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: int = 300
    retry_count: int = 3
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    execution_time: float = 0.0


@dataclass
class ModernWorkflowExecution:
    """Modern workflow execution context"""

    id: str
    name: str
    description: str
    steps: List[ModernWorkflowStep]
    status: WorkflowStatus = WorkflowStatus.PENDING
    context: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    execution_time: float = 0.0
    correlation_id: Optional[str] = None
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class ConsolidatedWorkflowService:
    """
    Consolidated workflow service combining legacy Azure workflow tracking
    with modern orchestration patterns.

    Provides both:
    - Legacy Azure workflow management (backward compatibility)
    - Modern workflow orchestration (new capabilities)
    """

    def __init__(self):
        # Legacy workflow tracking
        self.progress_trackers: Dict[str, ProgressStatus] = {}
        self.evidence_collector = AzureDataWorkflowEvidenceCollector()
        self.cost_tracker = AzureServiceCostTracker()

        # Modern orchestration
        self.active_workflows: Dict[str, ModernWorkflowExecution] = {}
        self.workflow_templates: Dict[str, Dict[str, Any]] = {}
        self.performance_service = EnhancedPerformanceTracker()

        # Shared state
        self._lock = threading.Lock()

        # Register built-in modern workflow templates
        self._register_builtin_workflows()

        logger.info(
            "Consolidated Workflow Service initialized with legacy + modern patterns"
        )

    # ===== LEGACY WORKFLOW METHODS (Backward Compatibility) =====

    def start_workflow_tracking(
        self, workflow_id: str, workflow_type: str
    ) -> ProgressStatus:
        """Legacy method: Start tracking an Azure workflow"""
        with self._lock:
            progress = ProgressStatus(
                current_step=WorkflowStep.INITIALIZATION,
                step_progress=0.0,
                total_progress=0.0,
                start_time=datetime.utcnow(),
            )
            self.progress_trackers[workflow_id] = progress

            logger.info(
                f"Started legacy workflow tracking: {workflow_id} ({workflow_type})"
            )
            return progress

    def update_workflow_progress(
        self, workflow_id: str, step: WorkflowStep, progress: float
    ) -> None:
        """Legacy method: Update workflow progress"""
        with self._lock:
            if workflow_id in self.progress_trackers:
                tracker = self.progress_trackers[workflow_id]
                tracker.current_step = step
                tracker.step_progress = progress
                tracker.total_progress = self._calculate_total_progress(step, progress)

                logger.debug(
                    f"Updated workflow progress: {workflow_id} - {step.value}: {progress:.1%}"
                )

    def get_workflow_progress(self, workflow_id: str) -> Optional[ProgressStatus]:
        """Legacy method: Get workflow progress"""
        return self.progress_trackers.get(workflow_id)

    def record_workflow_evidence(
        self, workflow_id: str, step: WorkflowStep, data: Dict[str, Any]
    ) -> None:
        """Legacy method: Record workflow evidence"""
        evidence = DataWorkflowEvidence(
            workflow_id=workflow_id,
            step=step.value,
            data=data,
            timestamp=datetime.utcnow(),
        )
        self.evidence_collector.record_evidence(evidence)

    def track_azure_costs(
        self,
        workflow_id: str,
        service_type: AzureServiceType,
        cost_data: Dict[str, Any],
    ) -> None:
        """Legacy method: Track Azure service costs"""
        self.cost_tracker.record_service_usage(
            workflow_id=workflow_id,
            service_type=service_type.value,
            cost_data=cost_data,
        )

    # ===== MODERN ORCHESTRATION METHODS =====

    def _register_builtin_workflows(self):
        """Register built-in modern workflow templates"""

        # Universal RAG Query Workflow
        self.workflow_templates["universal_query"] = {
            "name": "Universal RAG Query Processing",
            "description": "Complete query processing with tri-modal search",
            "steps": [
                {
                    "id": "query_validation",
                    "name": "Query Validation",
                    "description": "Validate and prepare query",
                    "handler": self._validate_query_step,
                    "dependencies": [],
                    "timeout_seconds": 30,
                },
                {
                    "id": "agent_analysis",
                    "name": "Agent Intelligence Analysis",
                    "description": "Coordinate with agent for intelligent analysis",
                    "handler": self._agent_analysis_step,
                    "dependencies": ["query_validation"],
                    "timeout_seconds": 120,
                },
                {
                    "id": "search_execution",
                    "name": "Tri-Modal Search Execution",
                    "description": "Execute vector, graph, and GNN search",
                    "handler": self._search_execution_step,
                    "dependencies": ["agent_analysis"],
                    "timeout_seconds": 180,
                },
                {
                    "id": "result_synthesis",
                    "name": "Result Synthesis",
                    "description": "Synthesize and optimize results",
                    "handler": self._result_synthesis_step,
                    "dependencies": ["search_execution"],
                    "timeout_seconds": 60,
                },
            ],
        }

    async def execute_workflow(
        self,
        workflow_name: str,
        context: Dict[str, Any],
        correlation_id: Optional[str] = None,
    ) -> OperationResult:
        """Modern method: Execute a workflow by template name"""
        start_time = time.time()
        correlation_id = correlation_id or f"wf_{uuid.uuid4().hex[:8]}"

        try:
            # Validate workflow template exists
            if workflow_name not in self.workflow_templates:
                return OperationResult(
                    status=OperationStatus.FAILURE,
                    error_message=f"Workflow template '{workflow_name}' not found",
                    correlation_id=correlation_id,
                    execution_time=time.time() - start_time,
                )

            # Create workflow execution
            execution = self._create_workflow_execution(
                workflow_name, context, correlation_id
            )

            # Store active workflow
            self.active_workflows[execution.id] = execution

            logger.info(
                f"Starting modern workflow execution: {workflow_name}",
                extra={
                    "workflow_id": execution.id,
                    "correlation_id": correlation_id,
                    "steps_count": len(execution.steps),
                },
            )

            # Execute workflow steps
            execution.status = WorkflowStatus.RUNNING
            execution.start_time = time.time()

            await self._execute_workflow_steps(execution)

            # Finalize execution
            execution.end_time = time.time()
            execution.execution_time = execution.end_time - execution.start_time

            # Determine final status
            if execution.status == WorkflowStatus.RUNNING:
                failed_steps = [
                    s for s in execution.steps if s.status == StepStatus.FAILED
                ]
                if failed_steps:
                    execution.status = WorkflowStatus.FAILED
                    execution.errors.extend([s.error for s in failed_steps if s.error])
                else:
                    execution.status = WorkflowStatus.COMPLETED

            # Record performance metrics
            await self.performance_service.record_request_metrics(
                operation=f"workflow_{workflow_name}",
                execution_time=execution.execution_time,
                success=(execution.status == WorkflowStatus.COMPLETED),
                correlation_id=correlation_id,
            )

            # Create result
            result_status = (
                OperationStatus.SUCCESS
                if execution.status == WorkflowStatus.COMPLETED
                else OperationStatus.FAILURE
            )

            result = OperationResult(
                status=result_status,
                data={
                    "workflow_id": execution.id,
                    "workflow_name": workflow_name,
                    "status": execution.status.value,
                    "results": execution.results,
                    "steps_completed": len(
                        [s for s in execution.steps if s.status == StepStatus.COMPLETED]
                    ),
                    "total_steps": len(execution.steps),
                    "execution_time": execution.execution_time,
                },
                correlation_id=correlation_id,
                execution_time=time.time() - start_time,
                performance_met=(execution.execution_time < 300),  # 5 minute SLA
                metadata={
                    "workflow_template": workflow_name,
                    "steps_executed": len(execution.steps),
                    "errors": execution.errors,
                },
            )

            # Clean up completed workflow
            if execution.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
                self.active_workflows.pop(execution.id, None)

            logger.info(
                f"Modern workflow execution completed: {workflow_name}",
                extra={
                    "workflow_id": execution.id,
                    "status": execution.status.value,
                    "execution_time": execution.execution_time,
                    "correlation_id": correlation_id,
                },
            )

            return result

        except Exception as e:
            logger.error(
                f"Modern workflow execution error: {e}",
                extra={
                    "workflow_name": workflow_name,
                    "correlation_id": correlation_id,
                    "error": str(e),
                },
            )

            return OperationResult(
                status=OperationStatus.FAILURE,
                error_message=f"Workflow execution failed: {str(e)}",
                correlation_id=correlation_id,
                execution_time=time.time() - start_time,
                performance_met=False,
            )

    def _create_workflow_execution(
        self, workflow_name: str, context: Dict[str, Any], correlation_id: str
    ) -> ModernWorkflowExecution:
        """Create workflow execution from template"""

        template = self.workflow_templates[workflow_name]
        execution_id = f"wf_{uuid.uuid4().hex[:12]}"

        # Create steps from template
        steps = []
        for step_template in template["steps"]:
            step = ModernWorkflowStep(
                id=step_template["id"],
                name=step_template["name"],
                description=step_template["description"],
                handler=step_template["handler"],
                dependencies=step_template.get("dependencies", []),
                timeout_seconds=step_template.get("timeout_seconds", 300),
                retry_count=step_template.get("retry_count", 3),
            )
            steps.append(step)

        return ModernWorkflowExecution(
            id=execution_id,
            name=template["name"],
            description=template["description"],
            steps=steps,
            context=context,
            correlation_id=correlation_id,
        )

    async def _execute_workflow_steps(self, execution: ModernWorkflowExecution):
        """Execute workflow steps with dependency management"""

        completed_steps = set()

        while len(completed_steps) < len(execution.steps):
            # Find ready steps (dependencies completed)
            ready_steps = []
            for step in execution.steps:
                if step.status == StepStatus.PENDING and all(
                    dep in completed_steps for dep in step.dependencies
                ):
                    ready_steps.append(step)

            if not ready_steps:
                # Check for failed dependencies
                failed_steps = [
                    s for s in execution.steps if s.status == StepStatus.FAILED
                ]
                if failed_steps:
                    logger.error(
                        f"Workflow blocked by failed steps: {[s.id for s in failed_steps]}"
                    )
                    execution.status = WorkflowStatus.FAILED
                    break
                else:
                    logger.warning(
                        "No ready steps found - possible circular dependency"
                    )
                    break

            # Execute ready steps (could be parallel in the future)
            for step in ready_steps:
                await self._execute_step(step, execution)

                if step.status == StepStatus.COMPLETED:
                    completed_steps.add(step.id)
                elif step.status == StepStatus.FAILED:
                    logger.error(f"Step failed: {step.id} - {step.error}")
                    # Continue with other steps that don't depend on this one

    async def _execute_step(
        self, step: ModernWorkflowStep, execution: ModernWorkflowExecution
    ):
        """Execute a single workflow step"""

        step.status = StepStatus.RUNNING
        step.start_time = time.time()

        try:
            logger.debug(
                f"Executing step: {step.id}",
                extra={"workflow_id": execution.id, "step_name": step.name},
            )

            # Execute step handler with timeout
            step.result = await asyncio.wait_for(
                step.handler(execution.context, step), timeout=step.timeout_seconds
            )

            step.status = StepStatus.COMPLETED
            execution.results[step.id] = step.result

        except asyncio.TimeoutError:
            step.error = f"Step timed out after {step.timeout_seconds} seconds"
            step.status = StepStatus.FAILED
            logger.error(
                f"Step timeout: {step.id}",
                extra={"timeout_seconds": step.timeout_seconds},
            )

        except Exception as e:
            step.error = str(e)
            step.status = StepStatus.FAILED
            logger.error(f"Step execution error: {step.id} - {e}")

        finally:
            step.end_time = time.time()
            step.execution_time = step.end_time - step.start_time

    # ===== STEP HANDLERS =====

    async def _validate_query_step(
        self, context: Dict[str, Any], step: ModernWorkflowStep
    ) -> Dict[str, Any]:
        """Validate query step handler"""
        query = context.get("query", "")

        if not query or len(query.strip()) < 3:
            raise ValueError("Query too short or empty")

        return {
            "validated_query": query.strip(),
            "query_length": len(query),
            "validation_passed": True,
        }

    async def _agent_analysis_step(
        self, context: Dict[str, Any], step: ModernWorkflowStep
    ) -> Dict[str, Any]:
        """Agent analysis step handler"""
        query = context.get("query", "")
        domain = context.get("domain")

        # Create agent request
        agent_request = AgentRequest(
            operation_type="workflow_analysis",
            query=query,
            domain=domain,
            context=context,
            correlation_id=context.get("correlation_id"),
        )

        # Placeholder for actual agent coordination
        # In full implementation: result = await agent.run(query, deps=deps)

        return {
            "analysis_completed": True,
            "query_complexity": "medium",
            "recommended_approach": "tri_modal_search",
            "domain_detected": domain or "general",
        }

    async def _search_execution_step(
        self, context: Dict[str, Any], step: ModernWorkflowStep
    ) -> Dict[str, Any]:
        """Search execution step handler"""
        # Placeholder for tri-modal search execution
        await asyncio.sleep(0.1)  # Simulate search time

        return {
            "search_completed": True,
            "vector_results": 10,
            "graph_results": 5,
            "gnn_results": 3,
            "total_results": 18,
        }

    async def _result_synthesis_step(
        self, context: Dict[str, Any], step: ModernWorkflowStep
    ) -> Dict[str, Any]:
        """Result synthesis step handler"""
        # Placeholder for result synthesis
        await asyncio.sleep(0.05)  # Simulate synthesis time

        return {
            "synthesis_completed": True,
            "final_answer": f"Synthesized response for: {context.get('query', 'unknown query')}",
            "confidence": 0.85,
        }

    # ===== UTILITY METHODS =====

    def _calculate_total_progress(
        self, step: WorkflowStep, step_progress: float
    ) -> float:
        """Calculate total workflow progress based on current step"""
        step_weights = {
            WorkflowStep.INITIALIZATION: 0.05,
            WorkflowStep.DATA_LOADING: 0.15,
            WorkflowStep.BLOB_STORAGE: 0.10,
            WorkflowStep.KNOWLEDGE_EXTRACTION: 0.25,
            WorkflowStep.SEARCH_INDEXING: 0.20,
            WorkflowStep.COSMOS_STORAGE: 0.15,
            WorkflowStep.VALIDATION: 0.05,
            WorkflowStep.COMPLETION: 0.05,
        }

        total_progress = 0.0
        step_list = list(WorkflowStep)
        current_index = step_list.index(step)

        # Add completed steps
        for i in range(current_index):
            total_progress += step_weights[step_list[i]]

        # Add current step progress
        total_progress += step_weights[step] * step_progress

        return min(total_progress, 1.0)

    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of active modern workflow"""
        execution = self.active_workflows.get(workflow_id)
        if not execution:
            return None

        return {
            "id": execution.id,
            "name": execution.name,
            "status": execution.status.value,
            "progress": len(
                [s for s in execution.steps if s.status == StepStatus.COMPLETED]
            )
            / len(execution.steps),
            "current_step": next(
                (s.name for s in execution.steps if s.status == StepStatus.RUNNING),
                None,
            ),
            "execution_time": time.time() - execution.start_time
            if execution.start_time
            else 0,
            "errors": execution.errors,
        }

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel active modern workflow"""
        execution = self.active_workflows.get(workflow_id)
        if not execution or execution.status != WorkflowStatus.RUNNING:
            return False

        execution.status = WorkflowStatus.CANCELLED
        return True

    async def health_check(self) -> Dict[str, Any]:
        """Health check for consolidated workflow service"""
        return {
            "status": "healthy",
            "legacy_workflows": len(self.progress_trackers),
            "active_modern_workflows": len(self.active_workflows),
            "registered_templates": len(self.workflow_templates),
            "templates": list(self.workflow_templates.keys()),
        }


# Backward compatibility alias
WorkflowService = ConsolidatedWorkflowService
