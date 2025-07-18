"""
Universal Workflow Manager
Provides detailed, three-layer progressive disclosure for real-time workflow tracking
Matches frontend WorkflowStep interface exactly for seamless integration
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass, asdict
import asyncio
import json

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow step status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class WorkflowStep:
    """
    Complete workflow step model matching frontend interface exactly
    Supports three-layer progressive disclosure:
    - Layer 1: User-friendly (query_id, user_friendly_name, status, progress)
    - Layer 2: Technical (step_name, technology, details, processing_time_ms)
    - Layer 3: Diagnostics (technical_data, fix_applied)
    """
    query_id: str
    step_number: int
    step_name: str
    user_friendly_name: str
    status: str  # Will be converted from WorkflowStatus enum
    progress_percentage: int
    technology: str
    details: str
    processing_time_ms: Optional[float] = None
    fix_applied: Optional[str] = None
    technical_data: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        """Set timestamp if not provided"""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    def to_layer_dict(self, layer: int) -> Dict[str, Any]:
        """
        Convert to dictionary for specific layer
        Layer 1: User-friendly (minimal info)
        Layer 2: Technical details
        Layer 3: Full diagnostics
        """
        base_data = {
            "query_id": self.query_id,
            "step_number": self.step_number,
            "status": self.status,
            "progress_percentage": self.progress_percentage,
            "timestamp": self.timestamp
        }

        if layer >= 1:
            # Layer 1: User-friendly
            base_data.update({
                "user_friendly_name": self.user_friendly_name,
            })

        if layer >= 2:
            # Layer 2: Technical details
            base_data.update({
                "step_name": self.step_name,
                "technology": self.technology,
                "details": self.details,
                "processing_time_ms": self.processing_time_ms,
            })

        if layer >= 3:
            # Layer 3: Full diagnostics
            base_data.update({
                "fix_applied": self.fix_applied,
                "technical_data": self.technical_data,
            })

        return base_data


class UniversalWorkflowManager:
    """
    Universal Workflow Manager for real-time progress tracking

    Provides sophisticated three-layer progressive disclosure:
    - Matches frontend WorkflowStep interface exactly
    - Supports real-time streaming via Server-Sent Events
    - Integrates with all Universal RAG components
    - Tracks detailed technical metrics and diagnostics
    """

    def __init__(self, query_id: str, query_text: str, domain: str = "general"):
        """
        Initialize workflow manager for a specific query

        Args:
            query_id: Unique identifier for the query
            query_text: The original query text
            domain: Domain name for context
        """
        self.query_id = query_id
        self.query_text = query_text
        self.domain = domain

        # Workflow tracking
        self.steps: List[WorkflowStep] = []
        self.current_step_number = 0
        self.start_time = time.time()
        self.total_progress = 0
        self.is_completed = False
        self.has_error = False
        self.error_message: Optional[str] = None

        # Event streaming
        self.event_subscribers: List[Callable] = []
        self.step_start_times: Dict[int, float] = {}

        # Performance tracking
        self.performance_metrics = {
            "total_steps": 0,
            "completed_steps": 0,
            "average_step_time": 0.0,
            "fixes_applied": 0,
            "technical_optimizations": 0
        }

        logger.info(f"Initialized Universal Workflow Manager for query: {query_id}")

    async def start_step(
        self,
        step_name: str,
        user_friendly_name: str,
        technology: str,
        estimated_progress: int,
        technical_data: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Start a new workflow step

        Args:
            step_name: Technical name of the step
            user_friendly_name: User-friendly description
            technology: Technology/component being used
            estimated_progress: Estimated progress percentage after this step
            technical_data: Optional technical diagnostic data

        Returns:
            Step number for tracking
        """
        self.current_step_number += 1
        step_number = self.current_step_number
        self.step_start_times[step_number] = time.time()

        step = WorkflowStep(
            query_id=self.query_id,
            step_number=step_number,
            step_name=step_name,
            user_friendly_name=user_friendly_name,
            status=WorkflowStatus.IN_PROGRESS.value,
            progress_percentage=estimated_progress,
            technology=technology,
            details="Processing...",
            technical_data=technical_data or {}
        )

        self.steps.append(step)
        self.performance_metrics["total_steps"] += 1

        # Notify subscribers
        await self._notify_subscribers("step_started", step)

        logger.info(f"Started step {step_number}: {step_name} ({user_friendly_name})")
        return step_number

    async def update_step(
        self,
        step_number: int,
        details: str,
        progress_percentage: Optional[int] = None,
        technical_data: Optional[Dict[str, Any]] = None,
        fix_applied: Optional[str] = None
    ) -> None:
        """
        Update an existing workflow step

        Args:
            step_number: Step number to update
            details: Updated details description
            progress_percentage: Updated progress percentage
            technical_data: Updated technical data
            fix_applied: Description of any fix/optimization applied
        """
        step = self._get_step(step_number)
        if not step:
            logger.warning(f"Attempted to update non-existent step: {step_number}")
            return

        step.details = details
        if progress_percentage is not None:
            step.progress_percentage = progress_percentage
            self.total_progress = max(self.total_progress, progress_percentage)

        if technical_data:
            step.technical_data = step.technical_data or {}
            step.technical_data.update(technical_data)

        if fix_applied:
            step.fix_applied = fix_applied
            self.performance_metrics["fixes_applied"] += 1

        step.timestamp = datetime.now().isoformat()

        # Notify subscribers
        await self._notify_subscribers("step_updated", step)

        logger.debug(f"Updated step {step_number}: {details}")

    async def complete_step(
        self,
        step_number: int,
        details: str,
        final_progress: int,
        technical_data: Optional[Dict[str, Any]] = None,
        fix_applied: Optional[str] = None
    ) -> None:
        """
        Complete a workflow step

        Args:
            step_number: Step number to complete
            details: Final details description
            final_progress: Final progress percentage
            technical_data: Final technical data
            fix_applied: Description of any fix/optimization applied
        """
        step = self._get_step(step_number)
        if not step:
            logger.warning(f"Attempted to complete non-existent step: {step_number}")
            return

        # Calculate processing time
        start_time = self.step_start_times.get(step_number, time.time())
        processing_time_ms = (time.time() - start_time) * 1000

        step.status = WorkflowStatus.COMPLETED.value
        step.details = details
        step.progress_percentage = final_progress
        step.processing_time_ms = processing_time_ms
        step.timestamp = datetime.now().isoformat()

        if technical_data:
            step.technical_data = step.technical_data or {}
            step.technical_data.update(technical_data)

        if fix_applied:
            step.fix_applied = fix_applied
            self.performance_metrics["fixes_applied"] += 1

        self.total_progress = max(self.total_progress, final_progress)
        self.performance_metrics["completed_steps"] += 1

        # Update average step time
        self._update_average_step_time(processing_time_ms)

        # Notify subscribers
        await self._notify_subscribers("step_completed", step)

        logger.info(f"Completed step {step_number}: {details} ({processing_time_ms:.1f}ms)")

    async def fail_step(
        self,
        step_number: int,
        error_details: str,
        technical_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Mark a workflow step as failed

        Args:
            step_number: Step number that failed
            error_details: Error description
            technical_data: Technical diagnostic data about the failure
        """
        step = self._get_step(step_number)
        if not step:
            logger.warning(f"Attempted to fail non-existent step: {step_number}")
            return

        # Calculate processing time
        start_time = self.step_start_times.get(step_number, time.time())
        processing_time_ms = (time.time() - start_time) * 1000

        step.status = WorkflowStatus.ERROR.value
        step.details = f"Error: {error_details}"
        step.processing_time_ms = processing_time_ms
        step.timestamp = datetime.now().isoformat()

        if technical_data:
            step.technical_data = step.technical_data or {}
            step.technical_data.update(technical_data)

        self.has_error = True
        self.error_message = error_details

        # Notify subscribers
        await self._notify_subscribers("step_failed", step)

        logger.error(f"Failed step {step_number}: {error_details}")

    async def complete_workflow(
        self,
        final_results: Dict[str, Any],
        total_processing_time: float
    ) -> None:
        """
        Complete the entire workflow

        Args:
            final_results: Final query processing results
            total_processing_time: Total processing time in seconds
        """
        self.is_completed = True
        self.total_progress = 100

        # Update performance metrics
        self.performance_metrics.update({
            "total_processing_time": total_processing_time,
            "steps_per_second": self.performance_metrics["total_steps"] / total_processing_time if total_processing_time > 0 else 0,
            "success_rate": self.performance_metrics["completed_steps"] / self.performance_metrics["total_steps"] if self.performance_metrics["total_steps"] > 0 else 0
        })

        # Notify subscribers
        await self._notify_subscribers("workflow_completed", {
            "query_id": self.query_id,
            "results": final_results,
            "performance": self.performance_metrics,
            "timestamp": datetime.now().isoformat()
        })

        logger.info(f"Completed workflow for query {self.query_id} in {total_processing_time:.2f}s")

    async def fail_workflow(self, error_message: str) -> None:
        """
        Mark the entire workflow as failed

        Args:
            error_message: Overall failure description
        """
        self.has_error = True
        self.error_message = error_message

        # Notify subscribers
        await self._notify_subscribers("workflow_failed", {
            "query_id": self.query_id,
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        })

        logger.error(f"Failed workflow for query {self.query_id}: {error_message}")

    def subscribe_to_events(self, callback: Callable) -> None:
        """
        Subscribe to workflow events

        Args:
            callback: Async function to call on events
        """
        self.event_subscribers.append(callback)

    def get_steps_for_layer(self, layer: int) -> List[Dict[str, Any]]:
        """
        Get all steps formatted for specific disclosure layer

        Args:
            layer: Disclosure layer (1=user-friendly, 2=technical, 3=diagnostic)

        Returns:
            List of step dictionaries for the specified layer
        """
        return [step.to_layer_dict(layer) for step in self.steps]

    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get comprehensive workflow summary"""
        total_time = (time.time() - self.start_time) * 1000

        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "domain": self.domain,
            "status": "completed" if self.is_completed else ("error" if self.has_error else "running"),
            "total_steps": len(self.steps),
            "completed_steps": len([s for s in self.steps if s.status == WorkflowStatus.COMPLETED.value]),
            "failed_steps": len([s for s in self.steps if s.status == WorkflowStatus.ERROR.value]),
            "total_progress": self.total_progress,
            "total_time_ms": total_time,
            "performance_metrics": self.performance_metrics,
            "error_message": self.error_message,
            "timestamp": datetime.now().isoformat()
        }

    def _get_step(self, step_number: int) -> Optional[WorkflowStep]:
        """Get step by number"""
        return next((s for s in self.steps if s.step_number == step_number), None)

    def _update_average_step_time(self, processing_time_ms: float) -> None:
        """Update average step processing time"""
        completed = self.performance_metrics["completed_steps"]
        if completed == 1:
            self.performance_metrics["average_step_time"] = processing_time_ms
        else:
            current_avg = self.performance_metrics["average_step_time"]
            new_avg = ((current_avg * (completed - 1)) + processing_time_ms) / completed
            self.performance_metrics["average_step_time"] = new_avg

    async def _notify_subscribers(self, event_type: str, data: Any) -> None:
        """Notify all event subscribers"""
        for callback in self.event_subscribers:
            try:
                await callback(event_type, data)
            except Exception as e:
                logger.error(f"Error notifying workflow subscriber: {e}", exc_info=True)


class WorkflowManagerRegistry:
    """
    Global registry for active workflow managers
    Provides centralized access for streaming endpoints
    """

    def __init__(self):
        self.active_workflows: Dict[str, UniversalWorkflowManager] = {}
        self.cleanup_threshold = 7200  # 2 hours instead of 1 hour
        self.completed_workflows: Dict[str, Dict[str, Any]] = {}  # Store completed results

    def register_workflow(self, workflow_manager: UniversalWorkflowManager) -> None:
        self.active_workflows[workflow_manager.query_id] = workflow_manager
        logger.debug(f"Registered workflow: {workflow_manager.query_id}")

    def get_workflow(self, query_id: str) -> Optional[UniversalWorkflowManager]:
        workflow = self.active_workflows.get(query_id)
        if workflow:
            return workflow
        # Check completed workflows and restore if needed
        if query_id in self.completed_workflows:
            logger.info(f"Restoring completed workflow for streaming: {query_id}")
            completed_data = self.completed_workflows[query_id]
            restored_workflow = UniversalWorkflowManager(
                query_id=query_id,
                query_text=completed_data.get("query_text", ""),
                domain=completed_data.get("domain", "general")
            )
            restored_workflow.is_completed = True
            restored_workflow.performance_metrics = completed_data.get("performance_metrics", {})
            # Rehydrate steps as WorkflowStep objects from dicts
            step_dicts = completed_data.get("steps", [])
            restored_steps = []
            for step_dict in step_dicts:
                try:
                    step = WorkflowStep(
                        query_id=step_dict.get("query_id", query_id),
                        step_number=step_dict.get("step_number", 0),
                        step_name=step_dict.get("step_name", "unknown"),
                        user_friendly_name=step_dict.get("user_friendly_name", "Processing..."),
                        status=step_dict.get("status", "completed"),
                        progress_percentage=step_dict.get("progress_percentage", 100),
                        technology=step_dict.get("technology", "Universal RAG"),
                        details=step_dict.get("details", "Step completed"),
                        processing_time_ms=step_dict.get("processing_time_ms"),
                        fix_applied=step_dict.get("fix_applied"),
                        technical_data=step_dict.get("technical_data"),
                        timestamp=step_dict.get("timestamp")
                    )
                    restored_steps.append(step)
                except Exception as e:
                    logger.warning(f"Failed to restore step from dict: {e}")
                    logger.debug(f"Problematic step dict: {step_dict}")
                    continue
            restored_workflow.steps = restored_steps
            self.active_workflows[query_id] = restored_workflow
            return restored_workflow
        return None

    def complete_workflow(self, query_id: str, results: Dict[str, Any]) -> None:
        if query_id in self.active_workflows:
            workflow = self.active_workflows[query_id]
            steps_data = []
            for step in workflow.steps:
                try:
                    step_dict = step.to_dict()
                    steps_data.append(step_dict)
                except Exception as e:
                    logger.warning(f"Failed to serialize step {step.step_number}: {e}")
                    continue
            self.completed_workflows[query_id] = {
                "query_text": workflow.query_text,
                "domain": workflow.domain,
                "steps": steps_data,
                "performance_metrics": workflow.performance_metrics,
                "results": results,
                "completed_at": time.time()
            }
            logger.info(f"Completed workflow stored for streaming: {query_id}")

    def unregister_workflow(self, query_id: str) -> None:
        if query_id in self.active_workflows:
            del self.active_workflows[query_id]
            logger.debug(f"Unregistered workflow: {query_id}")

    def cleanup_old_workflows(self) -> None:
        current_time = time.time()
        to_remove_active = []
        to_remove_completed = []
        for query_id, workflow in self.active_workflows.items():
            age = current_time - workflow.start_time
            if age > self.cleanup_threshold and (workflow.is_completed or workflow.has_error):
                to_remove_active.append(query_id)
        for query_id, completed_data in self.completed_workflows.items():
            age = current_time - completed_data.get("completed_at", current_time)
            if age > (self.cleanup_threshold * 2):
                to_remove_completed.append(query_id)
        for query_id in to_remove_active:
            self.unregister_workflow(query_id)
        for query_id in to_remove_completed:
            if query_id in self.completed_workflows:
                del self.completed_workflows[query_id]
        if to_remove_active or to_remove_completed:
            logger.info(f"Cleaned up {len(to_remove_active)} active and {len(to_remove_completed)} completed workflows")


# Global workflow registry
workflow_registry = WorkflowManagerRegistry()


def create_workflow_manager(query_text: str, domain: str = "general") -> UniversalWorkflowManager:
    """
    Create and register a new workflow manager

    Args:
        query_text: The query text
        domain: Domain name

    Returns:
        New workflow manager instance
    """
    query_id = str(uuid.uuid4())
    workflow_manager = UniversalWorkflowManager(query_id, query_text, domain)
    workflow_registry.register_workflow(workflow_manager)
    return workflow_manager


def get_workflow_manager(query_id: str) -> Optional[UniversalWorkflowManager]:
    """
    Get workflow manager by query ID

    Args:
        query_id: Query ID to look up

    Returns:
        Workflow manager if found, None otherwise
    """
    return workflow_registry.get_workflow(query_id)