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

# ...existing code...
import logging
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from datetime import datetime
from config.settings import azure_settings

logger = logging.getLogger(__name__)

class AzureServiceType(Enum):
    """Azure service enumeration for Universal RAG pipeline"""
    AZURE_OPENAI = "azure_openai"
    COGNITIVE_SEARCH = "cognitive_search"
    COSMOS_DB = "cosmos_db"
    BLOB_STORAGE = "blob_storage"
    ML_COMPUTE = "ml_compute"

@dataclass
class AzureRAGWorkflowStep:
    """
    Azure service-centric workflow step
    Eliminates layer complexity while providing full Azure service transparency
    """
    query_id: str
    step_number: int
    azure_service: AzureServiceType
    operation_name: str
    status: str
    processing_time_ms: Optional[float] = None
    azure_region: str = None
    service_endpoint: str = None
    request_id: str = None
    cost_estimate_usd: Optional[float] = None
    diagnostics_enabled: bool = False
    technical_diagnostics: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        """Initialize Azure service metadata from configuration"""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.azure_region is None:
            self.azure_region = getattr(azure_settings, 'azure_location', 'eastus')
        if self.service_endpoint is None:
            self.service_endpoint = self._get_service_endpoint()

    def _get_service_endpoint(self) -> str:
        """Get service endpoint from azure_settings configuration"""
        endpoint_mapping = {
            AzureServiceType.AZURE_OPENAI: getattr(azure_settings, 'azure_openai_endpoint', ''),
            AzureServiceType.COGNITIVE_SEARCH: getattr(azure_settings, 'azure_search_endpoint', ''),
            AzureServiceType.COSMOS_DB: getattr(azure_settings, 'cosmos_db_endpoint', ''),
            AzureServiceType.BLOB_STORAGE: getattr(azure_settings, 'azure_storage_account_url', ''),
            AzureServiceType.ML_COMPUTE: getattr(azure_settings, 'azure_ml_workspace_url', '')
        }
        return endpoint_mapping.get(self.azure_service, '')

    def to_dict(self, include_diagnostics: bool = None) -> Dict[str, Any]:
        """Convert to dictionary with optional diagnostics"""
        if include_diagnostics is None:
            include_diagnostics = self.diagnostics_enabled
        result = asdict(self)
        if not include_diagnostics:
            result.pop('technical_diagnostics', None)
        result['azure_service'] = self.azure_service.value
        return result

    def to_azure_monitor_event(self) -> Dict[str, Any]:
        """Convert to Azure Application Insights telemetry format"""
        return {
            "eventType": "AzureRAGWorkflowStep",
            "properties": {
                "query_id": self.query_id,
                "azure_service": self.azure_service.value,
                "operation_name": self.operation_name,
                "azure_region": self.azure_region,
                "status": self.status
            },
            "measurements": {
                "processing_time_ms": self.processing_time_ms or 0,
                "cost_estimate_usd": self.cost_estimate_usd or 0
            }
        }


class AzureRAGWorkflowManager:
    """
    Simplified Azure service-centric workflow manager
    Focuses on Azure service transparency and operational excellence
    """

    def __init__(self, query_id: str, query_text: str, domain: str = "general"):
        self.query_id = query_id
        self.query_text = query_text
        self.domain = domain
        self.azure_steps: List[AzureRAGWorkflowStep] = []
        self.current_step_number = 0
        self.start_time = time.time()
        self.is_completed = False
        self.error_message: Optional[str] = None
        self.azure_region = getattr(azure_settings, 'azure_location', 'eastus')
        self.diagnostics_enabled = getattr(azure_settings, 'enable_diagnostics', False)
        self.cost_tracking_enabled = getattr(azure_settings, 'enable_cost_tracking', False)
        self.event_subscribers: List[Callable] = []
        logger.info(f"Initialized Azure RAG Workflow Manager for query: {query_id}")

    async def start_azure_service_step(
        self,
        azure_service: AzureServiceType,
        operation_name: str,
        estimated_cost_usd: Optional[float] = None
    ) -> int:
        self.current_step_number += 1
        step = AzureRAGWorkflowStep(
            query_id=self.query_id,
            step_number=self.current_step_number,
            azure_service=azure_service,
            operation_name=operation_name,
            status="in_progress",
            azure_region=self.azure_region,
            cost_estimate_usd=estimated_cost_usd,
            diagnostics_enabled=self.diagnostics_enabled,
            request_id=f"{self.query_id}_{self.current_step_number}_{int(time.time())}"
        )
        self.azure_steps.append(step)
        await self._notify_azure_subscribers("step_started", step.to_dict())
        if hasattr(self, 'app_insights') and self.app_insights:
            self.app_insights.track_event("azure_rag_step_started", step.to_azure_monitor_event())
        return self.current_step_number

    async def complete_azure_service_step(
        self,
        step_number: int,
        processing_time_ms: float,
        actual_cost_usd: Optional[float] = None,
        technical_diagnostics: Optional[Dict[str, Any]] = None
    ) -> None:
        step = self._get_step_by_number(step_number)
        if not step:
            raise ValueError(f"Step {step_number} not found")
        step.status = "completed"
        step.processing_time_ms = processing_time_ms
        if actual_cost_usd and self.cost_tracking_enabled:
            step.cost_estimate_usd = actual_cost_usd
        if technical_diagnostics and self.diagnostics_enabled:
            step.technical_diagnostics = technical_diagnostics
        await self._notify_azure_subscribers("step_completed", step.to_dict())
        if hasattr(self, 'app_insights') and self.app_insights:
            self.app_insights.track_event("azure_rag_step_completed", step.to_azure_monitor_event())

    def get_azure_service_summary(self) -> Dict[str, Any]:
        completed_steps = [s for s in self.azure_steps if s.status == "completed"]
        service_metrics = {}
        for step in completed_steps:
            service_name = step.azure_service.value
            if service_name not in service_metrics:
                service_metrics[service_name] = {
                    "operations_count": 0,
                    "total_processing_time_ms": 0,
                    "total_cost_usd": 0,
                    "service_endpoint": step.service_endpoint,
                    "azure_region": step.azure_region
                }
            service_metrics[service_name]["operations_count"] += 1
            service_metrics[service_name]["total_processing_time_ms"] += step.processing_time_ms or 0
            service_metrics[service_name]["total_cost_usd"] += step.cost_estimate_usd or 0
        return {
            "query_id": self.query_id,
            "domain": self.domain,
            "azure_region": self.azure_region,
            "total_azure_services_used": len(service_metrics),
            "total_processing_time_ms": sum(s.processing_time_ms or 0 for s in completed_steps),
            "total_cost_usd": sum(s.cost_estimate_usd or 0 for s in completed_steps),
            "service_breakdown": service_metrics,
            "diagnostics_enabled": self.diagnostics_enabled
        }

    async def _notify_azure_subscribers(self, event_type: str, data: Any) -> None:
        for callback in self.event_subscribers:
            try:
                await callback(event_type, data)
            except Exception as e:
                logger.error(f"Error notifying Azure subscriber: {e}", exc_info=True)

    def _get_step_by_number(self, step_number: int) -> Optional[AzureRAGWorkflowStep]:
        return next((s for s in self.azure_steps if s.step_number == step_number), None)

# Backward compatibility wrapper
class WorkflowStep:
    """Legacy three-layer interface wrapper for backward compatibility"""
    def __init__(self, azure_step: AzureRAGWorkflowStep):
        self.azure_step = azure_step
    def to_layer_dict(self, layer: int) -> Dict[str, Any]:
        base_data = self.azure_step.to_dict(include_diagnostics=(layer >= 3))
        base_data.update({
            "user_friendly_name": f"🔧 {self.azure_step.azure_service.value.replace('_', ' ').title()}",
            "step_name": f"{self.azure_step.azure_service.value}_{self.azure_step.operation_name}",
            "technology": self.azure_step.azure_service.value,
            "details": f"Processing with {self.azure_step.azure_service.value}"
        })
        return base_data

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
        # Legacy: return layer dicts for backward compatibility
        return [WorkflowStep(step).to_layer_dict(layer) for step in self.azure_steps]

    # Legacy summary removed; use get_azure_service_summary from AzureRAGWorkflowManager

    # Legacy _get_step removed; use _get_step_by_number from AzureRAGWorkflowManager

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
        self.active_workflows: Dict[str, AzureRAGWorkflowManager] = {}
        self.cleanup_threshold = 7200  # 2 hours instead of 1 hour
        self.completed_workflows: Dict[str, Dict[str, Any]] = {}  # Store completed results

    def register_workflow(self, workflow_manager: AzureRAGWorkflowManager) -> None:
        self.active_workflows[workflow_manager.query_id] = workflow_manager
        logger.debug(f"Registered workflow: {workflow_manager.query_id}")

    def get_workflow(self, query_id: str) -> Optional[AzureRAGWorkflowManager]:
        workflow = self.active_workflows.get(query_id)
        if workflow:
            return workflow
        # Check completed workflows and restore if needed
        if query_id in self.completed_workflows:
            logger.info(f"Restoring completed workflow for streaming: {query_id}")
            completed_data = self.completed_workflows[query_id]
            restored_workflow = AzureRAGWorkflowManager(
                query_id=query_id,
                query_text=completed_data.get("query_text", ""),
                domain=completed_data.get("domain", "general")
            )
            restored_workflow.is_completed = True
            # Rehydrate steps as AzureRAGWorkflowStep objects from dicts
            step_dicts = completed_data.get("steps", [])
            restored_steps = []
            for step_dict in step_dicts:
                try:
                    step = AzureRAGWorkflowStep(
                        query_id=step_dict.get("query_id", query_id),
                        step_number=step_dict.get("step_number", 0),
                        azure_service=AzureServiceType(step_dict.get("azure_service", "azure_openai")),
                        operation_name=step_dict.get("operation_name", "unknown"),
                        status=step_dict.get("status", "completed"),
                        processing_time_ms=step_dict.get("processing_time_ms"),
                        azure_region=step_dict.get("azure_region"),
                        service_endpoint=step_dict.get("service_endpoint"),
                        request_id=step_dict.get("request_id"),
                        cost_estimate_usd=step_dict.get("cost_estimate_usd"),
                        diagnostics_enabled=step_dict.get("diagnostics_enabled", False),
                        technical_diagnostics=step_dict.get("technical_diagnostics"),
                        timestamp=step_dict.get("timestamp")
                    )
                    restored_steps.append(step)
                except Exception as e:
                    logger.warning(f"Failed to restore step from dict: {e}")
                    logger.debug(f"Problematic step dict: {step_dict}")
                    continue
            restored_workflow.azure_steps = restored_steps
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


def create_workflow_manager(query_text: str, domain: str = "general") -> AzureRAGWorkflowManager:
    """
    Create and register a new workflow manager

    Args:
        query_text: The query text
        domain: Domain name

    Returns:
        New workflow manager instance
    """
    query_id = str(int(time.time()))
    workflow_manager = AzureRAGWorkflowManager(query_id, query_text, domain)
    workflow_registry.register_workflow(workflow_manager)
    return workflow_manager


def get_workflow_manager(query_id: str) -> Optional[AzureRAGWorkflowManager]:
    """
    Get workflow manager by query ID

    Args:
        query_id: Query ID to look up

    Returns:
        Workflow manager if found, None otherwise
    """
    return workflow_registry.get_workflow(query_id)