"""
Simple Workflow Service - CODING_STANDARDS Compliant
Clean workflow service without over-engineering enterprise patterns.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SimpleWorkflowService:
    """
    Simple workflow service following CODING_STANDARDS.md:
    - Data-Driven Everything: Uses simple workflow state tracking
    - Universal Design: Works with any workflow type
    - Mathematical Foundation: Simple progress calculations
    """

    def __init__(self):
        """Initialize simple workflow service"""
        self.workflows = {}  # workflow_id -> workflow_state
        self.workflow_history = {}  # workflow_id -> list of states
        logger.info("Simple workflow service initialized")

    def start_workflow(
        self, workflow_id: str, workflow_type: str, steps: List[str]
    ) -> Dict[str, Any]:
        """Start a new workflow"""
        try:
            workflow_state = {
                "workflow_id": workflow_id,
                "workflow_type": workflow_type,
                "status": "running",
                "current_step": steps[0] if steps else "unknown",
                "completed_steps": [],
                "remaining_steps": steps.copy(),
                "progress": 0.0,
                "start_time": datetime.now(),
                "end_time": None,
                "results": {},
                "error": None,
            }

            self.workflows[workflow_id] = workflow_state
            self.workflow_history[workflow_id] = [workflow_state.copy()]

            logger.info(f"Workflow started: {workflow_id} ({workflow_type})")
            return {
                "success": True,
                "workflow_id": workflow_id,
                "status": "running",
                "message": "Workflow started successfully",
            }

        except Exception as e:
            logger.error(f"Failed to start workflow {workflow_id}: {e}")
            return {"success": False, "workflow_id": workflow_id, "error": str(e)}

    def update_workflow_progress(
        self, workflow_id: str, step_name: str, step_result: Any = None
    ) -> Dict[str, Any]:
        """Update workflow progress"""
        try:
            if workflow_id not in self.workflows:
                return {"success": False, "error": f"Workflow not found: {workflow_id}"}

            workflow = self.workflows[workflow_id]

            # Mark current step as completed
            if step_name not in workflow["completed_steps"]:
                workflow["completed_steps"].append(step_name)

            # Remove from remaining steps
            if step_name in workflow["remaining_steps"]:
                workflow["remaining_steps"].remove(step_name)

            # Store step result
            if step_result is not None:
                workflow["results"][step_name] = step_result

            # Calculate progress
            total_steps = len(workflow["completed_steps"]) + len(
                workflow["remaining_steps"]
            )
            if total_steps > 0:
                workflow["progress"] = (
                    len(workflow["completed_steps"]) / total_steps * 100
                )

            # Update current step
            if workflow["remaining_steps"]:
                workflow["current_step"] = workflow["remaining_steps"][0]
            else:
                workflow["current_step"] = "completed"
                workflow["status"] = "completed"
                workflow["end_time"] = datetime.now()
                workflow["progress"] = 100.0

            # Add to history
            self.workflow_history[workflow_id].append(workflow.copy())

            logger.info(
                f"Workflow updated: {workflow_id} - {step_name} completed ({workflow['progress']:.1f}%)"
            )
            return {
                "success": True,
                "workflow_id": workflow_id,
                "current_step": workflow["current_step"],
                "progress": workflow["progress"],
                "status": workflow["status"],
            }

        except Exception as e:
            logger.error(f"Failed to update workflow {workflow_id}: {e}")
            return {"success": False, "workflow_id": workflow_id, "error": str(e)}

    def fail_workflow(
        self, workflow_id: str, error: str, step_name: str = None
    ) -> Dict[str, Any]:
        """Mark workflow as failed"""
        try:
            if workflow_id not in self.workflows:
                return {"success": False, "error": f"Workflow not found: {workflow_id}"}

            workflow = self.workflows[workflow_id]
            workflow["status"] = "failed"
            workflow["error"] = error
            workflow["end_time"] = datetime.now()

            if step_name:
                workflow["current_step"] = step_name

            # Add to history
            self.workflow_history[workflow_id].append(workflow.copy())

            logger.error(f"Workflow failed: {workflow_id} - {error}")
            return {
                "success": True,
                "workflow_id": workflow_id,
                "status": "failed",
                "error": error,
            }

        except Exception as e:
            logger.error(f"Failed to mark workflow as failed {workflow_id}: {e}")
            return {"success": False, "workflow_id": workflow_id, "error": str(e)}

    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow status"""
        try:
            if workflow_id not in self.workflows:
                return {"success": False, "error": f"Workflow not found: {workflow_id}"}

            workflow = self.workflows[workflow_id]
            return {
                "success": True,
                "workflow_id": workflow_id,
                "status": workflow["status"],
                "current_step": workflow["current_step"],
                "progress": workflow["progress"],
                "completed_steps": workflow["completed_steps"],
                "remaining_steps": workflow["remaining_steps"],
                "start_time": (
                    workflow["start_time"].isoformat()
                    if workflow["start_time"]
                    else None
                ),
                "end_time": (
                    workflow["end_time"].isoformat() if workflow["end_time"] else None
                ),
                "results": workflow["results"],
                "error": workflow["error"],
            }

        except Exception as e:
            logger.error(f"Failed to get workflow status {workflow_id}: {e}")
            return {"success": False, "workflow_id": workflow_id, "error": str(e)}

    def list_workflows(self, status_filter: str = None) -> Dict[str, Any]:
        """List all workflows"""
        try:
            workflows = []
            for workflow_id, workflow in self.workflows.items():
                if status_filter is None or workflow["status"] == status_filter:
                    workflows.append(
                        {
                            "workflow_id": workflow_id,
                            "workflow_type": workflow["workflow_type"],
                            "status": workflow["status"],
                            "progress": workflow["progress"],
                            "current_step": workflow["current_step"],
                            "start_time": (
                                workflow["start_time"].isoformat()
                                if workflow["start_time"]
                                else None
                            ),
                        }
                    )

            return {
                "success": True,
                "workflows": workflows,
                "total_count": len(workflows),
            }

        except Exception as e:
            logger.error(f"Failed to list workflows: {e}")
            return {"success": False, "error": str(e)}

    def get_workflow_history(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow execution history"""
        try:
            if workflow_id not in self.workflow_history:
                return {
                    "success": False,
                    "error": f"Workflow history not found: {workflow_id}",
                }

            history = self.workflow_history[workflow_id]
            return {
                "success": True,
                "workflow_id": workflow_id,
                "history": history,
                "history_count": len(history),
            }

        except Exception as e:
            logger.error(f"Failed to get workflow history {workflow_id}: {e}")
            return {"success": False, "workflow_id": workflow_id, "error": str(e)}

    def cleanup_completed_workflows(self, older_than_hours: int = 24) -> Dict[str, Any]:
        """Clean up old completed workflows"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
            workflows_to_remove = []

            for workflow_id, workflow in self.workflows.items():
                if (
                    workflow["status"] in ["completed", "failed"]
                    and workflow["end_time"]
                    and workflow["end_time"] < cutoff_time
                ):
                    workflows_to_remove.append(workflow_id)

            # Remove old workflows
            for workflow_id in workflows_to_remove:
                del self.workflows[workflow_id]
                if workflow_id in self.workflow_history:
                    del self.workflow_history[workflow_id]

            logger.info(f"Cleaned up {len(workflows_to_remove)} old workflows")
            return {
                "success": True,
                "cleaned_count": len(workflows_to_remove),
                "remaining_count": len(self.workflows),
            }

        except Exception as e:
            logger.error(f"Failed to cleanup workflows: {e}")
            return {"success": False, "error": str(e)}


# Backward compatibility - Global instance
_workflow_service = SimpleWorkflowService()


# Backward compatibility functions
def start_workflow(
    workflow_id: str, workflow_type: str, steps: List[str]
) -> Dict[str, Any]:
    """Backward compatibility function"""
    return _workflow_service.start_workflow(workflow_id, workflow_type, steps)


def update_workflow_progress(
    workflow_id: str, step_name: str, step_result: Any = None
) -> Dict[str, Any]:
    """Backward compatibility function"""
    return _workflow_service.update_workflow_progress(
        workflow_id, step_name, step_result
    )


def fail_workflow(
    workflow_id: str, error: str, step_name: str = None
) -> Dict[str, Any]:
    """Backward compatibility function"""
    return _workflow_service.fail_workflow(workflow_id, error, step_name)


def get_workflow_status(workflow_id: str) -> Dict[str, Any]:
    """Backward compatibility function"""
    return _workflow_service.get_workflow_status(workflow_id)


def list_workflows(status_filter: str = None) -> Dict[str, Any]:
    """Backward compatibility function"""
    return _workflow_service.list_workflows(status_filter)


# Backward compatibility aliases
WorkflowService = SimpleWorkflowService
ConsolidatedWorkflowService = SimpleWorkflowService
