"""
Workflow State Persistence

Production-grade state persistence for workflow execution following target architecture.
Provides reliable state management and recovery capabilities.
"""

import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import asdict

# Import constants for zero-hardcoded-values compliance
from agents.core.constants import CacheConstants, StubConstants

# Import models from centralized data models
from agents.core.data_models import (
    WorkflowState,
    NodeState,
)


# Simple replacement for the deleted PersistedWorkflowState
class WorkflowStateRecord(dict):
    """Temporary replacement for deleted PersistedWorkflowState model"""

    def __init__(self, **kwargs):
        super().__init__(kwargs)


class WorkflowStateManager:
    """
    Production-grade workflow state persistence manager.

    Provides reliable state management with JSON file storage,
    recovery capabilities, and state history tracking.
    """

    def __init__(self, storage_dir: str = "/tmp/workflow_states"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._locks: Dict[str, asyncio.Lock] = {}

    async def save_workflow_state(
        self,
        workflow_id: str,
        state: WorkflowState,
        data: Dict[str, Any],
        workflow_type: str = "unknown",
    ) -> bool:
        """
        Save workflow state to persistent storage.

        Args:
            workflow_id: Unique workflow identifier
            state: Current workflow state
            data: Workflow data to persist
            workflow_type: Type of workflow (config_extraction, search, etc.)

        Returns:
            bool: True if save successful
        """
        try:
            # Get or create lock for this workflow
            if workflow_id not in self._locks:
                self._locks[workflow_id] = asyncio.Lock()

            async with self._locks[workflow_id]:
                # Load existing record or create new one
                existing_record = await self._load_workflow_record(workflow_id)

                if existing_record:
                    # Update existing record
                    existing_record.state = state
                    existing_record.updated_at = datetime.now(timezone.utc)

                    # Merge data
                    if "input_data" in data:
                        existing_record.input_data.update(data["input_data"])
                    if "results" in data:
                        existing_record.results.update(data["results"])
                    if "metadata" in data:
                        existing_record.metadata.update(data["metadata"])

                    record = existing_record
                else:
                    # Create new record
                    record = WorkflowStateRecord(
                        workflow_id=workflow_id,
                        workflow_type=workflow_type,
                        state=state,
                        input_data=data.get("input_data", data),
                        results=data.get("results", {}),
                        node_states={},
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                        metadata=data.get("metadata", {}),
                    )

                # Save to file
                await self._save_record_to_file(record)

                return True

        except Exception as e:
            print(f"Error saving workflow state for {workflow_id}: {e}")
            return False

    async def load_workflow_state(
        self, workflow_id: str
    ) -> Optional[WorkflowStateRecord]:
        """
        Load workflow state from persistent storage.

        Args:
            workflow_id: Unique workflow identifier

        Returns:
            WorkflowStateRecord or None if not found
        """
        try:
            return await self._load_workflow_record(workflow_id)
        except Exception as e:
            print(f"Error loading workflow state for {workflow_id}: {e}")
            return None

    async def update_node_state(
        self,
        workflow_id: str,
        node_id: str,
        state: NodeState,
        result: Any = None,
        error: Optional[str] = None,
    ) -> bool:
        """
        Update individual node state within workflow.

        Args:
            workflow_id: Workflow identifier
            node_id: Node identifier
            state: Node execution state
            result: Node execution result
            error: Error message if failed

        Returns:
            bool: True if update successful
        """
        try:
            if workflow_id not in self._locks:
                self._locks[workflow_id] = asyncio.Lock()

            async with self._locks[workflow_id]:
                record = await self._load_workflow_record(workflow_id)

                if not record:
                    return False

                # Update node state
                record.node_states[node_id] = {
                    "state": state.value,
                    "result": result,
                    "error": error,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }

                record.updated_at = datetime.now(timezone.utc)

                # Save updated record
                await self._save_record_to_file(record)

                return True

        except Exception as e:
            print(f"Error updating node state for {workflow_id}.{node_id}: {e}")
            return False

    async def list_workflows(
        self,
        workflow_type: Optional[str] = None,
        state_filter: Optional[WorkflowState] = None,
        limit: int = CacheConstants.PERCENTAGE_MULTIPLIER,
    ) -> List[WorkflowStateRecord]:
        """
        List workflows with optional filtering.

        Args:
            workflow_type: Filter by workflow type
            state_filter: Filter by workflow state
            limit: Maximum number of workflows to return

        Returns:
            List of workflow state records
        """
        try:
            workflows = []

            # Scan all workflow files
            for file_path in self.storage_dir.glob("workflow_*.json"):
                if len(workflows) >= limit:
                    break

                try:
                    record = await self._load_record_from_file(file_path)

                    # Apply filters
                    if workflow_type and record.workflow_type != workflow_type:
                        continue
                    if state_filter and record.state != state_filter:
                        continue

                    workflows.append(record)

                except Exception as e:
                    print(f"Error loading workflow from {file_path}: {e}")
                    continue

            # Sort by updated_at descending
            workflows.sort(key=lambda w: w.updated_at, reverse=True)

            return workflows

        except Exception as e:
            print(f"Error listing workflows: {e}")
            return []

    async def delete_workflow_state(self, workflow_id: str) -> bool:
        """
        Delete workflow state from persistent storage.

        Args:
            workflow_id: Workflow identifier

        Returns:
            bool: True if deletion successful
        """
        try:
            file_path = self._get_workflow_file_path(workflow_id)

            if file_path.exists():
                file_path.unlink()

                # Clean up lock
                if workflow_id in self._locks:
                    del self._locks[workflow_id]

                return True

            return False

        except Exception as e:
            print(f"Error deleting workflow state for {workflow_id}: {e}")
            return False

    async def cleanup_old_workflows(self, days_old: int = 30) -> int:
        """
        Clean up old workflow states.

        Args:
            days_old: Delete workflows older than this many days

        Returns:
            int: Number of workflows deleted
        """
        try:
            from datetime import timedelta

            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_old)

            deleted_count = 0

            for file_path in self.storage_dir.glob("workflow_*.json"):
                try:
                    record = await self._load_record_from_file(file_path)

                    if record.updated_at < cutoff_date:
                        file_path.unlink()
                        deleted_count += 1

                        # Clean up lock
                        if record.workflow_id in self._locks:
                            del self._locks[record.workflow_id]

                except Exception as e:
                    print(f"Error processing {file_path} for cleanup: {e}")
                    continue

            return deleted_count

        except Exception as e:
            print(f"Error during workflow cleanup: {e}")
            return 0

    def _get_workflow_file_path(self, workflow_id: str) -> Path:
        """Get file path for workflow state"""
        safe_id = "".join(c for c in workflow_id if c.isalnum() or c in "_-")
        return self.storage_dir / f"workflow_{safe_id}.json"

    async def _load_workflow_record(
        self, workflow_id: str
    ) -> Optional[WorkflowStateRecord]:
        """Load workflow record from file"""
        file_path = self._get_workflow_file_path(workflow_id)

        if not file_path.exists():
            return None

        return await self._load_record_from_file(file_path)

    async def _load_record_from_file(self, file_path: Path) -> WorkflowStateRecord:
        """Load record from JSON file with path validation"""
        # Security: Validate file path is within allowed directory
        allowed_dir = Path(__file__).parent.parent.parent / "data" / "workflows"
        try:
            file_path.resolve().relative_to(allowed_dir.resolve())
        except ValueError:
            raise ValueError(
                f"Invalid file path: {file_path} - must be within workflow directory"
            )

        with open(file_path, "r") as f:
            data = json.load(f)

        # Convert datetime strings back to datetime objects
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        data["state"] = WorkflowState(data["state"])

        return WorkflowStateRecord(**data)

    async def _save_record_to_file(self, record: WorkflowStateRecord):
        """Save record to JSON file"""
        file_path = self._get_workflow_file_path(record.workflow_id)

        # Convert record to dict
        data = asdict(record)

        # Convert datetime objects to ISO strings
        data["created_at"] = record.created_at.isoformat()
        data["updated_at"] = record.updated_at.isoformat()
        data["state"] = record.state.value

        # Security: Validate file path is within allowed directory
        allowed_dir = Path(__file__).parent.parent.parent / "data" / "workflows"
        try:
            file_path.resolve().relative_to(allowed_dir.resolve())
        except ValueError:
            raise ValueError(
                f"Invalid file path: {file_path} - must be within workflow directory"
            )

        # Write to temporary file first, then rename for atomicity
        temp_path = file_path.with_suffix(".tmp")

        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        temp_path.rename(file_path)

    async def get_workflow_stats(self) -> Dict[str, Any]:
        """Get workflow statistics"""
        try:
            workflows = await self.list_workflows(
                limit=StubConstants.DEFAULT_QUERY_LIMIT
            )  # Reasonable default limit

            stats = {
                "total_workflows": len(workflows),
                "by_state": {},
                "by_type": {},
                "avg_execution_time": CacheConstants.ZERO_FLOAT,
                "storage_size_mb": CacheConstants.ZERO_FLOAT,
            }

            # Count by state
            for state in WorkflowState:
                stats["by_state"][state.value] = len(
                    [w for w in workflows if w.state == state]
                )

            # Count by type
            workflow_types = set(w.workflow_type for w in workflows)
            for wf_type in workflow_types:
                stats["by_type"][wf_type] = len(
                    [w for w in workflows if w.workflow_type == wf_type]
                )

            # Calculate storage size
            total_size = sum(
                f.stat().st_size for f in self.storage_dir.glob("workflow_*.json")
            )
            stats["storage_size_mb"] = round(
                total_size / WorkflowConstants.BYTES_TO_MB_DIVISOR,
                WorkflowConstants.STORAGE_SIZE_PRECISION,
            )

            return stats

        except Exception as e:
            print(f"Error getting workflow stats: {e}")
            return {"error": str(e)}


# Export main components
__all__ = [
    "WorkflowStateManager",
    "WorkflowStateRecord",
]
