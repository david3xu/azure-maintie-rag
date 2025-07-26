#!/usr/bin/env python3
"""
Real-time Progress Tracker for Azure Workflows
=============================================

Provides real-time progress tracking with percentage updates,
detailed status information, and execution metrics.
"""

import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import threading
from dataclasses import dataclass
from enum import Enum


class WorkflowStep(Enum):
    """Workflow step enumeration for progress tracking"""
    INITIALIZATION = "initialization"
    DATA_LOADING = "data_loading"
    BLOB_STORAGE = "blob_storage"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    SEARCH_INDEXING = "search_indexing"
    COSMOS_STORAGE = "cosmos_storage"
    VALIDATION = "validation"
    COMPLETION = "completion"


@dataclass
class ProgressStep:
    """Individual progress step with detailed tracking"""
    name: str
    description: str
    weight: float  # Percentage weight of total workflow
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    details: Dict[str, Any] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class RealTimeProgressTracker:
    """
    Real-time progress tracker with percentage updates and detailed status
    """

    def __init__(self, workflow_name: str = "Azure Data Preparation"):
        self.workflow_name = workflow_name
        self.start_time = datetime.now()
        self.steps: List[ProgressStep] = []
        self.current_step_index = -1
        self.is_running = False
        self._lock = threading.Lock()

        # Initialize workflow steps with weights
        self._initialize_workflow_steps()

    def _initialize_workflow_steps(self):
        """Initialize workflow steps with their weights and descriptions"""
        self.steps = [
            ProgressStep(
                name="Initialization",
                description="Initializing Azure services and validating connections",
                weight=5.0
            ),
            ProgressStep(
                name="Data Loading",
                description="Loading raw documents from data directory",
                weight=10.0
            ),
            ProgressStep(
                name="Blob Storage",
                description="Uploading documents to Azure Blob Storage",
                weight=15.0
            ),
            ProgressStep(
                name="Knowledge Extraction",
                description="Extracting entities and relations using Azure OpenAI",
                weight=35.0
            ),
            ProgressStep(
                name="Search Indexing",
                description="Building search index with Azure Cognitive Search",
                weight=20.0
            ),
            ProgressStep(
                name="Cosmos Storage",
                description="Storing metadata and knowledge graph in Cosmos DB",
                weight=10.0
            ),
            ProgressStep(
                name="Validation",
                description="Validating workflow results and service health",
                weight=5.0
            )
        ]

    def start_workflow(self):
        """Start the workflow and begin progress tracking"""
        self.is_running = True
        self.start_time = datetime.now()
        print(f"\n🚀 Starting {self.workflow_name} Workflow")
        print(f"⏰ Start Time: {self.start_time.strftime('%H:%M:%S')}")
        print("=" * 60)

    def start_step(self, step_name: str, details: Dict[str, Any] = None):
        """Start a specific workflow step"""
        with self._lock:
            step = self._find_step(step_name)
            if step:
                step.status = "running"
                step.start_time = datetime.now()
                step.details = details or {}
                self.current_step_index = self.steps.index(step)

                print(f"\n🔄 Step {self.current_step_index + 1}/{len(self.steps)}: {step.name}")
                print(f"📝 {step.description}")
                if details:
                    for key, value in details.items():
                        print(f"   📊 {key}: {value}")

                self._display_progress()

    def update_step_progress(self, step_name: str, progress_details: Dict[str, Any]):
        """Update progress details for current step"""
        with self._lock:
            step = self._find_step(step_name)
            if step and step.status == "running":
                step.details.update(progress_details)
                self._display_progress()

    def complete_step(self, step_name: str, success: bool = True, error_message: str = None):
        """Complete a workflow step"""
        with self._lock:
            step = self._find_step(step_name)
            if step:
                step.status = "completed" if success else "failed"
                step.end_time = datetime.now()
                if step.start_time:
                    step.duration = (step.end_time - step.start_time).total_seconds()
                step.error_message = error_message

                status_icon = "✅" if success else "❌"
                print(f"\n{status_icon} Step {self.current_step_index + 1}/{len(self.steps)}: {step.name} - {'Completed' if success else 'Failed'}")
                if step.duration:
                    print(f"⏱️  Duration: {step.duration:.2f}s")
                if error_message:
                    print(f"❌ Error: {error_message}")

                self._display_progress()

    def _find_step(self, step_name: str) -> Optional[ProgressStep]:
        """Find step by name"""
        for step in self.steps:
            if step.name.lower() == step_name.lower():
                return step
        return None

    def _display_progress(self):
        """Display current progress with percentage and details"""
        total_progress = 0.0
        completed_weight = 0.0

        for step in self.steps:
            if step.status == "completed":
                completed_weight += step.weight
            elif step.status == "running":
                # Add partial weight for running step
                completed_weight += step.weight * 0.5

        total_progress = (completed_weight / sum(step.weight for step in self.steps)) * 100

        # Calculate elapsed time
        elapsed = datetime.now() - self.start_time
        elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds

        # Progress bar
        bar_length = 30
        filled_length = int(bar_length * total_progress / 100)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)

        print(f"\n📊 Progress: [{bar}] {total_progress:.1f}%")
        print(f"⏱️  Elapsed: {elapsed_str}")

        # Current step details
        if self.current_step_index >= 0 and self.current_step_index < len(self.steps):
            current_step = self.steps[self.current_step_index]
            if current_step.status == "running" and current_step.details:
                print(f"🔄 Current: {current_step.name}")
                for key, value in current_step.details.items():
                    if isinstance(value, (int, float)):
                        print(f"   📈 {key}: {value}")
                    else:
                        print(f"   📋 {key}: {value}")

    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get comprehensive workflow summary"""
        total_duration = (datetime.now() - self.start_time).total_seconds()
        completed_steps = [s for s in self.steps if s.status == "completed"]
        failed_steps = [s for s in self.steps if s.status == "failed"]

        return {
            "workflow_name": self.workflow_name,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_duration": total_duration,
            "total_steps": len(self.steps),
            "completed_steps": len(completed_steps),
            "failed_steps": len(failed_steps),
            "success_rate": len(completed_steps) / len(self.steps) * 100,
            "step_details": [
                {
                    "name": step.name,
                    "status": step.status,
                    "duration": step.duration,
                    "details": step.details,
                    "error": step.error_message
                }
                for step in self.steps
            ]
        }

    def finish_workflow(self, success: bool = True):
        """Finish the workflow and display final summary"""
        self.is_running = False
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()

        print("\n" + "=" * 60)
        print(f"🏁 {self.workflow_name} Workflow {'Completed' if success else 'Failed'}")
        print(f"⏰ Duration: {total_duration:.2f}s")
        print(f"📊 Success Rate: {self.get_workflow_summary()['success_rate']:.1f}%")
        print("=" * 60)

        # Display step summary
        for i, step in enumerate(self.steps):
            status_icon = "✅" if step.status == "completed" else "❌" if step.status == "failed" else "⏳"
            duration_str = f" ({step.duration:.2f}s)" if step.duration else ""
            print(f"{status_icon} Step {i+1}: {step.name}{duration_str}")


# Convenience functions for easy integration
def create_progress_tracker(workflow_name: str = "Azure Data Preparation") -> RealTimeProgressTracker:
    """Create a new progress tracker instance"""
    return RealTimeProgressTracker(workflow_name)


async def track_async_operation(tracker: RealTimeProgressTracker, step_name: str, operation, *args, **kwargs):
    """Track an async operation with progress updates"""
    tracker.start_step(step_name)
    try:
        result = await operation(*args, **kwargs)
        tracker.complete_step(step_name, success=True)
        return result
    except Exception as e:
        tracker.complete_step(step_name, success=False, error_message=str(e))
        raise


def track_sync_operation(tracker: RealTimeProgressTracker, step_name: str, operation, *args, **kwargs):
    """Track a synchronous operation with progress updates"""
    tracker.start_step(step_name)
    try:
        result = operation(*args, **kwargs)
        tracker.complete_step(step_name, success=True)
        return result
    except Exception as e:
        tracker.complete_step(step_name, success=False, error_message=str(e))
        raise