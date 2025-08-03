"""
Workflow State Models

Graph state models for workflow execution following target architecture.
Provides structured models for workflow and node state management.
"""

from typing import Dict, Any, List, Optional, Union, Literal
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class WorkflowExecutionState(str, Enum):
    """Workflow execution states"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class NodeExecutionState(str, Enum):
    """Node execution states"""
    READY = "ready"
    WAITING = "waiting"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class WorkflowType(str, Enum):
    """Supported workflow types"""
    CONFIG_EXTRACTION = "config_extraction"
    SEARCH = "search"
    KNOWLEDGE_PROCESSING = "knowledge_processing"
    DOMAIN_ANALYSIS = "domain_analysis"


class NodeMetrics(BaseModel):
    """Performance metrics for workflow nodes"""
    execution_time_seconds: float = Field(default=0.0, ge=0.0)
    memory_usage_mb: float = Field(default=0.0, ge=0.0)
    cpu_usage_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    retry_count: int = Field(default=0, ge=0)
    cache_hit: bool = Field(default=False)


class NodeResult(BaseModel):
    """Result data from node execution"""
    data: Any = Field(description="Node execution result data")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True


class WorkflowNodeState(BaseModel):
    """Complete state information for a workflow node"""
    node_id: str = Field(description="Unique node identifier")
    node_name: str = Field(description="Human-readable node name")
    state: NodeExecutionState = Field(default=NodeExecutionState.READY)
    dependencies: List[str] = Field(default_factory=list, description="Node dependencies")
    
    # Execution information
    start_time: Optional[datetime] = Field(default=None)
    end_time: Optional[datetime] = Field(default=None)
    result: Optional[NodeResult] = Field(default=None)
    error_message: Optional[str] = Field(default=None)
    
    # Performance metrics
    metrics: NodeMetrics = Field(default_factory=NodeMetrics)
    
    # Configuration
    max_retries: int = Field(default=3, ge=0)
    timeout_seconds: int = Field(default=300, gt=0)
    
    @property
    def execution_time(self) -> float:
        """Calculate execution time in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def is_terminal_state(self) -> bool:
        """Check if node is in a terminal state"""
        return self.state in [
            NodeExecutionState.COMPLETED,
            NodeExecutionState.FAILED,
            NodeExecutionState.SKIPPED,
            NodeExecutionState.CANCELLED
        ]


class WorkflowProgress(BaseModel):
    """Workflow execution progress information"""
    total_nodes: int = Field(ge=0)
    completed_nodes: int = Field(ge=0)
    failed_nodes: int = Field(ge=0)
    skipped_nodes: int = Field(ge=0)
    executing_nodes: int = Field(ge=0)
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage"""
        if self.total_nodes == 0:
            return 0.0
        
        finished_nodes = self.completed_nodes + self.failed_nodes + self.skipped_nodes
        return (finished_nodes / self.total_nodes) * 100.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of completed nodes"""
        finished_nodes = self.completed_nodes + self.failed_nodes
        if finished_nodes == 0:
            return 0.0
        
        return (self.completed_nodes / finished_nodes) * 100.0


class WorkflowConfiguration(BaseModel):
    """Workflow execution configuration"""
    workflow_type: WorkflowType
    parallel_execution: bool = Field(default=True)
    fail_fast: bool = Field(default=False)
    retry_failed_nodes: bool = Field(default=True)
    max_execution_time: int = Field(default=3600, gt=0, description="Max execution time in seconds")
    enable_state_persistence: bool = Field(default=True)
    enable_metrics_collection: bool = Field(default=True)


class WorkflowMetrics(BaseModel):
    """Overall workflow performance metrics"""
    total_execution_time: float = Field(default=0.0, ge=0.0)
    average_node_time: float = Field(default=0.0, ge=0.0)
    parallel_efficiency: float = Field(default=0.0, ge=0.0, le=1.0)
    memory_peak_mb: float = Field(default=0.0, ge=0.0)
    cache_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Performance targets
    meets_performance_target: bool = Field(default=False)
    performance_grade: str = Field(default="unknown")
    
    def calculate_performance_grade(self, target_time: float = 3.0):
        """Calculate performance grade based on execution time"""
        if self.total_execution_time <= target_time * 0.5:
            self.performance_grade = "excellent"
        elif self.total_execution_time <= target_time:
            self.performance_grade = "good"
        elif self.total_execution_time <= target_time * 1.5:
            self.performance_grade = "acceptable"
        else:
            self.performance_grade = "needs_optimization"
        
        self.meets_performance_target = self.total_execution_time <= target_time


class WorkflowState(BaseModel):
    """Complete workflow state information"""
    workflow_id: str = Field(description="Unique workflow identifier")
    workflow_name: str = Field(description="Human-readable workflow name")
    workflow_type: WorkflowType
    state: WorkflowExecutionState = Field(default=WorkflowExecutionState.PENDING)
    
    # Input and output
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    
    # Execution timing
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    
    # Node states
    nodes: Dict[str, WorkflowNodeState] = Field(default_factory=dict)
    
    # Progress and metrics
    progress: WorkflowProgress = Field(default_factory=WorkflowProgress)
    metrics: WorkflowMetrics = Field(default_factory=WorkflowMetrics)
    
    # Configuration
    configuration: WorkflowConfiguration = Field(default_factory=WorkflowConfiguration)
    
    # Error handling
    error_message: Optional[str] = Field(default=None)
    warnings: List[str] = Field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True
    
    def update_progress(self):
        """Update workflow progress based on node states"""
        self.progress.total_nodes = len(self.nodes)
        self.progress.completed_nodes = len([n for n in self.nodes.values() if n.state == NodeExecutionState.COMPLETED])
        self.progress.failed_nodes = len([n for n in self.nodes.values() if n.state == NodeExecutionState.FAILED])
        self.progress.skipped_nodes = len([n for n in self.nodes.values() if n.state == NodeExecutionState.SKIPPED])
        self.progress.executing_nodes = len([n for n in self.nodes.values() if n.state == NodeExecutionState.EXECUTING])
    
    def update_metrics(self):
        """Update workflow metrics based on node execution"""
        node_times = [n.execution_time for n in self.nodes.values() if n.execution_time > 0]
        
        if node_times:
            self.metrics.total_execution_time = max(node_times)  # Assuming parallel execution
            self.metrics.average_node_time = sum(node_times) / len(node_times)
        
        # Calculate cache hit rate
        total_nodes = len(self.nodes)
        cache_hits = len([n for n in self.nodes.values() if n.metrics.cache_hit])
        if total_nodes > 0:
            self.metrics.cache_hit_rate = cache_hits / total_nodes
        
        # Update performance grade
        self.metrics.calculate_performance_grade()
    
    @property
    def is_terminal_state(self) -> bool:
        """Check if workflow is in a terminal state"""
        return self.state in [
            WorkflowExecutionState.COMPLETED,
            WorkflowExecutionState.FAILED,
            WorkflowExecutionState.CANCELLED
        ]
    
    @property
    def execution_time(self) -> float:
        """Calculate total execution time in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0


class WorkflowEvent(BaseModel):
    """Workflow execution event for logging and monitoring"""
    event_id: str = Field(description="Unique event identifier")
    workflow_id: str = Field(description="Associated workflow ID")
    event_type: str = Field(description="Type of event")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Event details
    node_id: Optional[str] = Field(default=None)
    previous_state: Optional[str] = Field(default=None)
    new_state: Optional[str] = Field(default=None)
    
    # Event data
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = Field(default=None)
    
    class Config:
        arbitrary_types_allowed = True


# Workflow-specific state models
class ConfigExtractionWorkflowState(WorkflowState):
    """Specialized state for config extraction workflows"""
    workflow_type: Literal[WorkflowType.CONFIG_EXTRACTION] = Field(default=WorkflowType.CONFIG_EXTRACTION)
    
    # Specific output data
    discovered_domains: Dict[str, Any] = Field(default_factory=dict)
    generated_config: Dict[str, Any] = Field(default_factory=dict)
    extraction_results: Dict[str, Any] = Field(default_factory=dict)


class SearchWorkflowState(WorkflowState):
    """Specialized state for search workflows"""
    workflow_type: Literal[WorkflowType.SEARCH] = Field(default=WorkflowType.SEARCH)
    
    # Specific output data
    search_results: Dict[str, Any] = Field(default_factory=dict)
    detected_domain: Dict[str, Any] = Field(default_factory=dict)
    response_generated: Dict[str, Any] = Field(default_factory=dict)


# Export all models
__all__ = [
    "WorkflowExecutionState",
    "NodeExecutionState",
    "WorkflowType",
    "NodeMetrics",
    "NodeResult",
    "WorkflowNodeState",
    "WorkflowProgress",
    "WorkflowConfiguration",
    "WorkflowMetrics",
    "WorkflowState",
    "WorkflowEvent",
    "ConfigExtractionWorkflowState",
    "SearchWorkflowState",
]