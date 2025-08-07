"""
Foundation Base Models and PydanticAI Integration
================================================

Core base classes, enums, and foundation models that serve as the building
blocks for all other data models in the Azure Universal RAG system.

This module provides:
- PydanticAI integration base classes
- Core enums for status, processing, and workflow states
- Base request/response models
- Foundation analysis result models
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, computed_field

# =============================================================================
# FOUNDATION BASE CLASS FOR PYDANTIC AI INTEGRATION
# =============================================================================


class PydanticAIContextualModel(BaseModel):
    """Base model with PydanticAI RunContext integration for agent communication"""

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    @computed_field
    @property
    def run_context_data(self) -> Optional[Dict[str, Any]]:
        """Extract relevant data for PydanticAI RunContext"""
        # Override in subclasses to provide context-specific data
        return None

    def to_run_context(self) -> Dict[str, Any]:
        """Convert model to RunContext-compatible dictionary"""
        context_data = self.run_context_data or {}
        return {
            "model_type": self.__class__.__name__,
            "model_data": self.model_dump(exclude={"run_context_data"}),
            **context_data,
        }

    def model_dump_json(self, **kwargs) -> str:
        """Override to exclude computed fields from JSON serialization"""
        kwargs.setdefault("exclude", set()).add("run_context_data")
        return super().model_dump_json(**kwargs)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override to exclude computed fields from dict serialization"""
        exclude = kwargs.get("exclude", set())
        if isinstance(exclude, set):
            exclude.add("run_context_data")
        elif isinstance(exclude, dict):
            exclude["run_context_data"] = True
        else:
            kwargs["exclude"] = {"run_context_data"}
        return super().model_dump(**kwargs)


# =============================================================================
# UNIFIED AGENT CONFIGURATION MODEL
# =============================================================================


class UnifiedAgentConfiguration(PydanticAIContextualModel):
    """
    Unified agent configuration model that consolidates all configuration patterns.
    Replaces fragmented configuration resolution with a single, predictable interface.
    """

    agent_type: str = Field(description="Type of agent this configuration is for")
    domain_name: str = Field(
        description="Domain name for domain-specific configuration"
    )
    configuration_data: Dict[str, Any] = Field(
        description="Resolved configuration parameters"
    )
    resolved_at: datetime = Field(description="When this configuration was resolved")
    resolver_context: Dict[str, Any] = Field(
        default_factory=dict, description="Context used for resolution"
    )

    # Performance and quality tracking
    resolution_time_ms: Optional[float] = Field(
        default=None, description="Time taken to resolve configuration"
    )
    configuration_source: str = Field(
        default="unified_resolver", description="Source of configuration"
    )
    cache_hit: bool = Field(
        default=False, description="Whether this was served from cache"
    )

    @computed_field
    @property
    def run_context_data(self) -> Dict[str, Any]:
        """Provide RunContext data for PydanticAI agent operations"""
        return {
            "agent_type": self.agent_type,
            "domain_name": self.domain_name,
            "configuration": self.configuration_data,
            "resolved_at": self.resolved_at.isoformat(),
            "resolver_context": self.resolver_context,
        }

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get a specific configuration parameter with default fallback"""
        return self.configuration_data.get(key, default)

    def update_parameter(self, key: str, value: Any, reason: str = None):
        """Update a configuration parameter with tracking"""
        old_value = self.configuration_data.get(key)
        self.configuration_data[key] = value

        # Log parameter change for audit trail
        if reason:
            change_log = self.resolver_context.setdefault("parameter_changes", [])
            change_log.append(
                {
                    "parameter": key,
                    "old_value": old_value,
                    "new_value": value,
                    "reason": reason,
                    "updated_at": datetime.now().isoformat(),
                }
            )

    def validate_required_parameters(self, required_params: List[str]) -> List[str]:
        """Validate that required parameters are present and return missing ones"""
        missing = []
        for param in required_params:
            if param not in self.configuration_data:
                missing.append(param)
        return missing

    def merge_configuration(self, additional_config: Dict[str, Any], source: str):
        """Merge additional configuration with tracking"""
        for key, value in additional_config.items():
            if key in self.configuration_data:
                # Track overrides
                override_log = self.resolver_context.setdefault("overrides", [])
                override_log.append(
                    {
                        "parameter": key,
                        "old_value": self.configuration_data[key],
                        "new_value": value,
                        "source": source,
                        "merged_at": datetime.now().isoformat(),
                    }
                )
            self.configuration_data[key] = value


# =============================================================================
# CORE ENUMS
# =============================================================================


class HealthStatus(str, Enum):
    """Standard health status classifications across all agents"""

    HEALTHY = "healthy"
    PARTIAL = "partial"
    DEGRADED = "degraded"
    ERROR = "error"
    NOT_INITIALIZED = "not_initialized"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    """Processing status for agent operations"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowState(str, Enum):
    """Workflow execution states"""

    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeState(str, Enum):
    """Individual node execution states"""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ErrorSeverity(str, Enum):
    """Error severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error category classifications"""

    AZURE_SERVICE = "azure_service"
    CONFIGURATION = "configuration"
    PROCESSING = "processing"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    NETWORK = "network"
    CACHE_MEMORY = "cache_memory"
    DOMAIN_INTELLIGENCE = "domain_intelligence"
    AGENT_PROCESSING = "agent_processing"
    UNKNOWN = "unknown"


class SearchType(str, Enum):
    """Supported search types for tri-modal search"""

    VECTOR = "vector"
    GRAPH = "graph"
    GNN = "gnn"
    ALL = "all"


class MessageType(str, Enum):
    """Graph communication message types"""

    QUERY = "query"
    RESPONSE = "response"
    STATUS = "status"
    ERROR = "error"


class ConfidenceMethod(str, Enum):
    """Confidence calculation methods for domain intelligence and knowledge extraction"""

    ADAPTIVE = "adaptive"
    STATISTICAL = "statistical"
    ENSEMBLE = "ensemble"


class StateTransferType(str, Enum):
    """Types of state transfers between workflow graphs"""

    CONFIG_GENERATION = "config_generation"
    DOMAIN_ANALYSIS = "domain_analysis"
    PATTERN_LEARNING = "pattern_learning"
    PERFORMANCE_METRICS = "performance_metrics"
    ERROR_CONTEXT = "error_context"


class ExtractionStatus(str, Enum):
    """Knowledge extraction status values"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"


# =============================================================================
# BASE REQUEST/RESPONSE MODELS
# =============================================================================


class BaseRequest(BaseModel):
    """Base request model with common fields"""

    request_id: Optional[str] = Field(
        default=None, description="Unique request identifier"
    )
    timestamp: Optional[datetime] = Field(
        default_factory=datetime.now, description="Request timestamp"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context"
    )

    class Config:
        use_enum_values = True
        extra = "forbid"
        str_strip_whitespace = True


class BaseResponse(BaseModel):
    """Base response model with common fields"""

    request_id: Optional[str] = Field(
        default=None, description="Matching request identifier"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Response timestamp"
    )
    processing_time_seconds: Optional[float] = Field(
        default=None, description="Processing time"
    )
    success: bool = Field(default=True, description="Operation success status")

    class Config:
        use_enum_values = True


class BaseAnalysisResult(BaseModel):
    """Base analysis result model"""

    confidence: float = Field(ge=0.0, le=1.0, description="Analysis confidence score")
    processing_time: float = Field(ge=0.0, description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


# =============================================================================
# CONFIGURATION RESOLVER (UTILITY CLASS)
# =============================================================================


class ConfigurationResolver:
    """
    Utility class for resolving unified agent configurations
    from multiple configuration sources
    """

    def __init__(self):
        self.resolution_cache = {}
        self.configuration_sources = []

    def register_source(self, source_name: str, source_func):
        """Register a configuration source function"""
        self.configuration_sources.append((source_name, source_func))

    def resolve_configuration(
        self,
        agent_type: str,
        domain_name: str,
        context: Dict[str, Any] = None,
        use_cache: bool = True,
    ) -> UnifiedAgentConfiguration:
        """Resolve configuration for an agent from all sources"""
        context = context or {}
        cache_key = f"{agent_type}:{domain_name}:{hash(str(sorted(context.items())))}"

        # Check cache first
        if use_cache and cache_key in self.resolution_cache:
            cached_config = self.resolution_cache[cache_key]
            cached_config.cache_hit = True
            return cached_config

        start_time = datetime.now()
        configuration_data = {}
        resolver_context = {"sources_used": [], "resolution_order": []}

        # Resolve from all sources in order
        for source_name, source_func in self.configuration_sources:
            try:
                source_config = source_func(agent_type, domain_name, context)
                if source_config:
                    configuration_data.update(source_config)
                    resolver_context["sources_used"].append(source_name)
                    resolver_context["resolution_order"].append(
                        {
                            "source": source_name,
                            "keys": list(source_config.keys()),
                            "resolved_at": datetime.now().isoformat(),
                        }
                    )
            except Exception as e:
                resolver_context.setdefault("errors", []).append(
                    {"source": source_name, "error": str(e)}
                )

        # Create unified configuration
        resolution_time = (datetime.now() - start_time).total_seconds() * 1000
        unified_config = UnifiedAgentConfiguration(
            agent_type=agent_type,
            domain_name=domain_name,
            configuration_data=configuration_data,
            resolved_at=datetime.now(),
            resolver_context=resolver_context,
            resolution_time_ms=resolution_time,
            configuration_source="unified_resolver",
            cache_hit=False,
        )

        # Cache the result
        if use_cache:
            self.resolution_cache[cache_key] = unified_config

        return unified_config

    def clear_cache(self):
        """Clear the resolution cache"""
        self.resolution_cache.clear()
