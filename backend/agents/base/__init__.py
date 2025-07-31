"""
Universal RAG Base Agent Architecture - Legacy Types and Constants

This module provides essential types and constants preserved from the legacy
agent architecture. Most functionality has been migrated to PydanticAI.

Preserved Components:
- Essential agent types (AgentContext, AgentCapability, etc.)
- Performance constants and configuration helpers
- Legacy interfaces for backward compatibility

Note: This module is in transition. Components are being migrated to PydanticAI
and will be removed once all dependencies are updated.
"""

# Import essential types that are still being used
from .agent_types import (
    AgentInterface,
    AgentContext,
    AgentResponse,
    AgentCapability,
    ReasoningStep,
    ReasoningTrace
)

# Import constants that are still valuable
from .constants import (
    REASONING,
    MEMORY,
    CONTEXT,
    REACT,
    TEMPORAL,
    PLAN_EXECUTE,
    PERFORMANCE,
    QUALITY,
    RETRY,
    VALIDATION,
    get_agent_base_config,
    get_performance_targets,
    get_quality_thresholds,
    validate_config
)

# Import new PydanticAI system components
from .performance_cache import (
    PerformanceCache,
    get_performance_cache,
    cached_operation
)

from .error_handling import (
    ErrorHandler,
    get_error_handler,
    resilient_operation,
    ErrorSeverity,
    ErrorCategory
)

from .tool_chaining import (
    ToolChainManager,
    get_tool_chain_manager,
    ToolChain,
    ToolStep,
    ChainExecutionMode
)

# Version info - marked as legacy
__version__ = "2.1.0-legacy"
__author__ = "Universal RAG Team"
__status__ = "Legacy - Migrating to PydanticAI"

# Export only essential components still in use
__all__ = [
    # Essential Agent Types (still referenced by discovery system)
    "AgentInterface",
    "AgentContext",
    "AgentResponse",
    "AgentCapability",
    "ReasoningStep",
    "ReasoningTrace",
    
    # Constants and Configuration (still valuable)
    "REASONING",
    "MEMORY",
    "CONTEXT", 
    "REACT",
    "TEMPORAL",
    "PLAN_EXECUTE",
    "PERFORMANCE",
    "QUALITY",
    "RETRY",
    "VALIDATION",
    "get_agent_base_config",
    "get_performance_targets", 
    "get_quality_thresholds",
    "validate_config",
    
    # New PydanticAI System Components
    "PerformanceCache",
    "get_performance_cache",
    "cached_operation",
    "ErrorHandler",
    "get_error_handler",
    "resilient_operation",
    "ErrorSeverity",
    "ErrorCategory",
    "ToolChainManager",
    "get_tool_chain_manager",
    "ToolChain",
    "ToolStep",
    "ChainExecutionMode"
]