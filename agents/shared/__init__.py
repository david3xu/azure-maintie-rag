"""
Shared utilities and patterns for agent implementations.

This module provides common utilities, patterns, and tools that are shared
across different agent implementations to promote code reuse and consistency.
"""

# Import shared utilities and moved core utilities
from .capability_patterns import (
    CapabilityFactory,
    CacheCapability,
    MonitoringCapability,
)
from .common_tools import (
    generate_cache_key,
    calculate_confidence_score,
    format_processing_time,
)
from .toolsets import AzureServiceToolset, PerformanceToolset
from .graph_communication import MessageType

# Utilities moved from agents.core for better organization
from .memory_manager import UnifiedMemoryManager, get_memory_manager
from .config_enforcement import (
    AntiHardcodingEnforcer,
    validate_config,
    get_enforcement_report,
)
from .intelligent_config_provider import (
    IntelligentConfigProvider,
    get_intelligent_config,
)
from .workflow_state_bridge import WorkflowStateBridge

__all__ = [
    # Capability patterns
    "CapabilityFactory",
    "CacheCapability",
    "MonitoringCapability",
    # Common tools
    "generate_cache_key",
    "calculate_confidence_score",
    "format_processing_time",
    # Shared toolsets
    "AzureServiceToolset",
    "PerformanceToolset",
    # Graph communication models (migrated from supports/)
    "MessageType",
    # GraphMessage, GraphStatus - deleted in Phase 1
    # Utilities moved from core/ for better organization
    "UnifiedMemoryManager",
    "get_memory_manager",
    "AntiHardcodingEnforcer",
    "validate_config",
    "get_enforcement_report",
    "IntelligentConfigProvider",
    "get_intelligent_config",
    "WorkflowStateBridge",
]
