"""
Agent Contract Interfaces - Centralized Data Models

This module provides access to centralized agent contract models that eliminate
hardcoded values by establishing clear contracts between agents and Azure services.

All models are now centralized in agents.core.data_models to maintain the 
zero-hardcoded-values philosophy and single source of truth.
"""

# Import all agent contract models from centralized location
from agents.core.data_models import (
    # Azure Service Models
    AzureServiceMetrics,
    AzureMLModelMetadata,
    AzureSearchIndexSchema,
    AzureCosmosGraphSchema,
    
    # Statistical and Analysis Models
    StatisticalPattern,
    DomainStatistics,
    
    # Legacy Agent Contract Models (deprecated - use Enhanced versions)
    DomainAnalysisContract,
    KnowledgeExtractionContract,
    UniversalSearchContract,
    
    # Enhanced Agent Contract Models DELETED - redundant with basic contracts
    
    # Consolidated Configuration Models (RECOMMENDED)
    ConsolidatedAzureConfiguration,
    ConsolidatedExtractionConfiguration,
    ConsolidatedSearchConfiguration,
    
    # Foundation Models
    ConfigurationResolver,
    PydanticAIContextualModel,
    
    # Workflow Models
    WorkflowResultContract,
    
    # Infrastructure Models
    ErrorHandlingContract,
    # MonitoringContract deleted in Phase 9 - over-engineered monitoring with zero usage
    
    # Data-Driven Configuration Models (deleted in Phase 3 - were unused)
)

# Import constants for backward compatibility
from agents.core.constants import (
    StatisticalConstants,
    ProcessingConstants
)

# Constants for backward compatibility
CHI_SQUARE_SIGNIFICANCE_ALPHA = StatisticalConstants.CHI_SQUARE_SIGNIFICANCE_ALPHA
MIN_PATTERN_FREQUENCY = StatisticalConstants.MIN_PATTERN_FREQUENCY
STATISTICAL_CONFIDENCE_THRESHOLD = StatisticalConstants.STATISTICAL_CONFIDENCE_THRESHOLD
STATISTICAL_CONFIDENCE_MIN = StatisticalConstants.STATISTICAL_CONFIDENCE_MIN
STATISTICAL_CONFIDENCE_MAX = StatisticalConstants.STATISTICAL_CONFIDENCE_MAX
MAX_EXECUTION_TIME_SECONDS = ProcessingConstants.MAX_EXECUTION_TIME_SECONDS
MAX_EXECUTION_TIME_MIN = ProcessingConstants.MAX_EXECUTION_TIME_MIN
MAX_EXECUTION_TIME_LIMIT = ProcessingConstants.MAX_EXECUTION_TIME_LIMIT
MAX_AZURE_SERVICE_COST_USD = ProcessingConstants.MAX_AZURE_SERVICE_COST_USD

# Export all models for backward compatibility
__all__ = [
    # Azure Service Models
    "AzureServiceMetrics",
    "AzureMLModelMetadata", 
    "AzureSearchIndexSchema",
    "AzureCosmosGraphSchema",
    
    # Statistical and Analysis Models
    "StatisticalPattern",
    "DomainStatistics",
    
    # Agent Contract Models
    "DomainAnalysisContract",
    "KnowledgeExtractionContract",
    "UniversalSearchContract",
    
    # Workflow Models
    "WorkflowResultContract",
    
    # Infrastructure Models
    "ErrorHandlingContract",
    # "MonitoringContract" deleted in Phase 9
    
    # Data-Driven Configuration Models (deleted in Phase 3 - were unused)
]