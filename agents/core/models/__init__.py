"""
Modular Data Models - Central Exports
=====================================

This module provides backward compatibility by re-exporting all data models
from the modular structure. This allows existing imports to continue working
while providing the benefits of a modular architecture.

The models are organized into logical modules:
- base: Foundation models and PydanticAI integration
- azure: Azure service models and configurations
- agents: Agent contracts and dependency models
- workflow: Workflow state and execution models
- search: Search request/response models
- extraction: Knowledge extraction models
- validation: Validation and error models
- cache: Cache and performance models
"""

# =============================================================================
# BASE MODELS AND ENUMS
# =============================================================================

from .agents import (  # Statistical and Domain Models; Agent Contracts; Configuration Models; Dependency Models; Service Container; Domain Intelligence Models
    AzureServicesDeps,
    DomainAnalysisContract,
    DomainAnalysisResult,
    DomainConfig,
    DomainIntelligenceConfig,
    DomainIntelligenceDeps,
    DomainStatistics,
    ExtractionConfiguration,
    GNNSearchConfig,
    GraphSearchConfig,
    KnowledgeExtractionContract,
    KnowledgeExtractionDeps,
    ServiceContainerConfig,
    StatisticalPattern,
    SynthesisWeights,
    UniversalSearchContract,
    UniversalSearchDeps,
    VectorSearchConfig,
)
from .azure import (  # Azure Service Configuration; ML Models; Graph Connection; Consolidated Models
    AzureCosmosGraphSchema,
    AzureMLModelMetadata,
    AzureSearchIndexSchema,
    AzureServiceConfiguration,
    AzureServiceMetrics,
    ConsolidatedAzureConfiguration,
    ConsolidatedAzureServices,
    GraphConnectionInfo,
    InfrastructureConfig,
    MLModelConfig,
)
from .base import (  # PydanticAI Integration; Core Enums; Base Models
    BaseAnalysisResult,
    BaseRequest,
    BaseResponse,
    ConfidenceMethod,
    ConfigurationResolver,
    ErrorCategory,
    ErrorSeverity,
    ExtractionStatus,
    HealthStatus,
    MessageType,
    NodeState,
    ProcessingStatus,
    PydanticAIContextualModel,
    SearchType,
    StateTransferType,
    UnifiedAgentConfiguration,
    WorkflowState,
)
from .cache import (
    CacheEntry,
    CacheMetrics,
    CachePerformanceMetrics,
    MemoryStatus,
)
from .cache import (
    PerformanceFeedbackPoint as CachePerformanceFeedbackPoint,  # Cache Models; Memory Management; Service Health; Performance Monitoring; Alias to avoid conflicts
)
from .cache import (
    ServiceHealth,
    SystemPerformanceSnapshot,
)
from .extraction import (  # Core Extraction; Knowledge Extraction (PydanticAI); Entity and Relationship Models; Text Analysis; Confidence Scoring; Content Preprocessing; PydanticAI Output Validation; Consolidated Configuration
    CleanedContent,
    ConfidenceScore,
    ConsolidatedExtractionConfiguration,
    ContentAnalysisOutput,
    ContentChunk,
    ContentChunker,
    EntityExtractionResult,
    ExtractedKnowledge,
    ExtractionContext,
    ExtractionQualityOutput,
    ExtractionResults,
    KnowledgeExtractionResult,
    KnowledgeValidationResult,
    RelationshipConfidenceFactors,
    RelationshipExtractionResult,
    TextCleaningOptions,
    TextStatistics,
    UnifiedExtractionResult,
    ValidatedEntity,
    ValidatedRelationship,
)
from .search import (  # Search Requests; Search Responses; Tri-Modal Results; Search Configuration
    AnalysisResult,
    ConsolidatedSearchConfiguration,
    DomainDetectionRequest,
    DomainDetectionResult,
    DynamicSearchConfig,
    GraphSearchRequest,
    ModalityResult,
    PatternLearningRequest,
    QueryRequest,
    SearchCoordinationResult,
    SearchResponse,
    SearchResult,
    TriModalResult,
    TriModalSearchRequest,
    TriModalSearchResult,
    VectorSearchRequest,
)
from .validation import (  # Validation Results; Error Handling; Performance Feedback; Quality Assurance
    ErrorContext,
    ErrorHandlingContract,
    ErrorMetrics,
    PerformanceFeedbackPoint,
    QualityGate,
    ValidationResult,
    ValidationResultPydanticAI,
    ValidationSummary,
)
from .workflow import (  # Workflow Execution; State Transfer; Background Processing; Statistical Analysis; Pattern Learning; Domain Intelligence
    BackgroundProcessingConfig,
    BackgroundProcessingResult,
    ExtractedPatterns,
    LearnedPattern,
    LLMExtraction,
    NodeExecutionResult,
    StateTransferPacket,
    StatisticalAnalysis,
    UnifiedAnalysis,
    WorkflowExecutionState,
    WorkflowResultContract,
    WorkflowStateBridge,
)

# =============================================================================
# AZURE SERVICE MODELS
# =============================================================================


# =============================================================================
# AGENT MODELS
# =============================================================================


# =============================================================================
# WORKFLOW MODELS
# =============================================================================


# =============================================================================
# SEARCH MODELS
# =============================================================================


# =============================================================================
# EXTRACTION MODELS
# =============================================================================


# =============================================================================
# VALIDATION AND ERROR MODELS
# =============================================================================


# =============================================================================
# CACHE AND PERFORMANCE MODELS
# =============================================================================


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Provide aliases for any models that might have been referenced differently
ValidationResultAI = ValidationResultPydanticAI
PydanticAIValidationResult = ValidationResultPydanticAI

# =============================================================================
# MODULE METADATA
# =============================================================================

__version__ = "2.0.0"
__author__ = "Azure Universal RAG Team"
__description__ = "Modular data models for the Azure Universal RAG system"

# List all exported models for introspection
__all__ = [
    # Base Models
    "PydanticAIContextualModel",
    "UnifiedAgentConfiguration",
    "ConfigurationResolver",
    "HealthStatus",
    "ProcessingStatus",
    "WorkflowState",
    "NodeState",
    "ErrorSeverity",
    "ErrorCategory",
    "SearchType",
    "MessageType",
    "ConfidenceMethod",
    "StateTransferType",
    "ExtractionStatus",
    "BaseRequest",
    "BaseResponse",
    "BaseAnalysisResult",
    # Azure Models
    "AzureServiceConfiguration",
    "AzureServiceMetrics",
    "AzureMLModelMetadata",
    "AzureSearchIndexSchema",
    "AzureCosmosGraphSchema",
    "MLModelConfig",
    "GraphConnectionInfo",
    "ConsolidatedAzureConfiguration",
    "ConsolidatedAzureServices",
    "InfrastructureConfig",
    # Agent Models
    "StatisticalPattern",
    "DomainStatistics",
    "DomainAnalysisContract",
    "KnowledgeExtractionContract",
    "UniversalSearchContract",
    "SynthesisWeights",
    "DomainConfig",
    "ExtractionConfiguration",
    "VectorSearchConfig",
    "GraphSearchConfig",
    "GNNSearchConfig",
    "AzureServicesDeps",
    "DomainIntelligenceDeps",
    "KnowledgeExtractionDeps",
    "UniversalSearchDeps",
    "ServiceContainerConfig",
    "DomainIntelligenceConfig",
    "DomainAnalysisResult",
    # Workflow Models
    "WorkflowExecutionState",
    "NodeExecutionResult",
    "WorkflowStateBridge",
    "WorkflowResultContract",
    "StateTransferPacket",
    "BackgroundProcessingConfig",
    "BackgroundProcessingResult",
    "StatisticalAnalysis",
    "LearnedPattern",
    "ExtractedPatterns",
    "UnifiedAnalysis",
    "LLMExtraction",
    # Search Models
    "QueryRequest",
    "VectorSearchRequest",
    "GraphSearchRequest",
    "TriModalSearchRequest",
    "DomainDetectionRequest",
    "PatternLearningRequest",
    "SearchResult",
    "SearchResponse",
    "DomainDetectionResult",
    "AnalysisResult",
    "TriModalSearchResult",
    "SearchCoordinationResult",
    "TriModalResult",
    "ModalityResult",
    "ConsolidatedSearchConfiguration",
    "DynamicSearchConfig",
    # Extraction Models
    "ExtractedKnowledge",
    "ExtractionResults",
    "UnifiedExtractionResult",
    "ExtractionContext",
    "KnowledgeExtractionResult",
    "KnowledgeValidationResult",
    "EntityExtractionResult",
    "RelationshipExtractionResult",
    "TextStatistics",
    "ConfidenceScore",
    "RelationshipConfidenceFactors",
    "TextCleaningOptions",
    "CleanedContent",
    "ContentChunker",
    "ContentChunk",
    "ExtractionQualityOutput",
    "ValidatedEntity",
    "ValidatedRelationship",
    "ContentAnalysisOutput",
    "ConsolidatedExtractionConfiguration",
    # Validation Models
    "ValidationResult",
    "ValidationResultPydanticAI",
    "ErrorHandlingContract",
    "ErrorContext",
    "ErrorMetrics",
    "PerformanceFeedbackPoint",
    "QualityGate",
    "ValidationSummary",
    # Cache Models
    "CacheEntry",
    "CacheMetrics",
    "CachePerformanceMetrics",
    "MemoryStatus",
    "ServiceHealth",
    "SystemPerformanceSnapshot",
    # Aliases
    "ValidationResultAI",
    "PydanticAIValidationResult",
]
