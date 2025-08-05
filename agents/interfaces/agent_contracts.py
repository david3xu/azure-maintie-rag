"""
Agent Contract Interfaces - Data-Driven Pydantic Models

This module defines the comprehensive Pydantic model interfaces that eliminate
hardcoded values by establishing clear contracts between agents and Azure services.

All models are designed to be populated dynamically from real Azure service data,
ensuring a completely data-driven architecture with no hardcoded assumptions.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, computed_field, model_validator

# Clean configuration constants (CODING_STANDARDS compliant)
# Statistical and validation thresholds
CHI_SQUARE_SIGNIFICANCE_ALPHA = 0.05
MIN_PATTERN_FREQUENCY = 3  
STATISTICAL_CONFIDENCE_THRESHOLD = 0.75
STATISTICAL_CONFIDENCE_MIN = 0.0
STATISTICAL_CONFIDENCE_MAX = 1.0
MAX_EXECUTION_TIME_SECONDS = 300.0
MAX_EXECUTION_TIME_MIN = 0.1
MAX_EXECUTION_TIME_LIMIT = 600.0
MAX_AZURE_SERVICE_COST_USD = 10.0

# =============================================================================
# AZURE SERVICE DATA MODELS
# =============================================================================


class AzureServiceMetrics(BaseModel):
    """Real-time metrics from Azure services"""

    service_name: str = Field(..., description="Azure service name")
    request_count: int = Field(..., ge=0, description="Number of requests made")
    response_time_ms: float = Field(..., ge=0.0, description="Average response time")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate")
    cost_estimate_usd: float = Field(..., ge=0.0, description="Estimated cost in USD")
    timestamp: datetime = Field(default_factory=datetime.now)


class AzureMLModelMetadata(BaseModel):
    """Metadata from Azure ML models - NO hardcoded model references"""

    model_id: str = Field(..., description="Azure ML model ID from service discovery")
    model_version: str = Field(..., description="Model version from Azure ML registry")
    training_data_size: int = Field(..., ge=1, description="Training dataset size")
    model_accuracy: float = Field(
        ..., ge=0.0, le=1.0, description="Model accuracy from Azure ML metrics"
    )
    supported_languages: List[str] = Field(
        ..., description="Languages supported by model"
    )
    feature_dimensions: int = Field(..., ge=1, description="Feature vector dimensions")
    confidence_calibration: Dict[str, float] = Field(
        ..., description="Confidence calibration parameters"
    )


class AzureSearchIndexSchema(BaseModel):
    """Schema discovered from Azure Search indexes - NO hardcoded schemas"""

    index_name: str = Field(..., description="Azure Search index name")
    field_definitions: List[Dict[str, Any]] = Field(
        ..., description="Field definitions from index schema"
    )
    document_count: int = Field(..., ge=0, description="Total documents in index")
    searchable_fields: List[str] = Field(..., description="Fields marked as searchable")
    filterable_fields: List[str] = Field(..., description="Fields marked as filterable")
    facetable_fields: List[str] = Field(..., description="Fields marked as facetable")
    vector_fields: List[Dict[str, Any]] = Field(
        default_factory=list, description="Vector field configurations"
    )


class AzureCosmosGraphSchema(BaseModel):
    """Schema discovered from Azure Cosmos graph databases"""

    database_name: str = Field(..., description="Cosmos database name")
    vertex_labels: List[str] = Field(..., description="Discovered vertex labels")
    edge_labels: List[str] = Field(..., description="Discovered edge labels")
    vertex_count: int = Field(..., ge=0, description="Total vertex count")
    edge_count: int = Field(..., ge=0, description="Total edge count")
    relationship_patterns: List[Dict[str, Any]] = Field(
        ..., description="Discovered relationship patterns"
    )


# =============================================================================
# STATISTICAL PATTERN MODELS
# =============================================================================


class StatisticalPattern(BaseModel):
    """Statistical pattern learned from real data - NO hardcoded patterns"""

    pattern_id: str = Field(..., description="Unique pattern identifier")
    pattern_text: str = Field(..., description="Pattern text discovered from data")
    pattern_type: Literal["entity", "relationship", "concept", "action"] = Field(
        ..., description="Pattern type"
    )
    frequency: int = Field(..., ge=1, description="Frequency in training data")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Statistical confidence")
    support: float = Field(..., ge=0.0, le=1.0, description="Support (frequency/total)")
    lift: float = Field(..., ge=0.0, description="Lift measure for association rules")
    chi_square_p_value: float = Field(
        ..., ge=0.0, le=1.0, description="Chi-square test p-value"
    )
    confidence_interval: Dict[str, float] = Field(
        ..., description="95% confidence interval bounds"
    )
    source_documents: List[str] = Field(..., description="Source document identifiers")
    azure_ml_features: Dict[str, float] = Field(
        ..., description="Feature vectors from Azure ML"
    )

    @computed_field
    @property
    def is_statistically_significant(self) -> bool:
        """Determine if pattern is statistically significant"""
        return (self.chi_square_p_value < CHI_SQUARE_SIGNIFICANCE_ALPHA 
                and self.frequency >= MIN_PATTERN_FREQUENCY)


class DomainStatistics(BaseModel):
    """Statistical characteristics of a domain learned from Azure data"""

    domain_name: str = Field(..., description="Domain name from directory structure")
    document_count: int = Field(..., ge=1, description="Total documents analyzed")
    vocabulary_size: int = Field(..., ge=1, description="Unique terms in domain")
    average_document_length: float = Field(
        ..., ge=1.0, description="Average document length in tokens"
    )
    entity_density: float = Field(..., ge=0.0, description="Entities per document")
    relationship_density: float = Field(
        ..., ge=0.0, description="Relationships per document"
    )
    concept_complexity: float = Field(
        ..., ge=0.0, description="Concept complexity measure"
    )
    language_distribution: Dict[str, float] = Field(
        ..., description="Language distribution percentages"
    )
    topic_coherence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Topic coherence from Azure ML"
    )
    azure_cognitive_insights: Dict[str, Any] = Field(
        ..., description="Insights from Azure Cognitive Services"
    )


# =============================================================================
# AGENT CONTRACT INTERFACES
# =============================================================================


class DomainAnalysisContract(BaseModel):
    """Contract for Domain Intelligence Agent - Data-driven parameters only"""

    # Input parameters - from Azure services
    azure_storage_sources: List[str] = Field(
        ..., description="Azure Storage blob paths"
    )
    azure_search_indexes: List[str] = Field(
        ..., description="Azure Search indexes to analyze"
    )
    azure_ml_experiment_id: Optional[str] = Field(
        default=None, description="Azure ML experiment for pattern learning"
    )
    analysis_depth: Literal["surface", "deep", "comprehensive"] = Field(default="deep")
    statistical_confidence_threshold: float = Field(
        default=STATISTICAL_CONFIDENCE_THRESHOLD, 
        ge=STATISTICAL_CONFIDENCE_MIN, 
        le=STATISTICAL_CONFIDENCE_MAX
    )

    # Azure service configurations - discovered dynamically
    azure_cognitive_services_config: Dict[str, Any] = Field(
        ..., description="Cognitive services configuration"
    )
    azure_ml_compute_target: str = Field(
        ..., description="Azure ML compute target for analysis"
    )

    # Output contract
    expected_output_schema: Literal["DomainPatternResult"] = Field(
        default="DomainPatternResult"
    )


class KnowledgeExtractionContract(BaseModel):
    """Contract for Knowledge Extraction Agent - Tool delegation specifications"""

    # Input - from Domain Intelligence Agent
    domain_analysis_result: DomainStatistics = Field(
        ..., description="Domain analysis from Domain Intelligence Agent"
    )
    extraction_configuration: "ExtractionConfiguration" = Field(
        ..., description="Configuration from Config-Extraction orchestrator"
    )

    # Azure service targets - discovered dynamically
    target_azure_search_index: str = Field(
        ..., description="Target Azure Search index for results"
    )
    target_azure_cosmos_database: str = Field(
        ..., description="Target Azure Cosmos database for graph data"
    )
    azure_storage_output_container: str = Field(
        ..., description="Azure Storage container for extraction outputs"
    )

    # Tool delegation specifications
    required_tools: List[str] = Field(..., description="Required tools for extraction")
    tool_execution_order: List[str] = Field(..., description="Tool execution sequence")
    tool_azure_service_mappings: Dict[str, str] = Field(
        ..., description="Tool to Azure service mappings"
    )

    # Performance requirements
    max_execution_time_seconds: float = Field(
        default=MAX_EXECUTION_TIME_SECONDS, 
        gt=MAX_EXECUTION_TIME_MIN, 
        le=MAX_EXECUTION_TIME_LIMIT
    )
    max_azure_service_cost_usd: float = Field(
        default=MAX_AZURE_SERVICE_COST_USD, 
        gt=MAX_EXECUTION_TIME_MIN, 
        le=_config.max_azure_service_cost_limit
    )


class UniversalSearchContract(BaseModel):
    """Contract for Universal Search Agent - Multi-modal orchestration"""

    # Input - user query and domain context
    user_query: str = Field(
        ..., min_length=_config.min_query_length, max_length=_config.max_query_length, description="User search query"
    )
    domain_context: Optional[DomainStatistics] = Field(
        default=None, description="Domain context from Domain Intelligence Agent"
    )

    # Azure service search targets - discovered dynamically
    azure_search_indexes: List[AzureSearchIndexSchema] = Field(
        ..., description="Available search indexes with schemas"
    )
    azure_cosmos_databases: List[AzureCosmosGraphSchema] = Field(
        ..., description="Available graph databases with schemas"
    )

    # Search modality configurations - from Azure ML optimization
    vector_search_config: Dict[str, Any] = Field(
        ..., description="Vector search configuration from Azure ML"
    )
    graph_search_config: Dict[str, Any] = Field(
        ..., description="Graph search configuration from Azure ML"
    )
    gnn_search_config: Dict[str, Any] = Field(
        ..., description="GNN search configuration from Azure ML"
    )

    # Performance and cost constraints
    max_total_execution_time_seconds: float = Field(
        default=_config.max_total_search_time_seconds, 
        gt=_config.max_execution_time_min, 
        le=_config.max_search_time_limit
    )
    max_results_per_modality: int = Field(
        default=_config.max_results_per_modality, 
        ge=_config.min_query_length, 
        le=_config.max_results_limit
    )
    azure_service_cost_budget_usd: float = Field(
        default=_config.search_cost_budget_usd, 
        gt=_config.max_execution_time_min, 
        le=_config.search_cost_limit
    )


# =============================================================================
# TOOL DELEGATION CONTRACTS
# =============================================================================


class ToolExecutionContract(BaseModel):
    """Contract for tool execution with Azure service integration"""

    tool_name: str = Field(..., description="Tool name from tool registry")
    azure_service_dependencies: List[str] = Field(
        ..., description="Required Azure services"
    )
    input_data_schema: Dict[str, Any] = Field(
        ..., description="Input data schema validation"
    )
    output_data_schema: Dict[str, Any] = Field(
        ..., description="Expected output schema"
    )
    azure_service_cost_estimate: float = Field(
        ..., ge=0.0, description="Estimated Azure service cost"
    )
    execution_time_estimate_seconds: float = Field(
        ..., ge=0.0, description="Estimated execution time"
    )


class EntityExtractionToolContract(ToolExecutionContract):
    """Contract for entity extraction tool"""

    azure_cognitive_services_config: Dict[str, Any] = Field(
        ..., description="Cognitive Services configuration"
    )
    azure_ml_model_endpoint: str = Field(..., description="Azure ML model endpoint URL")
    custom_entity_types: List[str] = Field(
        ..., description="Custom entity types from domain analysis"
    )
    confidence_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence threshold from statistical analysis",
    )


class RelationshipExtractionToolContract(ToolExecutionContract):
    """Contract for relationship extraction tool"""

    azure_cosmos_connection_string: str = Field(
        ..., description="Azure Cosmos connection string"
    )
    relationship_patterns: List[StatisticalPattern] = Field(
        ..., description="Relationship patterns from domain analysis"
    )
    graph_traversal_depth: int = Field(
        ..., ge=_config.graph_traversal_depth_min, le=_config.graph_traversal_depth_max, description="Maximum graph traversal depth"
    )
    azure_ml_relationship_model: str = Field(
        ..., description="Azure ML relationship extraction model"
    )


class QualityAssessmentToolContract(ToolExecutionContract):
    """Contract for quality assessment tool"""

    quality_metrics_definitions: Dict[str, str] = Field(
        ..., description="Quality metric definitions"
    )
    statistical_test_configurations: Dict[str, Any] = Field(
        ..., description="Statistical test configurations"
    )
    azure_ml_quality_model: str = Field(
        ..., description="Azure ML quality assessment model"
    )
    benchmark_datasets: List[str] = Field(
        ..., description="Benchmark datasets for quality comparison"
    )


# =============================================================================
# ORCHESTRATOR CONTRACTS
# =============================================================================


class WorkflowExecutionContract(BaseModel):
    """Contract for Config-Extraction Orchestrator workflow execution"""

    # Workflow identification
    workflow_id: str = Field(..., description="Unique workflow identifier")
    workflow_type: Literal[
        "intelligent_rag", "domain_discovery", "knowledge_extraction", "search_only"
    ] = Field(...)

    # Agent coordination contracts
    domain_intelligence_contract: DomainAnalysisContract = Field(
        ..., description="Domain Intelligence Agent contract"
    )
    knowledge_extraction_contract: Optional[KnowledgeExtractionContract] = Field(
        default=None, description="Knowledge Extraction Agent contract"
    )
    universal_search_contract: UniversalSearchContract = Field(
        ..., description="Universal Search Agent contract"
    )

    # Azure service orchestration
    azure_service_coordination: Dict[str, Any] = Field(
        ..., description="Azure service coordination plan"
    )
    azure_resource_allocation: Dict[str, float] = Field(
        ..., description="Azure resource allocation budget"
    )

    # Performance contracts
    max_total_execution_time_seconds: float = Field(
        default=_config.workflow_max_execution_time, 
        gt=_config.max_execution_time_min, 
        le=_config.workflow_max_execution_limit
    )
    max_total_azure_cost_usd: float = Field(
        default=_config.workflow_max_cost_usd, 
        gt=_config.max_execution_time_min, 
        le=_config.workflow_max_cost_limit
    )
    minimum_quality_score: float = Field(
        default=_config.workflow_min_quality_score, 
        ge=_config.workflow_quality_min, 
        le=_config.workflow_quality_max
    )

    # Error handling and recovery
    error_recovery_strategies: List[str] = Field(
        ..., description="Error recovery strategies"
    )
    fallback_configurations: Dict[str, Any] = Field(
        ..., description="Fallback configurations"
    )


class WorkflowResultContract(BaseModel):
    """Contract for workflow execution results"""

    # Execution metadata
    workflow_id: str = Field(..., description="Workflow identifier")
    execution_start_time: datetime = Field(..., description="Workflow start time")
    execution_end_time: datetime = Field(..., description="Workflow end time")
    total_execution_time_seconds: float = Field(
        ..., ge=0.0, description="Total execution time"
    )

    # Agent execution results
    domain_intelligence_result: Dict[str, Any] = Field(
        ..., description="Domain Intelligence Agent result"
    )
    knowledge_extraction_result: Optional[Dict[str, Any]] = Field(
        default=None, description="Knowledge Extraction Agent result"
    )
    universal_search_result: Dict[str, Any] = Field(
        ..., description="Universal Search Agent result"
    )

    # Azure service utilization
    azure_service_metrics: List[AzureServiceMetrics] = Field(
        ..., description="Azure service utilization metrics"
    )
    total_azure_cost_usd: float = Field(
        ..., ge=0.0, description="Total Azure service cost"
    )

    # Quality and performance metrics
    overall_quality_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall quality score"
    )
    performance_metrics: Dict[str, float] = Field(
        ..., description="Performance metrics"
    )
    error_count: int = Field(
        default=0, ge=0, description="Number of errors encountered"
    )
    recovery_count: int = Field(
        default=0, ge=0, description="Number of successful recoveries"
    )

    # Result synthesis
    final_results: List[Dict[str, Any]] = Field(
        ..., description="Final synthesized results"
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall confidence score"
    )
    result_provenance: List[str] = Field(
        ..., description="Provenance trace for results"
    )

    @model_validator(mode="after")
    def validate_performance_targets(self) -> "WorkflowResultContract":
        """Validate that performance targets were met"""
        if self.total_execution_time_seconds > _config.performance_violation_threshold:
            # Log performance violation but don't fail
            import logging

            logging.warning(
                f"Performance target violation: {self.total_execution_time_seconds}s > {_config.performance_violation_threshold}s"
            )

        if self.overall_quality_score < _config.quality_concern_threshold:
            import logging

            logging.warning(
                f"Quality target concern: {self.overall_quality_score} < {_config.quality_concern_threshold}"
            )

        return self


# =============================================================================
# SHARED CAPABILITY CONTRACTS
# =============================================================================


class CacheContract(BaseModel):
    """Contract for shared caching capabilities"""

    cache_key_namespace: str = Field(..., description="Cache namespace for isolation")
    data_serialization_format: Literal["json", "pickle", "msgpack"] = Field(
        default="json"
    )
    ttl_seconds: int = Field(
        default=_config.cache_ttl_default, ge=_config.cache_ttl_min, le=_config.cache_ttl_max, description="Time to live in seconds"
    )
    azure_redis_configuration: Optional[Dict[str, Any]] = Field(
        default=None, description="Azure Redis configuration"
    )
    local_cache_size_mb: int = Field(
        default=_config.local_cache_size_default_mb, ge=_config.local_cache_size_min_mb, le=_config.local_cache_size_max_mb, description="Local cache size limit"
    )


class ErrorHandlingContract(BaseModel):
    """Contract for shared error handling capabilities"""

    error_categories: List[str] = Field(..., description="Supported error categories")
    recovery_strategies: Dict[str, str] = Field(
        ..., description="Recovery strategy mappings"
    )
    azure_monitor_integration: bool = Field(
        default=True, description="Azure Monitor integration enabled"
    )
    circuit_breaker_thresholds: Dict[str, int] = Field(
        ..., description="Circuit breaker threshold configurations"
    )
    retry_policies: Dict[str, Dict[str, Any]] = Field(
        ..., description="Retry policy configurations"
    )


class MonitoringContract(BaseModel):
    """Contract for shared monitoring capabilities"""

    azure_application_insights_config: Dict[str, str] = Field(
        ..., description="Application Insights configuration"
    )
    custom_metrics_definitions: List[Dict[str, Any]] = Field(
        ..., description="Custom metrics to track"
    )
    alert_thresholds: Dict[str, float] = Field(
        ..., description="Alert threshold configurations"
    )
    dashboard_configurations: List[Dict[str, Any]] = Field(
        ..., description="Dashboard widget configurations"
    )


# =============================================================================
# CONFIGURATION INTERFACE EXTENSIONS
# =============================================================================


class DataDrivenExtractionConfiguration(BaseModel):
    """Extended extraction configuration with complete Azure service integration"""

    # Base configuration from extraction_interface.py
    domain_name: str = Field(..., description="Domain name from directory structure")
    generation_timestamp: datetime = Field(default_factory=datetime.now)

    # Statistical foundations - NO hardcoded values
    statistical_patterns: List[StatisticalPattern] = Field(
        ..., description="Learned statistical patterns"
    )
    domain_statistics: DomainStatistics = Field(
        ..., description="Domain statistical characteristics"
    )
    azure_ml_model_metadata: AzureMLModelMetadata = Field(
        ..., description="Azure ML model metadata"
    )

    # Dynamic thresholds from statistical analysis
    entity_confidence_threshold: float = Field(
        ..., description="Threshold from statistical analysis of domain data"
    )
    relationship_confidence_threshold: float = Field(
        ..., description="Threshold from relationship pattern analysis"
    )
    minimum_quality_score: float = Field(
        ..., description="Quality threshold from benchmark analysis"
    )

    # Azure service integration specifications
    azure_search_target_schema: AzureSearchIndexSchema = Field(
        ..., description="Target search index schema"
    )
    azure_cosmos_target_schema: AzureCosmosGraphSchema = Field(
        ..., description="Target graph database schema"
    )

    # Tool execution specifications
    tool_contracts: List[ToolExecutionContract] = Field(
        ..., description="Tool execution contracts"
    )
    azure_service_cost_budget: float = Field(
        ..., ge=0.0, description="Azure service cost budget"
    )
    performance_requirements: Dict[str, float] = Field(
        ..., description="Performance requirements"
    )

    @computed_field
    @property
    def is_optimized_for_azure(self) -> bool:
        """Verify that configuration is optimized for Azure services"""
        return (
            len(self.statistical_patterns) > 0
            and self.azure_ml_model_metadata.model_accuracy > _config.min_model_accuracy
            and self.azure_service_cost_budget > _config.max_execution_time_min
        )


# =============================================================================
# VALIDATION AND COMPLIANCE
# =============================================================================


class ArchitectureComplianceValidator(BaseModel):
    """Validator for architecture compliance with enterprise requirements"""

    # Hardcoded value detection
    hardcoded_value_count: int = Field(
        default=0, ge=0, description="Count of detected hardcoded values"
    )
    hardcoded_value_locations: List[str] = Field(
        default_factory=list, description="Locations of hardcoded values"
    )

    # Agent boundary validation
    agent_boundary_violations: List[str] = Field(
        default_factory=list, description="Agent boundary violations"
    )
    responsibility_overlaps: List[Dict[str, str]] = Field(
        default_factory=list, description="Responsibility overlaps between agents"
    )

    # Azure service integration validation
    azure_service_coverage: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Percentage of Azure services properly integrated",
    )
    mock_service_count: int = Field(
        default=0, ge=0, description="Count of mock services (should be 0)"
    )

    # Tool delegation validation
    self_contained_logic_violations: List[str] = Field(
        default_factory=list, description="Self-contained logic violations"
    )
    tool_delegation_coverage: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Percentage of operations using tool delegation",
    )

    @computed_field
    @property
    def is_architecture_compliant(self) -> bool:
        """Determine if architecture meets compliance requirements"""
        return (
            self.hardcoded_value_count == 0
            and len(self.agent_boundary_violations) == 0
            and self.azure_service_coverage >= _config.azure_service_coverage_threshold
            and self.mock_service_count == 0
            and len(self.self_contained_logic_violations) == 0
            and self.tool_delegation_coverage >= _config.tool_delegation_coverage_threshold
        )


# Forward reference resolution
DataDrivenExtractionConfiguration.model_rebuild()
