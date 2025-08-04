"""
🎯 Centralized Configuration System
============================================================

This module provides a single source of truth for all configuration values
previously hardcoded throughout the agent system. Follows the "Data-Driven
Everything" principle by externalizing all magic numbers, thresholds, and
entity lists to configurable parameters.

Architecture:
- ConfigurationManager: Central registry for all config values
- ConfigSection: Typed configuration sections for different domains
- load_from_env(): Environment variable override support
- validate_config(): Configuration validation and defaults
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
import json


@dataclass
class CacheConfiguration:
    """Cache-related configuration parameters"""
    default_ttl_seconds: int = 3600  # 1 hour
    redis_ttl_seconds: int = 300     # 5 minutes
    hit_rate_threshold_excellent: int = 80
    hit_rate_threshold_good: int = 60
    max_entries_per_namespace: int = 10000
    cleanup_interval_hours: int = 24


@dataclass
class ConfidenceConfiguration:
    """Confidence and statistical threshold parameters"""
    default_confidence_level: float = 0.95
    minimum_pattern_confidence: float = 0.6
    entity_confidence_threshold: float = 0.7
    relationship_confidence_threshold: float = 0.65
    statistical_significance_alpha: float = 0.05
    high_confidence_threshold: float = 0.8
    medium_confidence_threshold: float = 0.5
    low_confidence_threshold: float = 0.2


@dataclass
class ProcessingConfiguration:
    """Processing and performance thresholds"""
    max_workers: int = 4
    max_concurrent_chunks: int = 5
    chunk_size_default: int = 1000
    chunk_overlap_ratio: float = 0.2
    max_features_vectorizer: int = 1000
    min_document_frequency: float = 0.01
    max_document_frequency: float = 0.95
    timeout_base_seconds: int = 30
    cache_ttl_seconds: int = 3600  # 1 hour default


@dataclass
class EntityExtractionConfiguration:
    """Entity extraction and classification parameters"""
    min_entity_length: int = 2
    max_entities_per_chunk: int = 15
    frequency_threshold: int = 5
    top_tokens_limit: int = 100
    code_elements_limit: int = 10
    api_interfaces_limit: int = 10
    data_structures_limit: int = 10
    general_concepts_limit: int = 15


@dataclass
class DomainAnalysisConfiguration:
    """Domain analysis and classification thresholds"""
    high_diversity_threshold: float = 0.7
    medium_diversity_threshold: float = 0.3
    high_technical_density: float = 0.3
    high_complexity_threshold: float = 0.7
    high_vocabulary_richness: float = 0.5
    long_document_threshold: int = 2000
    medium_document_threshold: int = 800
    concept_frequency_threshold: int = 50


@dataclass
class MLConfiguration:
    """Machine learning and clustering parameters"""
    kmeans_clusters: int = 5
    random_state: int = 42
    n_init_kmeans: int = 10
    learning_rate_uncertainty_factor: float = 0.5
    pattern_frequency_minimum: int = 3
    overlap_threshold: float = 0.3
    entity_boost_factor: float = 1.5


@dataclass
class ValidationConfiguration:
    """Validation and quality assessment parameters"""
    # Quality thresholds
    min_entity_quality_score: float = 0.7
    min_relationship_quality_score: float = 0.6
    min_coverage_score: float = 0.5
    min_consistency_score: float = 0.8
    min_overall_quality: float = 0.6
    
    # Entity validation
    min_entity_confidence: float = 0.3
    max_entity_confidence: float = 1.0
    entity_frequency_threshold: int = 2
    min_entity_length: int = 1
    max_entity_length: int = 100
    
    # Relationship validation  
    min_relationship_confidence: float = 0.2
    max_relationship_confidence: float = 1.0
    relationship_frequency_threshold: int = 1
    max_relationships_per_entity: int = 50
    
    # Statistical validation
    confidence_interval_alpha: float = 0.05
    outlier_detection_threshold: float = 2.0
    statistical_significance_level: float = 0.05
    sample_size_minimum: int = 10
    
    # Performance validation
    max_processing_time_seconds: float = 30.0
    min_extraction_rate_per_second: float = 1.0
    max_memory_usage_mb: float = 1000.0
    
    # Graph validation
    min_graph_connectivity: float = 0.1
    max_isolated_nodes_ratio: float = 0.3
    min_average_degree: float = 1.0
    max_graph_diameter: int = 10
    
    # Anomaly detection
    anomaly_score_threshold: float = 0.8
    max_anomaly_ratio: float = 0.1
    confidence_deviation_threshold: float = 0.3


@dataclass
class RelationshipProcessingConfiguration:
    """Relationship extraction and processing parameters"""
    # Confidence thresholds
    high_confidence_threshold: float = 0.8
    base_syntactic_confidence: float = 0.6
    base_semantic_confidence: float = 0.7
    base_pattern_confidence: float = 0.8
    high_semantic_confidence: float = 0.8
    context_factor_default: float = 0.7
    context_factor_high: float = 0.9
    
    # Distance and proximity factors
    distance_divisor: float = 100.0
    min_distance_factor: float = 0.3
    max_distance_factor: float = 1.0
    context_window_size: int = 100
    context_window_small: int = 50
    
    # Length and prominence factors
    max_sentence_length_divisor: float = 50.0
    min_length_factor: float = 0.5
    max_prominence_divisor: float = 4.0
    max_pattern_parts_divisor: float = 5.0
    
    # Confidence weighting factors
    syntactic_base_weight: float = 0.4
    syntactic_distance_weight: float = 0.3
    syntactic_context_weight: float = 0.3
    semantic_base_weight: float = 0.5
    semantic_length_weight: float = 0.3
    semantic_prominence_weight: float = 0.2
    pattern_base_weight: float = 0.6
    pattern_specificity_weight: float = 0.2
    pattern_context_weight: float = 0.2
    
    # Pattern matching
    min_pattern_parts: int = 3
    min_regex_groups: int = 2
    
    # Performance tracking
    avg_time_initial: float = 0.0
    avg_relationships_initial: float = 0.0
    extraction_count_initial: int = 0
    
    # Validation and quality
    min_unique_entity_pairs: int = 1
    max_confidence_value: float = 1.0
    default_connected_components: int = 0
    default_graph_density: float = 0.0


@dataclass
class EntityProcessingConfiguration:
    """Entity extraction and processing parameters"""
    # Confidence thresholds
    high_confidence_threshold: float = 0.8
    high_technical_confidence: float = 0.8
    base_nlp_confidence: float = 0.6
    length_bonus_small: float = 0.1
    length_bonus_large: float = 0.1
    frequency_bonus: float = 0.1
    frequency_bonus_small: float = 0.05
    
    # Position and context factors
    early_position_factor: float = 0.8
    late_position_factor: float = 0.6
    position_boundary_divisor: float = 4.0
    context_window_size: int = 50
    context_window_small: int = 30
    context_factor_default: float = 0.7
    context_factor_high: float = 0.9
    case_factor_default: float = 0.8
    case_factor_high: float = 0.9
    
    # Text analysis factors
    text_length_divisor: float = 20.0
    max_confidence_value: float = 1.0
    confidence_boost_factor: float = 1.2
    min_entity_length: int = 3
    long_entity_threshold: int = 10
    caps_min_length: int = 2
    
    # Weighting factors
    length_weight: float = 0.3
    position_weight: float = 0.2
    context_weight: float = 0.3
    case_weight: float = 0.2
    
    # Pattern matching
    technical_vocab_limit: int = 50
    caps_pattern_min_length: int = 2
    
    # Frequency analysis
    single_frequency: int = 1
    low_frequency_threshold: int = 3
    
    # Performance tracking
    avg_time_initial: float = 0.0
    avg_entities_initial: float = 0.0
    extraction_count_initial: int = 0
    
    # Validation constraints
    min_entities_default: int = 0
    coverage_percentage_initial: float = 0.0
    missing_types_ratio_threshold: float = 0.5
    missing_types_display_limit: int = 5
    
    # Confidence distribution bins
    confidence_very_high_threshold: float = 0.9


@dataclass
class CapabilityPatternsConfiguration:
    """Shared capability patterns configuration"""
    # Cache configuration
    default_ttl_seconds: int = 3600
    local_cache_ttl_seconds: int = 300  # 5 minutes
    max_local_cache_minutes: int = 5
    cache_stats_initial: int = 0
    cache_hit_rate_excellent: int = 80
    cache_hit_rate_good: int = 60
    max_cache_requests: int = 1
    cache_invalidated_initial: int = 0
    
    # Statistical analysis configuration
    confidence_level_default: float = 0.95
    min_sample_size: int = 2
    default_interval_bounds: float = 0.0
    significance_threshold: float = 0.05
    min_pattern_support: float = 0.05  # 5%
    min_pattern_confidence: float = 0.7  # 70%
    default_lift: float = 1.0
    default_chi_square_p: float = 0.05
    pattern_frequency_min: int = 3
    pattern_confidence_min: float = 0.6
    effect_size_default: float = 0.0
    degrees_freedom_offset: int = 1
    
    # Performance monitoring
    default_time_window_hours: int = 24
    performance_stats_initial: int = 0
    avg_time_initial: float = 0.0
    
    # Azure service optimization
    ml_request_threshold: int = 100
    ml_savings_percent: int = 25
    search_query_threshold: int = 1000
    search_savings_percent: int = 30
    cosmos_ru_threshold: int = 10000
    cosmos_savings_percent: int = 20
    
    # Cost estimation
    base_cost_ml: float = 0.1
    base_cost_search: float = 0.01
    base_cost_cosmos: float = 0.005
    base_cost_default: float = 0.01
    compute_cost_factor: float = 0.05
    
    # Search optimization
    max_search_results: int = 50
    
    # Performance tracking multipliers
    percentage_multiplier: int = 100


@dataclass
class ModelConfiguration:
    """Model deployment and API configuration"""
    # Azure OpenAI deployment names
    gpt4o_deployment_name: str = "gpt-4o"
    gpt4o_mini_deployment_name: str = "gpt-4o-mini"
    text_embedding_deployment_name: str = "text-embedding-ada-002"
    
    # API versions
    openai_api_version: str = "2024-08-01-preview"
    search_api_version: str = "2023-11-01"
    cosmos_api_version: str = "2023-09-15"
    
    # Model parameters
    default_temperature: float = 0.0
    default_max_tokens: int = 4000
    default_top_p: float = 1.0
    default_frequency_penalty: float = 0.0
    default_presence_penalty: float = 0.0


@dataclass
class InfrastructureConfiguration:
    """Infrastructure timeouts, retries, and connection parameters"""
    # Connection timeouts (seconds)
    openai_timeout: int = 60
    search_timeout: int = 30
    cosmos_timeout: int = 45
    storage_timeout: int = 120
    ml_timeout: int = 300
    
    # Retry configuration
    max_retries: int = 3
    retry_backoff_factor: float = 2.0
    retry_base_delay: float = 1.0
    
    # Connection pool settings
    max_pool_connections: int = 20
    max_keepalive_connections: int = 10
    keepalive_expiry: int = 5
    
    # Batch processing limits
    max_batch_size: int = 10
    batch_timeout: int = 30
    max_concurrent_requests: int = 5


@dataclass
class ServiceEndpointConfiguration:
    """Azure service endpoint configuration"""
    # Base endpoint patterns (to be populated from environment)
    openai_endpoint_pattern: str = "https://{resource_name}.openai.azure.com/"
    search_endpoint_pattern: str = "https://{resource_name}.search.windows.net"
    cosmos_endpoint_pattern: str = "https://{resource_name}.gremlin.cosmos.azure.com:443/"
    storage_endpoint_pattern: str = "https://{resource_name}.blob.core.windows.net/"
    ml_endpoint_pattern: str = "https://{resource_name}.{region}.inference.ml.azure.com"
    
    # Default resource names (overridden by environment)
    default_openai_resource: str = "maintie-rag-openai"
    default_search_resource: str = "maintie-rag-search"
    default_cosmos_resource: str = "maintie-rag-cosmos"
    default_storage_resource: str = "maintie-rag-storage"
    default_ml_resource: str = "maintie-rag-ml"


@dataclass
class DomainIntelligenceDecisionConfiguration:
    """Domain intelligence decision tree thresholds and parameters"""
    # Vocabulary diversity thresholds (from toolsets.py:288-293)
    vocabulary_diversity_high_threshold: float = 0.7
    vocabulary_diversity_medium_threshold: float = 0.3
    base_threshold_high: float = 0.8
    base_threshold_medium: float = 0.7
    base_threshold_low: float = 0.6
    
    # Complexity score thresholds for SLA estimation
    complexity_high_threshold: float = 1.5
    complexity_medium_threshold: float = 0.8
    sla_high_complexity: float = 5.0
    sla_medium_complexity: float = 3.5
    sla_low_complexity: float = 2.5
    
    # Document length processing ratios
    long_doc_ratio: float = 0.4
    long_doc_max: int = 1500
    medium_doc_ratio: float = 0.6
    medium_doc_max: int = 1200
    short_doc_ratio: float = 0.8
    short_doc_min: int = 400
    short_doc_max: int = 800
    
    # Document length classification thresholds
    long_document_threshold: int = 2000
    medium_document_threshold: int = 800
    
    # Entity density thresholds
    high_entity_density_threshold: int = 20
    low_entity_density_threshold: int = 5
    
    # Technical content indicators
    technical_density_threshold: float = 0.3
    vocabulary_richness_threshold: float = 0.5
    complexity_score_threshold: float = 0.7


@dataclass
class QualityAssessmentConfiguration:
    """Quality assessment weights and thresholds for knowledge extraction"""
    # Quality score weights (from knowledge_extraction processors)
    entity_quality_weight: float = 0.4
    relationship_quality_weight: float = 0.3
    coverage_score_weight: float = 0.2
    consistency_score_weight: float = 0.1
    
    # Quality thresholds
    entity_quality_threshold: float = 0.7
    relationship_quality_threshold: float = 0.6
    coverage_score_threshold: float = 0.5
    minimum_overall_quality_score: float = 0.6
    
    # Default configuration values (should come from domain intelligence)
    default_entity_confidence_threshold: float = 0.7
    default_relationship_confidence_threshold: float = 0.65
    default_minimum_quality_score: float = 0.6
    
    # Processing optimization parameters
    precision_improvement_factor: float = 0.15
    recall_improvement_factor: float = 0.12
    confidence_boost_factor: float = 0.85


@dataclass
class ConfidenceCalculationConfiguration:
    """Confidence calculation formulas and coefficients"""
    # Entity confidence calculation weights
    entity_length_weight: float = 0.3
    entity_position_weight: float = 0.2
    entity_context_weight: float = 0.3
    entity_case_weight: float = 0.2
    
    # Relationship confidence calculation weights
    relationship_base_weight: float = 0.4
    relationship_distance_weight: float = 0.3
    relationship_context_weight: float = 0.3
    
    # Alternative relationship weighting
    relationship_distance_factor_weight: float = 0.7
    relationship_frequency_factor_weight: float = 0.3
    
    # Syntactic confidence parameters
    syntactic_base_weight: float = 0.4
    syntactic_distance_weight: float = 0.3
    syntactic_context_weight: float = 0.3
    
    # Semantic confidence parameters
    semantic_base_weight: float = 0.5
    semantic_length_weight: float = 0.3
    semantic_prominence_weight: float = 0.2
    
    # Pattern confidence parameters
    pattern_base_weight: float = 0.6
    pattern_specificity_weight: float = 0.2
    pattern_context_weight: float = 0.2
    
    # Base confidence values
    base_syntactic_confidence: float = 0.6
    base_semantic_confidence: float = 0.7
    base_pattern_confidence: float = 0.8
    high_confidence_threshold: float = 0.8


@dataclass  
class WorkflowTimeoutConfiguration:
    """Workflow and processing timeout configuration"""
    # Workflow execution timeouts
    default_workflow_timeout_seconds: int = 300
    tri_modal_orchestrator_timeout: float = 2.5
    search_workflow_timeout: float = 3.0
    extraction_workflow_timeout: int = 300
    
    # Node-level timeouts
    default_node_timeout_seconds: int = 300
    max_node_retries: int = 3
    
    # Processing timeouts
    document_processing_timeout: int = 600
    knowledge_extraction_timeout: int = 300
    graph_building_timeout: int = 180
    
    # Azure service specific timeouts (inherited from infrastructure but workflow-specific)
    workflow_openai_timeout: int = 60
    workflow_search_timeout: int = 30
    workflow_cosmos_timeout: int = 45


@dataclass
class MachineLearningHyperparametersConfiguration:
    """Machine learning and neural network hyperparameters"""
    # GNN Configuration presets
    simple_gnn_config: Dict[str, Any] = field(default_factory=lambda: {
        "node_feature_dim": 64,
        "hidden_dim": 128,
        "num_layers": 2,
        "learning_rate": 0.001,
    })
    
    medium_gnn_config: Dict[str, Any] = field(default_factory=lambda: {
        "node_feature_dim": 128,
        "hidden_dim": 256,
        "num_layers": 3,
        "learning_rate": 0.001,
    })
    
    complex_gnn_config: Dict[str, Any] = field(default_factory=lambda: {
        "node_feature_dim": 256,
        "hidden_dim": 512,
        "num_layers": 4,
        "learning_rate": 0.0005,
    })
    
    # Statistical thresholds
    sub_millisecond_threshold: float = 0.001
    
    # Feature extraction parameters
    max_features_default: int = 1000
    min_document_frequency: float = 0.01
    max_document_frequency: float = 0.95
    
    # Statistical significance
    statistical_significance_alpha: float = 0.05
    chi_square_p_value_default: float = 0.01
    confidence_interval_lower: float = 0.6
    confidence_interval_upper: float = 0.9


@dataclass
class DomainAnalyzerConfiguration:
    """Domain analyzer specific configuration parameters"""
    # Regex patterns for entity recognition
    technical_terms_pattern: str = r"\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b"
    model_names_pattern: str = r"\b(?:model|algorithm|system|framework)\s+\w+\b"
    process_steps_pattern: str = r"\b(?:step|phase|stage|procedure)\s+\d+\b"
    measurements_pattern: str = r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m|kg|g|%|degrees?)\b"
    identifiers_pattern: str = r"\b[A-Z]\d+(?:-[A-Z]\d+)*\b"
    
    # Action pattern regexes
    instructions_pattern: str = r"\b(?:install|configure|setup|initialize|create|build|deploy)\b"
    operations_pattern: str = r"\b(?:run|execute|perform|conduct|analyze|process)\b"
    maintenance_pattern: str = r"\b(?:maintain|repair|replace|check|inspect|clean)\b"
    troubleshooting_pattern: str = r"\b(?:troubleshoot|debug|fix|resolve|diagnose)\b"
    
    # Complexity scoring weights
    unique_ratio_weight: float = 0.3
    tech_density_weight: float = 0.3
    sentence_complexity_weight: float = 0.2
    concept_richness_weight: float = 0.2
    
    # Statistical thresholds
    sentence_complexity_normalizer: int = 20
    concept_richness_normalizer: int = 50
    user_domain_boost_factor: float = 1.5
    technical_density_threshold: float = 0.1
    complexity_score_threshold: float = 0.7
    high_technical_density_threshold: float = 0.3
    rich_vocabulary_threshold: float = 0.5
    concept_rich_threshold: int = 20
    
    # Array size limits
    top_concepts_limit: int = 50
    top_entities_limit: int = 20
    top_actions_limit: int = 15
    indicator_concepts_limit: int = 5
    relevant_entities_limit: int = 3
    relevant_actions_limit: int = 3
    
    # Content quality validation thresholds
    min_meaningful_content_words: int = 50
    min_vocabulary_richness: float = 0.1
    min_sentence_length: float = 3.0
    min_concepts_for_quality: int = 5
    top_alternatives_limit: int = 6  # Top 5 alternatives + primary
    
    # Domain scoring multipliers
    domain_indicator_multiplier: float = 10.0
    entity_score_multiplier: float = 2.0
    action_score_multiplier: float = 1.5
    technical_density_score_multiplier: float = 2.0
    complexity_score_multiplier: float = 1.5
    vocabulary_richness_multiplier: float = 1.2
    concept_frequency_divisor: float = 50.0
    
    # Stop words list
    stop_words: List[str] = field(default_factory=lambda: [
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", 
        "up", "about", "into", "through", "during", "before", "after", "above", "below", "down", "out", 
        "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", 
        "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", 
        "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", 
        "should", "now"
    ])
    
    # Concept analysis parameters
    min_word_length_for_concepts: int = 4
    min_concept_length: int = 5
    min_meaningful_entity_length: int = 2


@dataclass
class PatternRecognitionConfiguration:
    """Regex patterns and entity detection patterns"""
    # Technical analysis patterns
    technical_patterns: List[str] = field(default_factory=lambda: [
        r"\bAPI\b", r"\bSDK\b", r"\bJSON\b", r"\bXML\b", r"\bREST\b",
        r"\bGET\b", r"\bPOST\b", r"\bHTTP\b", r"\bURL\b", r"\bURI\b"
    ])
    
    # Technical measurement patterns
    tech_patterns: List[str] = field(default_factory=lambda: [
        r"\b\d+(?:\.\d+)?(?:mm|cm|m|kg|g|%|degrees?)\b",  # Measurements
        r"\b[A-Z]\d+(?:-[A-Z]\d+)*\b",  # Technical codes
        r"\b(?:v\d+\.\d+|version\s+\d+)\b",  # Versions
        r"\b\w+\(\w*\)\b",  # Function calls
    ])
    
    # Process-indicating patterns
    process_patterns: List[str] = field(default_factory=lambda: [
        r"\bstep\s*\d+",
        r"\bphase\s*\d+",
        r"\bstage\s*\d+",
        r"\bprocess\b",
        r"\bprocedure\b",
    ])
    
    # Academic/research patterns
    academic_patterns: List[str] = field(default_factory=lambda: [
        r"\bresearch\b", r"\bstudy\b", r"\banalysis\b", r"\bmethod\b",
        r"\bfindings\b", r"\bresults\b", r"\bconclusion\b", r"\babstract\b",
        r"\bliterature\b", r"\btheory\b", r"\bhypothesis\b", r"\bdata\b"
    ])
    
    # Configuration/setup patterns
    config_patterns: List[str] = field(default_factory=lambda: [
        r"\bconfig\b", r"\bsettings\b", r"\bsetup\b", r"\binstall\b",
        r"\bdeployment\b", r"\benvironment\b", r"\bparameters\b"
    ])
    
    # Maintenance/operational patterns  
    maintenance_patterns: List[str] = field(default_factory=lambda: [
        r"\bmaintenance\b", r"\bupdate\b", r"\bpatch\b", r"\bfix\b",
        r"\bbug\b", r"\bissue\b", r"\berror\b", r"\btroubleshoot\b"
    ])
    
    # Analytical/statistical patterns
    analytical_patterns: List[str] = field(default_factory=lambda: [
        r"\banalysis\b", r"\bstatistics\b", r"\bmetrics\b", r"\breport\b",
        r"\bdashboard\b", r"\bKPI\b", r"\bperformance\b", r"\btrend\b"
    ])
    
    # Statistical pattern keywords
    statistical_patterns: List[str] = field(default_factory=lambda: [
        "distribution", "correlation", "regression", "variance", "deviation",
        "probability", "confidence", "significance", "hypothesis", "sample"
    ])
    
    # Security/dangerous patterns for input validation
    dangerous_patterns: List[str] = field(default_factory=lambda: [
        "drop table", "delete from", "insert into", "update set", "truncate",
        "alter table", "create table", "drop database", "exec", "execute"
    ])


@dataclass
class DomainAdaptiveConfiguration:
    """Configuration ranges for domain intelligence agent to determine dynamically"""
    # Entity extraction ranges
    entity_confidence_range: tuple = (0.3, 0.9)
    max_entities_per_chunk_range: tuple = (5, 50)
    
    # Relationship extraction ranges  
    relationship_confidence_range: tuple = (0.2, 0.8)
    max_relationships_per_entity_range: tuple = (3, 20)
    
    # Processing parameter ranges
    chunk_size_range: tuple = (500, 2000)
    chunk_overlap_range: tuple = (0.1, 0.3)
    
    # Domain-specific thresholds
    technical_density_range: tuple = (0.0, 1.0)
    vocabulary_richness_range: tuple = (0.0, 1.0)
    complexity_score_range: tuple = (0.0, 1.0)


@dataclass
class AzureServicesConfiguration:
    """Azure services and integration configuration parameters"""
    # Azure OpenAI defaults
    default_azure_openai_endpoint: str = "https://example.openai.azure.com/"
    default_openai_api_version: str = "2024-02-15-preview"
    fallback_api_version: str = "2024-02-15-preview"
    
    # Azure services scope
    cognitive_services_scope: str = "https://cognitiveservices.azure.com/.default"
    
    # Health check defaults
    health_percentage_multiplier: int = 100
    max_total_checked_divisor: int = 1
    min_healthy_services: int = 0
    degraded_threshold: int = 0
    healthy_threshold: int = 0
    
    # Service status defaults
    services_initial_count: int = 0
    successful_services_initial: int = 0
    has_service_default: bool = False
    overall_health_degraded: str = "degraded"
    overall_health_healthy: str = "healthy"
    
    # Health check status values
    status_not_initialized: str = "not_initialized"
    status_healthy: str = "healthy"
    status_unhealthy_prefix: str = "unhealthy: "
    status_degraded: str = "degraded"
    status_unknown: str = "unknown"


@dataclass
class KnowledgeExtractionAgentConfiguration:
    """Knowledge extraction agent configuration parameters"""
    # Azure OpenAI configuration
    azure_endpoint: str = "https://oai-maintie-rag-prod-fymhwfec3ra2w.openai.azure.com/"
    api_version: str = "2024-08-01-preview"
    deployment_name: str = "gpt-4o-mini"
    
    # Fallback model defaults
    fallback_domain_name: str = "general"
    fallback_entity_confidence_threshold: float = 0.7
    fallback_relationship_confidence_threshold: float = 0.65
    fallback_minimum_quality_score: float = 0.6
    fallback_enable_caching: bool = True
    fallback_cache_ttl_seconds: int = 3600
    fallback_max_concurrent_chunks: int = 5
    fallback_extraction_timeout_seconds: int = 300
    fallback_enable_monitoring: bool = True
    
    # Performance metrics defaults
    memory_usage_default_mb: float = 50.0
    cpu_utilization_default_percent: float = 60.0
    cache_hit_rate_default: float = 0.8
    
    # Quality score multipliers
    entity_precision_multiplier: float = 0.9
    entity_recall_multiplier: float = 0.8
    relationship_precision_multiplier: float = 0.85
    relationship_recall_multiplier: float = 0.75
    
    # Processing defaults
    processing_time_initial: float = 0.0
    extraction_count_initial: int = 0
    confidence_default: float = 0.0
    cache_hit_rate_disabled: float = 0.0
    max_documents_divisor: int = 1
    max_successful_extractions: int = 1
    validation_error_count_default: int = 1


@dataclass
class AgentContractsConfiguration:
    """Agent contracts and service integration parameters"""
    # Statistical significance thresholds
    chi_square_significance_alpha: float = 0.05
    min_pattern_frequency: int = 3
    confidence_interval_level: float = 0.95
    statistical_confidence_threshold_default: float = 0.7
    statistical_confidence_min: float = 0.5
    statistical_confidence_max: float = 0.95

    # Performance and timeout constraints
    max_execution_time_seconds: float = 300.0
    max_execution_time_min: float = 0.0
    max_execution_time_limit: float = 1800.0
    max_azure_service_cost_usd: float = 10.0
    max_azure_service_cost_limit: float = 100.0

    # Search and query limits
    max_query_length: int = 1000
    min_query_length: int = 1
    max_total_search_time_seconds: float = 3.0
    max_search_time_limit: float = 10.0
    max_results_per_modality: int = 20
    max_results_limit: int = 100
    search_cost_budget_usd: float = 1.0
    search_cost_limit: float = 10.0

    # Graph traversal constraints
    graph_traversal_depth_min: int = 1
    graph_traversal_depth_max: int = 5

    # Workflow orchestration
    workflow_max_execution_time: float = 10.0
    workflow_max_execution_limit: float = 30.0
    workflow_max_cost_usd: float = 5.0
    workflow_max_cost_limit: float = 50.0
    workflow_min_quality_score: float = 0.8
    workflow_quality_min: float = 0.5
    workflow_quality_max: float = 1.0

    # Performance target thresholds
    performance_violation_threshold: float = 10.0
    quality_concern_threshold: float = 0.7

    # Cache configuration
    cache_ttl_default: int = 3600
    cache_ttl_min: int = 60
    cache_ttl_max: int = 86400
    local_cache_size_default_mb: int = 100
    local_cache_size_min_mb: int = 10
    local_cache_size_max_mb: int = 1000

    # Architecture compliance
    azure_service_coverage_threshold: float = 0.95
    tool_delegation_coverage_threshold: float = 0.90

    # Model and service thresholds
    min_model_accuracy: float = 0.7
    min_document_count: int = 1
    min_vocabulary_size: int = 1
    min_document_length: float = 1.0
    min_topic_coherence: float = 0.0
    max_topic_coherence: float = 1.0


@dataclass
class HybridDomainAnalyzerConfiguration:
    """Hybrid domain analyzer biases and predetermined knowledge parameters"""
    # Chunk size calculations and multipliers (from hybrid_domain_analyzer.py)
    base_chunk_size: int = 1000
    chunk_size_multiplier_high_complexity: float = 0.7
    chunk_size_multiplier_medium_complexity: float = 1.0  
    chunk_size_multiplier_low_complexity: float = 1.3
    chunk_size_min: int = 500
    chunk_size_max: int = 2000
    
    # Entity density thresholds for chunk sizing
    high_entity_density_chunk_threshold: float = 20.0
    low_entity_density_chunk_threshold: float = 5.0
    entity_density_multiplier_high: float = 0.8
    entity_density_multiplier_low: float = 1.2
    technical_specs_multiplier: float = 0.8
    
    # Chunk overlap calculations
    base_overlap_ratio: float = 0.2
    high_relationships_overlap_ratio: float = 0.25
    procedural_content_overlap_ratio: float = 0.3
    low_complexity_overlap_ratio: float = 0.15
    overlap_ratio_min: float = 0.1
    overlap_ratio_max: float = 0.4
    
    # Domain classification multipliers and biases
    domain_confidence_multiplier_high: float = 0.9
    domain_confidence_multiplier_medium: float = 1.0
    domain_confidence_multiplier_low: float = 1.1
    domain_adjustment_technical: float = 0.95
    domain_adjustment_academic: float = 1.0
    domain_adjustment_process: float = 0.9
    domain_adjustment_general: float = 1.05
    
    # Confidence threshold calculations (hardcoded base thresholds)
    base_entity_confidence_threshold: float = 0.7
    base_relationship_confidence_threshold: float = 0.6
    base_overall_confidence_threshold: float = 0.65
    confidence_threshold_min: float = 0.5
    confidence_threshold_max: float = 0.9
    
    # Complexity scoring for chunk optimization
    complexity_multiplier_high: float = 1.5
    complexity_multiplier_medium: float = 1.0
    complexity_multiplier_low: float = 0.7
    
    # Processing load estimation factors
    base_load_per_1000_words: float = 1.0
    entity_factor_divisor: float = 100.0
    relationship_factor_divisor: float = 50.0
    
    # Performance optimization thresholds
    max_concurrent_base: int = 2
    max_concurrent_scaling_divisor: int = 2000
    max_concurrent_max: int = 10
    timeout_base_seconds: int = 30
    complexity_timeout_multiplier_high: float = 1.5
    complexity_timeout_multiplier_medium: float = 1.0
    complexity_timeout_multiplier_low: float = 0.8
    
    # Hybrid confidence calculation weights
    llm_confidence_weight: float = 0.6
    statistical_confidence_weight: float = 0.4
    llm_confidence_high_score: float = 0.9
    llm_confidence_medium_score: float = 0.7
    llm_confidence_low_score: float = 0.5
    
    # Statistical confidence calculation components
    entity_density_weight: float = 0.3
    vocabulary_complexity_weight: float = 0.3
    processing_load_weight: float = 0.4
    entity_density_divisor: float = 20.0
    processing_load_divisor: float = 10.0
    hybrid_confidence_min: float = 0.1
    hybrid_confidence_max: float = 1.0
    
    # Vocabulary complexity calculation
    ttr_weight: float = 0.6  # Type-token ratio weight
    tech_density_weight: float = 0.4
    tech_density_normalizer: float = 50.0
    
    # LLM extraction limits and constraints
    max_text_length_multiplier: int = 3  # 3x chunk size
    entity_min_default: int = 5
    entity_max_default: int = 15
    vocab_limit_default: int = 50
    
    # Pattern age degradation (30-day half-life bias)
    pattern_age_half_life_days: int = 30
    pattern_age_factor_min: float = 0.5
    
    # Quality settings biases
    min_entity_count_divisor: int = 2
    min_relationship_count_divisor: int = 2
    cache_ttl_seconds: int = 3600
    
    # Extraction strategy mapping biases
    extraction_strategy_technical: str = "TECHNICAL_CONTENT"
    extraction_strategy_process: str = "STRUCTURED_DATA"
    extraction_strategy_general: str = "MIXED_CONTENT"
    extraction_strategy_academic: str = "CONVERSATIONAL"
    
    # Complexity score calculation components
    complexity_to_score_high: float = 0.8
    complexity_to_score_medium: float = 0.5
    complexity_to_score_low: float = 0.2


@dataclass
class PatternEngineConfiguration:
    """Pattern engine biases and hardcoded extraction rules"""
    # Pattern extraction confidence multipliers
    entity_pattern_confidence_multiplier: float = 0.8
    action_pattern_confidence_multiplier: float = 0.8
    relationship_pattern_confidence_multiplier: float = 0.7
    temporal_pattern_confidence_multiplier: float = 0.6
    
    # Pattern filtering thresholds
    min_pattern_frequency: int = 2
    min_confidence_threshold: float = 0.3
    max_patterns_per_type: int = 100
    
    # Semantic clustering parameters
    word_overlap_threshold: float = 0.3  # 30% overlap threshold
    max_clusters_returned: int = 10
    
    # Pattern age and relevance scoring
    pattern_age_half_life_days: int = 30
    age_factor_min: float = 0.5
    frequency_boost_divisor: float = 100.0
    domain_match_boost: float = 1.5
    
    # Entity extraction patterns (hardcoded regex biases)
    technical_terms_pattern: str = r"\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b"
    model_names_pattern: str = r"\b(?:model|algorithm|system|framework|version)\s+([A-Za-z0-9\-\.]+)\b"
    identifiers_pattern: str = r"\b[A-Z]\d+(?:-[A-Z0-9]+)*\b"
    measurements_pattern: str = r"\b\d+(?:\.\d+)?\s*(mm|cm|m|kg|g|%|degrees?|rpm|hz|mhz|ghz)\b"
    codes_pattern: str = r"\b[A-Z]{2,}\d{2,}\b"
    proper_nouns_pattern: str = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
    
    # Action extraction patterns (hardcoded action biases)
    instructions_pattern: str = r"\b(install|configure|setup|initialize|create|build|deploy|start|stop|restart)\b"
    operations_pattern: str = r"\b(run|execute|perform|conduct|analyze|process|calculate|measure)\b"
    maintenance_pattern: str = r"\b(maintain|repair|replace|check|inspect|clean|service|troubleshoot)\b"
    monitoring_pattern: str = r"\b(monitor|observe|track|log|record|measure|assess)\b"
    management_pattern: str = r"\b(manage|control|supervise|coordinate|organize|plan)\b"
    
    # Relationship extraction patterns (hardcoded relationship biases)
    causation_pattern: str = r"\b(\w+(?:\s+\w+)*)\s+(?:causes?|results?\s+in|leads?\s+to)\s+(\w+(?:\s+\w+)*)\b"
    dependency_pattern: str = r"\b(\w+(?:\s+\w+)*)\s+(?:depends?\s+on|requires?|needs?)\s+(\w+(?:\s+\w+)*)\b"
    composition_pattern: str = r"\b(\w+(?:\s+\w+)*)\s+(?:contains?|includes?|comprises?)\s+(\w+(?:\s+\w+)*)\b"
    association_pattern: str = r"\b(\w+(?:\s+\w+)*)\s+(?:is\s+(?:associated|connected|linked)\s+with|relates?\s+to)\s+(\w+(?:\s+\w+)*)\b"
    
    # Temporal patterns (hardcoded temporal assumptions)
    time_expressions_pattern: str = r"\b(?:after|before|during|while|when|then|next|first|finally)\s+\w+(?:\s+\w+)*\b"
    sequences_pattern: str = r"\b(?:step|phase|stage)\s+\d+[:\s]+[^.!?]*"
    durations_pattern: str = r"\b\d+\s*(?:minutes?|hours?|days?|weeks?|months?|years?)\b"
    
    # Query matching and scoring biases
    min_word_length_for_matching: int = 2
    domain_pattern_score_boost: float = 1.5
    
    # Statistical filtering parameters
    min_pattern_length_entities: int = 2
    min_pattern_length_relationships: int = 5
    min_pattern_length_temporal: int = 5


@dataclass
class StatisticalDomainAnalyzerConfiguration:
    """Statistical domain analyzer biases despite claiming pure mathematics"""
    # TF-IDF vectorizer biases (language and parameter assumptions)
    max_features: int = 1000
    stop_words_language: str = "english"  # Language bias
    ngram_range_min: int = 1
    ngram_range_max: int = 3
    min_document_frequency: float = 0.01
    max_document_frequency: float = 0.95
    
    # K-means clustering biases (arbitrary cluster assumption)
    n_clusters: int = 5  # Assumes all domains fit into 5 clusters
    random_state: int = 42
    n_init: int = 10
    
    # Domain hypothesis scoring formulas (predetermined domain assumptions)
    # Technical domain hypothesis weights
    technical_lexical_diversity_weight: float = 0.4
    technical_entropy_weight: float = 0.3
    technical_word_length_weight: float = 0.3
    technical_word_length_normalizer: float = 10.0
    
    # Process domain hypothesis weights  
    process_sentence_complexity_weight: float = 0.5
    process_frequency_skew_weight: float = 0.3
    process_vocabulary_richness_weight: float = 0.2
    
    # Academic domain hypothesis weights
    academic_entropy_weight: float = 0.4
    academic_lexical_diversity_weight: float = 0.3
    academic_sentence_length_weight: float = 0.3
    academic_sentence_length_normalizer: float = 20.0
    
    # General domain fallback score
    general_domain_baseline_score: float = 0.5
    
    # Confidence calculation weights (predetermined importance hierarchy)
    confidence_base_score_weight: float = 0.6
    confidence_sample_size_weight: float = 0.2
    confidence_entropy_weight: float = 0.1
    confidence_complexity_weight: float = 0.1
    
    # Statistical thresholds (arbitrary boundaries)
    sample_size_confidence_divisor: float = 1000.0  # Words for full confidence
    entropy_confidence_divisor: float = 5.0
    complexity_confidence_divisor: float = 3.0
    
    # Entropy categorization thresholds (hardcoded boundaries)
    entropy_low_threshold: float = 2.0
    entropy_medium_threshold: float = 4.0
    entropy_high_threshold: float = 6.0
    entropy_category_low: float = 0.2
    entropy_category_medium: float = 0.5
    entropy_category_high: float = 0.8
    entropy_category_very_high: float = 1.0
    
    # Evidence generation thresholds (arbitrary statistical boundaries)
    high_entropy_threshold: float = 3.0
    high_vocabulary_richness_threshold: float = 0.6
    high_complexity_threshold: float = 0.5
    large_sample_size_threshold: int = 1000
    
    # Clause marker assumptions (English language bias)
    clause_markers: List[str] = field(default_factory=lambda: [
        ",", ";", ":", "and", "but", "or", "because", "since", "while"
    ])
    
    # Statistical significance assumptions
    significance_alpha: float = 0.05  # Standard but may not fit domain classification
    confidence_interval_multiplier: float = 1.96  # 95% confidence interval
    
    # TF-IDF feature processing limits
    top_features_limit: int = 20
    min_tfidf_score_threshold: float = 0.0
    
    # Confidence interval margin assumptions
    complexity_confidence_margin: float = 0.1  # 10% margin of error
    
    # Frequency skew calculation minimum samples
    min_frequency_samples_for_skew: int = 3
    
    # Vocabulary concentration (Gini coefficient) parameters
    gini_coefficient_adjustment: float = 1.0
    
    # Cluster membership thresholds
    high_importance_threshold: float = 0.1
    medium_importance_threshold_min: float = 0.05
    medium_importance_threshold_max: float = 0.1


@dataclass
class BackgroundProcessorConfiguration:
    """Background processor filesystem and processing biases"""
    # Hardcoded data directory path (filesystem structure bias)
    default_data_directory: str = "/workspace/azure-maintie-rag/data/raw"
    
    # File format restrictions (file type bias)
    supported_file_extensions: List[str] = field(default_factory=lambda: ["*.md", "*.txt"])
    
    # Directory discovery assumptions
    ignore_hidden_directories: bool = True  # Assumes hidden dirs aren't relevant
    directory_prefix_ignore: str = "."
    
    # Processing assumptions
    parallel_processing_enabled: bool = True
    require_files_for_domain: bool = True  # Assumes domains need files
    
    # Cache optimization assumptions
    pattern_index_optimization: bool = True
    expired_entries_cleanup: bool = True
    
    # Performance logging assumptions
    log_completion_statistics: bool = True
    log_processing_rates: bool = True
    log_success_rates: bool = True
    
    # Domain signature assumptions
    top_concepts_limit: int = 5
    high_confidence_pattern_threshold: float = 0.8
    
    # Aggregation and merging assumptions
    merge_similar_patterns_enabled: bool = True
    calculate_overall_confidence: bool = True
    
    # Statistical confidence fallback
    confidence_fallback_score: float = 0.3


@dataclass
class ConfigGeneratorConfiguration:
    """Config generator biases and predetermined relationship mappings"""
    # Primary concepts selection limits
    primary_concepts_fallback_limit: int = 3
    primary_concepts_main_limit: int = 2
    
    # Resource naming biases
    fallback_secondary_concept: str = "data"
    
    # Complexity assessment thresholds (hardcoded complexity boundaries)
    complexity_score_simple_threshold: int = 4
    complexity_score_medium_threshold: int = 2
    
    # Entity count complexity thresholds
    entity_count_high_threshold: int = 100
    entity_count_medium_threshold: int = 50
    
    # High confidence complexity thresholds
    high_confidence_count_high_threshold: int = 20
    high_confidence_count_medium_threshold: int = 10
    
    # Relationship count thresholds
    relationship_count_threshold: int = 5
    
    # Extraction confidence threshold
    extraction_confidence_threshold: float = 0.8
    
    # ML model scaling parameters
    entity_count_node_feature_multiplier: int = 4
    entity_count_hidden_dim_multiplier: int = 8
    entity_types_layers_divisor: int = 10
    entity_types_layers_base: int = 2
    
    # ML model dimension constraints
    node_feature_dim_min: int = 64
    node_feature_dim_max: int = 256
    hidden_dim_min: int = 128
    hidden_dim_max: int = 512
    num_layers_min: int = 2
    num_layers_max: int = 4
    
    # Learning rate adjustment factors
    low_confidence_learning_rate_factor: float = 0.5
    low_confidence_threshold: float = 0.6
    
    # Relationship inference parameters
    relationship_confidence_threshold: float = 0.6
    relationship_fallback_confidence: float = 0.5
    
    # Hardcoded relationship verb biases (predetermined relationship types)
    relationship_verbs_connect: List[str] = field(default_factory=lambda: ["connect", "link", "join", "bind"])
    relationship_verbs_contain: List[str] = field(default_factory=lambda: ["contain", "include", "hold", "have"])
    relationship_verbs_use: List[str] = field(default_factory=lambda: ["use", "utiliz", "employ", "apply"])
    relationship_verbs_create: List[str] = field(default_factory=lambda: ["create", "generat", "produc", "make"])
    relationship_verbs_part: List[str] = field(default_factory=lambda: ["part", "component", "element", "piece"])
    relationship_verbs_depend: List[str] = field(default_factory=lambda: ["depend", "rel", "requir", "need"])
    
    # Relationship type mappings (hardcoded relationship assumptions)
    relationship_type_connects: str = "connects"
    relationship_type_contains: str = "contains"
    relationship_type_uses: str = "uses"
    relationship_type_creates: str = "creates"
    relationship_type_part_of: str = "part_of"
    relationship_type_relates_to: str = "relates_to"
    
    # Top relationship limit
    top_relationships_limit: int = 5
    
    # Resource name cleaning parameters
    min_resource_name_length: int = 3
    max_resource_name_length: int = 50
    resource_name_fallback: str = "data"


@dataclass
class AgentConfiguration:
    """Agent.py biuses and model assumptions"""
    # Model deployment defaults (Azure OpenAI bias)
    default_openai_api_version: str = "2024-08-01-preview"
    default_model_deployment: str = "gpt-4.1"
    
    # Azure resource naming assumptions (filesystem structure bias)
    # Assumes domain names can be inferred from directory structure
    domain_from_directory_enabled: bool = True
    
    # Model configuration biases
    azure_openai_required: bool = True  # Assumes Azure OpenAI is required


@dataclass
class VectorSearchConfiguration:
    """Vector search engine parameters"""
    # Search parameters
    search_type: str = "vector_similarity"
    similarity_threshold: float = 0.7
    default_top_k: int = 10
    max_results: int = 100
    
    # Embedding configuration
    embedding_dimensions: int = 1536
    embedding_model_name: str = "text-embedding-ada-002"
    similarity_metric: str = "cosine"
    
    # Performance parameters
    search_timeout_seconds: int = 30
    batch_size: int = 50
    cache_embeddings: bool = True
    cache_ttl_seconds: int = 3600
    simulated_processing_delay: float = 0.1  # For testing/simulation


@dataclass
class GraphSearchConfiguration:
    """Graph search engine parameters"""
    # Traversal parameters
    search_type: str = "graph_relationships"
    max_depth: int = 3
    max_entities: int = 10
    max_relationships: int = 50
    
    # Scoring weights (eliminate hardcoded bias)
    relationship_weights: Dict[str, float] = field(default_factory=lambda: {
        "contains": 1.0,
        "uses": 0.8,
        "implements": 0.9,
        "inherits": 0.7,
        "depends_on": 0.6,
    })
    
    # Performance parameters
    traversal_timeout_seconds: int = 45
    entity_relevance_threshold: float = 0.5
    path_scoring_weight: float = 0.8
    relationship_confidence_threshold: float = 0.6
    simulated_processing_delay: float = 0.15  # For testing/simulation


@dataclass  
class GNNSearchConfiguration:
    """GNN search engine parameters"""
    # Model parameters
    search_type: str = "gnn_prediction"
    pattern_threshold: float = 0.7
    max_predictions: int = 20
    confidence_threshold: float = 0.6
    
    # ML model configuration
    model_architecture: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_layers": [128, 64, 32],
        "dropout_rate": 0.2,
        "activation": "relu",
        "output_dim": 16,
    })
    
    # Training parameters
    training_config: Dict[str, Any] = field(default_factory=lambda: {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "early_stopping_patience": 10,
    })
    
    # Performance parameters
    prediction_timeout_seconds: int = 60
    model_cache_ttl_seconds: int = 7200
    model_name: str = "gnn_semantic_predictor"
    min_training_examples: int = 100
    simulated_processing_delay: float = 0.2  # For testing/simulation


@dataclass
class TriModalOrchestrationConfiguration:
    """Tri-modal search orchestration parameters"""
    # Result weighting (transparent, configurable)
    modality_weights: Dict[str, float] = field(default_factory=lambda: {
        "vector": 0.4,
        "graph": 0.3,
        "gnn": 0.3,
    })
    
    # Synthesis parameters
    result_synthesis_threshold: float = 0.5
    cross_modal_confidence_boost: float = 1.2
    minimum_modality_agreement: int = 2
    
    # Result synthesis weighting factors
    confidence_weight: float = 0.4
    agreement_weight: float = 0.3
    quality_weight: float = 0.3
    high_confidence_threshold: float = 0.8
    
    # Search orchestration parameters
    default_search_types: List[str] = field(default_factory=lambda: ["vector", "graph", "gnn"])
    max_results_per_modality: int = 10
    
    # Performance parameters
    total_search_timeout_seconds: int = 120
    parallel_execution: bool = True
    result_deduplication: bool = True
    max_final_results: int = 50


@dataclass
class UniversalSearchAgentConfiguration:
    """Main Universal Search Agent configuration"""
    # Sub-configurations
    vector_search: VectorSearchConfiguration = field(default_factory=VectorSearchConfiguration)
    graph_search: GraphSearchConfiguration = field(default_factory=GraphSearchConfiguration)
    gnn_search: GNNSearchConfiguration = field(default_factory=GNNSearchConfiguration)
    orchestration: TriModalOrchestrationConfiguration = field(default_factory=TriModalOrchestrationConfiguration)
    
    # Agent-level parameters
    default_search_mode: str = "tri_modal"
    enable_caching: bool = True
    cache_search_results: bool = True
    result_cache_ttl_seconds: int = 1800
    
    # Azure OpenAI configuration
    azure_endpoint: str = "https://oai-maintie-rag-prod-fymhwfec3ra2w.openai.azure.com/"
    api_version: str = "2024-08-01-preview"
    deployment_name: str = "gpt-4o-mini"


@dataclass
class CentralizedConfiguration:
    """Master configuration containing all subsystem configurations"""
    cache: CacheConfiguration = field(default_factory=CacheConfiguration)
    confidence: ConfidenceConfiguration = field(default_factory=ConfidenceConfiguration)
    processing: ProcessingConfiguration = field(default_factory=ProcessingConfiguration)
    entity_extraction: EntityExtractionConfiguration = field(default_factory=EntityExtractionConfiguration)
    domain_analysis: DomainAnalysisConfiguration = field(default_factory=DomainAnalysisConfiguration)
    ml: MLConfiguration = field(default_factory=MLConfiguration)
    validation: ValidationConfiguration = field(default_factory=ValidationConfiguration)
    relationship_processing: RelationshipProcessingConfiguration = field(default_factory=RelationshipProcessingConfiguration)
    entity_processing: EntityProcessingConfiguration = field(default_factory=EntityProcessingConfiguration)
    capability_patterns: CapabilityPatternsConfiguration = field(default_factory=CapabilityPatternsConfiguration)
    azure_services: AzureServicesConfiguration = field(default_factory=AzureServicesConfiguration)
    knowledge_extraction_agent: KnowledgeExtractionAgentConfiguration = field(default_factory=KnowledgeExtractionAgentConfiguration)
    agent_contracts: AgentContractsConfiguration = field(default_factory=AgentContractsConfiguration)
    # New configuration sections
    model: ModelConfiguration = field(default_factory=ModelConfiguration)
    infrastructure: InfrastructureConfiguration = field(default_factory=InfrastructureConfiguration)
    service_endpoints: ServiceEndpointConfiguration = field(default_factory=ServiceEndpointConfiguration)
    domain_adaptive: DomainAdaptiveConfiguration = field(default_factory=DomainAdaptiveConfiguration)
    # Critical hardcoded value sections
    domain_intelligence_decisions: DomainIntelligenceDecisionConfiguration = field(default_factory=DomainIntelligenceDecisionConfiguration)
    quality_assessment: QualityAssessmentConfiguration = field(default_factory=QualityAssessmentConfiguration)
    confidence_calculation: ConfidenceCalculationConfiguration = field(default_factory=ConfidenceCalculationConfiguration)
    pattern_recognition: PatternRecognitionConfiguration = field(default_factory=PatternRecognitionConfiguration)
    workflow_timeouts: WorkflowTimeoutConfiguration = field(default_factory=WorkflowTimeoutConfiguration)  
    ml_hyperparameters: MachineLearningHyperparametersConfiguration = field(default_factory=MachineLearningHyperparametersConfiguration)
    domain_analyzer: DomainAnalyzerConfiguration = field(default_factory=DomainAnalyzerConfiguration)
    # Domain intelligence bias centralization
    hybrid_domain_analyzer: HybridDomainAnalyzerConfiguration = field(default_factory=HybridDomainAnalyzerConfiguration)
    pattern_engine: PatternEngineConfiguration = field(default_factory=PatternEngineConfiguration)
    statistical_domain_analyzer: StatisticalDomainAnalyzerConfiguration = field(default_factory=StatisticalDomainAnalyzerConfiguration)
    background_processor: BackgroundProcessorConfiguration = field(default_factory=BackgroundProcessorConfiguration)
    config_generator: ConfigGeneratorConfiguration = field(default_factory=ConfigGeneratorConfiguration)
    agent: AgentConfiguration = field(default_factory=AgentConfiguration)
    # Universal Search Agent configurations
    vector_search: VectorSearchConfiguration = field(default_factory=VectorSearchConfiguration)
    graph_search: GraphSearchConfiguration = field(default_factory=GraphSearchConfiguration)
    gnn_search: GNNSearchConfiguration = field(default_factory=GNNSearchConfiguration)
    tri_modal_orchestration: TriModalOrchestrationConfiguration = field(default_factory=TriModalOrchestrationConfiguration)
    universal_search_agent: UniversalSearchAgentConfiguration = field(default_factory=UniversalSearchAgentConfiguration)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization"""
        result = {}
        for section_name in ['cache', 'confidence', 'processing', 'entity_extraction', 'domain_analysis', 'ml', 'validation', 'relationship_processing', 'entity_processing', 'capability_patterns', 'azure_services', 'knowledge_extraction_agent', 'agent_contracts', 'model', 'infrastructure', 'service_endpoints', 'domain_adaptive', 'domain_intelligence_decisions', 'quality_assessment', 'confidence_calculation', 'pattern_recognition', 'workflow_timeouts', 'ml_hyperparameters', 'domain_analyzer', 'hybrid_domain_analyzer', 'pattern_engine', 'statistical_domain_analyzer', 'background_processor', 'config_generator', 'agent', 'vector_search', 'graph_search', 'gnn_search', 'tri_modal_orchestration', 'universal_search_agent']:
            section = getattr(self, section_name)
            result[section_name] = {
                field.name: getattr(section, field.name)
                for field in section.__dataclass_fields__.values()
            }
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CentralizedConfiguration':
        """Create configuration from dictionary"""
        config = cls()

        for section_name, section_data in config_dict.items():
            if hasattr(config, section_name):
                section = getattr(config, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)

        return config


class ConfigurationManager:
    """Central configuration manager with environment override support"""

    def __init__(self, config_file: Optional[Path] = None):
        self.config = CentralizedConfiguration()
        self.config_file = config_file or Path(__file__).parent / "agent_config.json"
        self._load_configuration()
        self._apply_environment_overrides()

    def _load_configuration(self):
        """Load configuration from file if it exists"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                self.config = CentralizedConfiguration.from_dict(config_data)
            except Exception as e:
                print(f"Warning: Failed to load config from {self.config_file}: {e}")
                print("Using default configuration values")

    def _apply_environment_overrides(self):
        """Apply environment variable overrides"""
        env_mappings = {
            # Cache configuration
            'AGENT_CACHE_TTL': ('cache', 'default_ttl_seconds', int),
            'AGENT_REDIS_TTL': ('cache', 'redis_ttl_seconds', int),
            'AGENT_CACHE_HIT_RATE_EXCELLENT': ('cache', 'hit_rate_threshold_excellent', int),
            'AGENT_CACHE_HIT_RATE_GOOD': ('cache', 'hit_rate_threshold_good', int),

            # Confidence configuration
            'AGENT_CONFIDENCE_LEVEL': ('confidence', 'default_confidence_level', float),
            'AGENT_MIN_PATTERN_CONFIDENCE': ('confidence', 'minimum_pattern_confidence', float),
            'AGENT_ENTITY_CONFIDENCE': ('confidence', 'entity_confidence_threshold', float),

            # Processing configuration
            'AGENT_MAX_WORKERS': ('processing', 'max_workers', int),
            'AGENT_CHUNK_SIZE': ('processing', 'chunk_size_default', int),
            'AGENT_TIMEOUT_SECONDS': ('processing', 'timeout_base_seconds', int),

            # ML configuration
            'AGENT_KMEANS_CLUSTERS': ('ml', 'kmeans_clusters', int),
            'AGENT_RANDOM_STATE': ('ml', 'random_state', int),
            
            # Model configuration
            'AZURE_OPENAI_DEPLOYMENT_GPT4O': ('model', 'gpt4o_deployment_name', str),
            'AZURE_OPENAI_DEPLOYMENT_GPT4O_MINI': ('model', 'gpt4o_mini_deployment_name', str),
            'AZURE_OPENAI_DEPLOYMENT_EMBEDDING': ('model', 'text_embedding_deployment_name', str),
            'AZURE_OPENAI_API_VERSION': ('model', 'openai_api_version', str),
            
            # Infrastructure configuration
            'AGENT_OPENAI_TIMEOUT': ('infrastructure', 'openai_timeout', int),
            'AGENT_SEARCH_TIMEOUT': ('infrastructure', 'search_timeout', int),
            'AGENT_COSMOS_TIMEOUT': ('infrastructure', 'cosmos_timeout', int),
            'AGENT_MAX_RETRIES': ('infrastructure', 'max_retries', int),
            'AGENT_MAX_BATCH_SIZE': ('infrastructure', 'max_batch_size', int),
        }

        for env_var, (section_name, field_name, type_converter) in env_mappings.items():
            if env_var in os.environ:
                try:
                    value = type_converter(os.environ[env_var])
                    section = getattr(self.config, section_name)
                    setattr(section, field_name, value)
                except ValueError as e:
                    print(f"Warning: Invalid {env_var} value: {os.environ[env_var]} ({e})")

    def save_configuration(self):
        """Save current configuration to file"""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

    def get_cache_config(self) -> CacheConfiguration:
        """Get cache configuration"""
        return self.config.cache

    def get_confidence_config(self) -> ConfidenceConfiguration:
        """Get confidence configuration"""
        return self.config.confidence

    def get_processing_config(self) -> ProcessingConfiguration:
        """Get processing configuration"""
        return self.config.processing

    def get_entity_extraction_config(self) -> EntityExtractionConfiguration:
        """Get entity extraction configuration"""
        return self.config.entity_extraction

    def get_domain_analysis_config(self) -> DomainAnalysisConfiguration:
        """Get domain analysis configuration"""
        return self.config.domain_analysis

    def get_ml_config(self) -> MLConfiguration:
        """Get ML configuration"""
        return self.config.ml

    def get_validation_config(self) -> ValidationConfiguration:
        """Get validation configuration"""
        return self.config.validation
    
    def get_relationship_processing_config(self) -> RelationshipProcessingConfiguration:
        """Get relationship processing configuration"""
        return self.config.relationship_processing
    
    def get_entity_processing_config(self) -> EntityProcessingConfiguration:
        """Get entity processing configuration"""
        return self.config.entity_processing
    
    def get_capability_patterns_config(self) -> CapabilityPatternsConfiguration:
        """Get capability patterns configuration"""
        return self.config.capability_patterns
    
    def get_azure_services_config(self) -> AzureServicesConfiguration:
        """Get azure services configuration"""
        return self.config.azure_services
    
    def get_knowledge_extraction_agent_config(self) -> KnowledgeExtractionAgentConfiguration:
        """Get knowledge extraction agent configuration"""
        return self.config.knowledge_extraction_agent
    
    def get_agent_contracts_config(self) -> AgentContractsConfiguration:
        """Get agent contracts configuration"""
        return self.config.agent_contracts
    
    def get_model_config(self) -> ModelConfiguration:
        """Get model configuration"""
        return self.config.model
    
    def get_infrastructure_config(self) -> InfrastructureConfiguration:
        """Get infrastructure configuration"""
        return self.config.infrastructure
    
    def get_service_endpoints_config(self) -> ServiceEndpointConfiguration:
        """Get service endpoints configuration"""
        return self.config.service_endpoints
    
    def get_domain_adaptive_config(self) -> DomainAdaptiveConfiguration:
        """Get domain adaptive configuration"""
        return self.config.domain_adaptive
    
    def get_domain_intelligence_decisions_config(self) -> DomainIntelligenceDecisionConfiguration:
        """Get domain intelligence decision configuration"""
        return self.config.domain_intelligence_decisions
    
    def get_quality_assessment_config(self) -> QualityAssessmentConfiguration:
        """Get quality assessment configuration"""
        return self.config.quality_assessment
    
    def get_confidence_calculation_config(self) -> ConfidenceCalculationConfiguration:
        """Get confidence calculation configuration"""
        return self.config.confidence_calculation
    
    def get_pattern_recognition_config(self) -> PatternRecognitionConfiguration:
        """Get pattern recognition configuration"""
        return self.config.pattern_recognition
    
    def get_workflow_timeouts_config(self) -> WorkflowTimeoutConfiguration:
        """Get workflow timeouts configuration"""
        return self.config.workflow_timeouts
    
    def get_ml_hyperparameters_config(self) -> MachineLearningHyperparametersConfiguration:
        """Get machine learning hyperparameters configuration"""
        return self.config.ml_hyperparameters
    
    def get_domain_analyzer_config(self) -> DomainAnalyzerConfiguration:
        """Get domain analyzer configuration"""
        return self.config.domain_analyzer
    
    def get_hybrid_domain_analyzer_config(self) -> HybridDomainAnalyzerConfiguration:
        """Get hybrid domain analyzer configuration"""
        return self.config.hybrid_domain_analyzer
    
    def get_pattern_engine_config(self) -> PatternEngineConfiguration:
        """Get pattern engine configuration"""
        return self.config.pattern_engine
    
    def get_statistical_domain_analyzer_config(self) -> StatisticalDomainAnalyzerConfiguration:
        """Get statistical domain analyzer configuration"""
        return self.config.statistical_domain_analyzer
    
    def get_background_processor_config(self) -> BackgroundProcessorConfiguration:
        """Get background processor configuration"""
        return self.config.background_processor
    
    def get_config_generator_config(self) -> ConfigGeneratorConfiguration:
        """Get config generator configuration"""
        return self.config.config_generator
    
    def get_agent_config(self) -> AgentConfiguration:
        """Get agent configuration"""
        return self.config.agent
    
    def get_vector_search_config(self) -> VectorSearchConfiguration:
        """Get vector search configuration"""
        return self.config.vector_search
    
    def get_graph_search_config(self) -> GraphSearchConfiguration:
        """Get graph search configuration"""
        return self.config.graph_search
    
    def get_gnn_search_config(self) -> GNNSearchConfiguration:
        """Get GNN search configuration"""
        return self.config.gnn_search
    
    def get_tri_modal_orchestration_config(self) -> TriModalOrchestrationConfiguration:
        """Get tri-modal orchestration configuration"""
        return self.config.tri_modal_orchestration
    
    def get_universal_search_agent_config(self) -> UniversalSearchAgentConfiguration:
        """Get universal search agent configuration"""
        return self.config.universal_search_agent


# Global configuration manager instance
_config_manager = None

def get_config_manager() -> ConfigurationManager:
    """Get the global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager

def get_cache_config() -> CacheConfiguration:
    """Convenience function to get cache configuration"""
    return get_config_manager().get_cache_config()

def get_confidence_config() -> ConfidenceConfiguration:
    """Convenience function to get confidence configuration"""
    return get_config_manager().get_confidence_config()

def get_processing_config() -> ProcessingConfiguration:
    """Convenience function to get processing configuration"""
    return get_config_manager().get_processing_config()

def get_entity_extraction_config() -> EntityExtractionConfiguration:
    """Convenience function to get entity extraction configuration"""
    return get_config_manager().get_entity_extraction_config()

def get_domain_analysis_config() -> DomainAnalysisConfiguration:
    """Convenience function to get domain analysis configuration"""
    return get_config_manager().get_domain_analysis_config()

def get_ml_config() -> MLConfiguration:
    """Convenience function to get ML configuration"""
    return get_config_manager().get_ml_config()

def get_validation_config() -> ValidationConfiguration:
    """Convenience function to get validation configuration"""
    return get_config_manager().get_validation_config()

def get_relationship_processing_config() -> RelationshipProcessingConfiguration:
    """Convenience function to get relationship processing configuration"""
    return get_config_manager().get_relationship_processing_config()

def get_entity_processing_config() -> EntityProcessingConfiguration:
    """Convenience function to get entity processing configuration"""
    return get_config_manager().get_entity_processing_config()

def get_capability_patterns_config() -> CapabilityPatternsConfiguration:
    """Convenience function to get capability patterns configuration"""
    return get_config_manager().get_capability_patterns_config()

def get_azure_services_config() -> AzureServicesConfiguration:
    """Convenience function to get azure services configuration"""
    return get_config_manager().get_azure_services_config()

def get_knowledge_extraction_agent_config() -> KnowledgeExtractionAgentConfiguration:
    """Convenience function to get knowledge extraction agent configuration"""
    return get_config_manager().get_knowledge_extraction_agent_config()

def get_agent_contracts_config() -> AgentContractsConfiguration:
    """Convenience function to get agent contracts configuration"""
    return get_config_manager().get_agent_contracts_config()

def get_model_config() -> ModelConfiguration:
    """Convenience function to get model configuration"""
    return get_config_manager().get_model_config()

def get_infrastructure_config() -> InfrastructureConfiguration:
    """Convenience function to get infrastructure configuration"""
    return get_config_manager().get_infrastructure_config()

def get_service_endpoints_config() -> ServiceEndpointConfiguration:
    """Convenience function to get service endpoints configuration"""
    return get_config_manager().get_service_endpoints_config()

def get_domain_adaptive_config() -> DomainAdaptiveConfiguration:
    """Convenience function to get domain adaptive configuration"""
    return get_config_manager().get_domain_adaptive_config()

def get_domain_intelligence_decisions_config() -> DomainIntelligenceDecisionConfiguration:
    """Convenience function to get domain intelligence decision configuration"""
    return get_config_manager().get_domain_intelligence_decisions_config()

def get_quality_assessment_config() -> QualityAssessmentConfiguration:
    """Convenience function to get quality assessment configuration"""
    return get_config_manager().get_quality_assessment_config()

def get_confidence_calculation_config() -> ConfidenceCalculationConfiguration:
    """Convenience function to get confidence calculation configuration"""
    return get_config_manager().get_confidence_calculation_config()

def get_pattern_recognition_config() -> PatternRecognitionConfiguration:
    """Convenience function to get pattern recognition configuration"""
    return get_config_manager().get_pattern_recognition_config()

def get_workflow_timeouts_config() -> WorkflowTimeoutConfiguration:
    """Convenience function to get workflow timeouts configuration"""
    return get_config_manager().get_workflow_timeouts_config()

def get_ml_hyperparameters_config() -> MachineLearningHyperparametersConfiguration:
    """Convenience function to get machine learning hyperparameters configuration"""
    return get_config_manager().get_ml_hyperparameters_config()

def get_domain_analyzer_config() -> DomainAnalyzerConfiguration:
    """Convenience function to get domain analyzer configuration"""
    return get_config_manager().get_domain_analyzer_config()

def get_hybrid_domain_analyzer_config() -> HybridDomainAnalyzerConfiguration:
    """Convenience function to get hybrid domain analyzer configuration"""
    return get_config_manager().get_hybrid_domain_analyzer_config()

def get_pattern_engine_config() -> PatternEngineConfiguration:
    """Convenience function to get pattern engine configuration"""
    return get_config_manager().get_pattern_engine_config()

def get_statistical_domain_analyzer_config() -> StatisticalDomainAnalyzerConfiguration:
    """Convenience function to get statistical domain analyzer configuration"""
    return get_config_manager().get_statistical_domain_analyzer_config()

def get_background_processor_config() -> BackgroundProcessorConfiguration:
    """Convenience function to get background processor configuration"""
    return get_config_manager().get_background_processor_config()

def get_config_generator_config() -> ConfigGeneratorConfiguration:
    """Convenience function to get config generator configuration"""
    return get_config_manager().get_config_generator_config()

def get_agent_config() -> AgentConfiguration:
    """Convenience function to get agent configuration"""
    return get_config_manager().get_agent_config()

def get_vector_search_config() -> VectorSearchConfiguration:
    """Convenience function to get vector search configuration"""
    return get_config_manager().get_vector_search_config()

def get_graph_search_config() -> GraphSearchConfiguration:
    """Convenience function to get graph search configuration"""
    return get_config_manager().get_graph_search_config()

def get_gnn_search_config() -> GNNSearchConfiguration:
    """Convenience function to get GNN search configuration"""
    return get_config_manager().get_gnn_search_config()

def get_tri_modal_orchestration_config() -> TriModalOrchestrationConfiguration:
    """Convenience function to get tri-modal orchestration configuration"""
    return get_config_manager().get_tri_modal_orchestration_config()

def get_universal_search_agent_config() -> UniversalSearchAgentConfiguration:
    """Convenience function to get universal search agent configuration"""
    return get_config_manager().get_universal_search_agent_config()
