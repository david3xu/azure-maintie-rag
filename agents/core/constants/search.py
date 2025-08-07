"""
Search and ML Model Constants
=============================

This module contains constants for ML models, algorithms, and search functionality.
Some are algorithmically determined (static), while others can be learned through
optimization processes.

Key Areas:
1. ML Model Architecture - potentially learnable through architecture search
2. Training Hyperparameters - learnable through hyperparameter optimization
3. Statistical Analysis - mix of standard values and learnable thresholds
4. Universal Search Agent - learnable from search quality optimization

AUTO-GENERATION POTENTIAL:
- MLModelStaticConstants: LOW-MEDIUM (some hyperparameters learnable)
- StatisticalConstants: MEDIUM (mix of standards and learnable thresholds)
- UniversalSearchConstants: MEDIUM-HIGH (many adaptive to search quality)
"""

from .domain import DomainAdaptiveConstants


class MLModelStaticConstants:
    """ML model constants that are algorithmically determined"""

    # AUTO-GENERATION POTENTIAL: LOW-MEDIUM

    # GNN Architecture - Could be learned through architecture search
    GNN_HIDDEN_DIM = 128  # POTENTIALLY LEARNABLE: architecture optimization
    GNN_NUM_LAYERS = 2  # POTENTIALLY LEARNABLE: architecture optimization

    # Training Hyperparameters - Should be learned through hyperparameter optimization
    GNN_LEARNING_RATE = 0.001  # LEARNABLE: hyperparameter optimization
    BATCH_SIZE = 32  # LEARNABLE: memory vs convergence optimization


class StatisticalConstants:
    """Statistical analysis constants - mix of standard values and learnable thresholds"""

    # AUTO-GENERATION POTENTIAL: MEDIUM

    # Standard Statistical Values - These are mathematically standard
    CHI_SQUARE_SIGNIFICANCE_ALPHA = 0.05  # STANDARD: convention

    # Domain-Specific Statistical Thresholds - LEARNABLE
    STATISTICAL_CONFIDENCE_THRESHOLD = 0.75  # LEARNABLE: domain confidence requirements
    STATISTICAL_CONFIDENCE_MIN = 0.0  # Minimum confidence bound
    STATISTICAL_CONFIDENCE_MAX = 1.0  # Maximum confidence bound
    MIN_PATTERN_FREQUENCY = 3  # LEARNABLE: significant pattern frequency

    # Domain Classification Thresholds - Referenced from domain module
    MIN_DOMAIN_CONFIDENCE = DomainAdaptiveConstants.MIN_DOMAIN_CONFIDENCE
    DOMAIN_CLASSIFICATION_THRESHOLD = (
        DomainAdaptiveConstants.DOMAIN_CLASSIFICATION_THRESHOLD
    )
    TECHNICAL_CONTENT_SIMILARITY_THRESHOLD = (
        DomainAdaptiveConstants.TECHNICAL_CONTENT_SIMILARITY_THRESHOLD
    )
    RICH_VOCABULARY_FACTOR = 0.6  # Factor for rich vocabulary content analysis


class UniversalSearchConstants:
    """Universal Search Agent specific constants"""

    # Many of these could be performance adaptive based on search quality

    DEFAULT_GRAPH_HOP_COUNT = 2  # LEARNABLE: optimal graph traversal depth
    DEFAULT_GNN_NODE_EMBEDDINGS = 128  # LEARNABLE: optimal embedding dimensionality
    GRAPH_MIN_RELATIONSHIP_STRENGTH = 0.5  # LEARNABLE: domain relationship quality
    GNN_MIN_PREDICTION_CONFIDENCE = 0.6  # ADAPTIVE: GNN calibration

    # Search Result Processing - Could be performance adaptive
    DEFAULT_VECTOR_TOP_K = 10  # ADAPTIVE: optimal result count for quality
    DEFAULT_MAX_DEPTH = 3  # Maximum graph traversal depth
    DEFAULT_MAX_ENTITIES = 50  # Maximum entities to extract
    DEFAULT_MAX_PREDICTIONS = 20  # Maximum GNN predictions
    DEFAULT_PATTERN_THRESHOLD = 0.7  # GNN pattern recognition threshold
    DEFAULT_MIN_TRAINING_EXAMPLES = 100  # Minimum examples for GNN training
    DEFAULT_RELATIONSHIP_THRESHOLD = 0.5  # Default relationship confidence threshold
    VECTOR_SIMILARITY_THRESHOLD = 0.7  # LEARNABLE: domain semantic similarity patterns
    MAX_SEARCH_RESULTS = 20  # ADAPTIVE: user interaction optimization
    HIGH_CONFIDENCE_THRESHOLD = 0.8  # LEARNABLE: domain confidence distributions

    # Multi-modal search weights - must sum to 1.0 (imported from performance module)
    MULTI_MODAL_WEIGHT_VECTOR = 0.4  # ADAPTIVE: based on vector search effectiveness
    MULTI_MODAL_WEIGHT_GRAPH = 0.3  # ADAPTIVE: based on graph search effectiveness
    MULTI_MODAL_WEIGHT_GNN = 0.3  # ADAPTIVE: based on GNN search effectiveness

    # Query Complexity Analysis - Could be learned from query patterns
    QUERY_LENGTH_SIMPLE_THRESHOLD = 5  # LEARNABLE: query complexity classification
    QUERY_LENGTH_COMPLEX_THRESHOLD = 10  # LEARNABLE: query complexity classification
    MAX_GRAPH_HOP_COUNT = 5  # Maximum graph traversal depth
    QUERY_COMPLEXITY_SIMPLE_MULTIPLIER = 0.8  # ADAPTIVE: complexity-based optimization
    QUERY_COMPLEXITY_MEDIUM_MULTIPLIER = 1.2  # ADAPTIVE: complexity-based optimization
    QUERY_COMPLEXITY_COMPLEX_MULTIPLIER = 1.3  # ADAPTIVE: complexity-based optimization

    # Fallback constants for backward compatibility
    FALLBACK_VECTOR_SIMILARITY_THRESHOLD = 0.7  # Domain semantic similarity patterns
    FALLBACK_VECTOR_TOP_K = 10  # Optimal result count for quality
    FALLBACK_GRAPH_HOP_COUNT = 2  # Based on relationship depth analysis
    FALLBACK_GRAPH_MIN_RELATIONSHIP_STRENGTH = 0.5  # Domain-specific threshold
    FALLBACK_GNN_PREDICTION_CONFIDENCE = 0.6  # Based on model performance for domain
    FALLBACK_GNN_NODE_EMBEDDINGS = 128  # Optimized for domain complexity
    FALLBACK_RESULT_SYNTHESIS_THRESHOLD = 0.6  # Quality threshold for domain


# Export all constants
__all__ = [
    "MLModelStaticConstants",
    "StatisticalConstants",
    "UniversalSearchConstants",
]
