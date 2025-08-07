"""
Mathematical Expressions - Centralized Mathematical Constants

This module centralizes all mathematical expressions and constants found throughout
the agents directory. Eliminates hardcoded mathematical values in calculations.

Purpose: Replace all hardcoded mathematical constants and expressions.
"""

# =============================================================================
# MATHEMATICAL EXPRESSION CONSTANTS
# =============================================================================


class MathConstants:
    """Mathematical constants extracted from all complex expressions in agents/"""

    # =============================================================================
    # NORMALIZATION AND SCALING FACTORS
    # =============================================================================

    # Text and content normalization divisors
    TEXT_LENGTH_NORMALIZER = 20.0  # Used in: len(text) / 20.0
    ENTROPY_NORMALIZER = 10.0  # Used in: entropy / 10.0
    WORD_COUNT_NORMALIZER = 1000.0  # Used in: word_count / 1000.0
    SENTENCE_LENGTH_NORMALIZER = 20.0  # Used in: avg_sentence_length / 20.0
    DOCUMENT_COUNT_NORMALIZER = 20.0  # Used in: len(documents) / 20.0

    # =============================================================================
    # CONFIDENCE AND QUALITY CALCULATIONS
    # =============================================================================

    # Quality threshold calculation factors (from hybrid_configuration_generator.py)
    QUALITY_BOUND_FACTOR = 0.57  # Used in: MIN_DOMAIN_CONFIDENCE * 0.57
    CONFIDENCE_SCALING = (
        0.57  # Used in: confidence_adjustment * MIN_DOMAIN_CONFIDENCE * 0.57
    )
    MIN_THRESHOLD_FACTOR = (
        0.43  # Used in: MIN_DOMAIN_CONFIDENCE * 0.43 (for 0.3 calculation)
    )

    # Confidence boost and adjustment factors
    CONFIDENCE_BOOST_MULTIPLIER = 1.3  # Used in: best_relationship.confidence * 1.3
    CONFIDENCE_BOOST_FACTOR = 0.2  # Used in: llm_analysis.confidence_score * 0.2
    PATTERN_BOOST_MULTIPLIER = 0.05  # Used in: len(relationship_patterns) * 0.05
    PATTERN_BOOST_LIMIT = 0.2  # Used in: min(0.2, pattern_calculation)

    # =============================================================================
    # RESPONSE SCORING WEIGHTS
    # =============================================================================

    # Response scoring weights (from responses.py)
    MODALITY_WEIGHT = 0.4  # Used in: modality_score calculation
    PERFORMANCE_WEIGHT = 0.3  # Used in: performance_score calculation
    QUALITY_WEIGHT = 0.3  # Used in: quality_score calculation
    MAX_MODALITIES = 3.0  # Used in: len(search_types_used) / 3.0
    SLA_BONUS = 0.1  # Used in: sla_bonus calculation
    BASE_SCORE_WEIGHT = 0.6  # Used in: (base_score * 0.6) + ...

    # =============================================================================
    # PROCESSING AND SIZING FACTORS
    # =============================================================================

    # Chunk and size calculations
    CHUNK_OVERLAP_RATIO = 0.15  # Used in: int(chunk_size * 0.15)
    SMALL_CHUNK_OVERLAP_RATIO = (
        0.1  # Used in: int(optimal_chunk_size * 0.1) - 10% overlap
    )
    SIZE_MULTIPLIER_LARGE = 1.5  # Used in: int(base_max * 1.5)

    # Threshold adjustment factors
    RELATIONSHIP_STRENGTH_FACTOR = 0.8  # Used in: relationship_threshold * 0.8
    QUALITY_VALIDATION_FACTOR = 0.9  # Used in: entity_threshold * 0.9

    # Progress calculation constants
    PERCENTAGE_MULTIPLIER = 100.0  # Used in: (finished_nodes / total_nodes) * 100.0
    PERFORMANCE_THRESHOLD_MULTIPLIER = 1.5  # Used in: target_time * 1.5

    # =============================================================================
    # PRECISION AND TOLERANCE VALUES
    # =============================================================================

    # Float precision tolerance (from responses.py)
    PRECISION_TOLERANCE = 0.01  # Used in: (0.99 <= total <= 1.01)

    # =============================================================================
    # CROSS-MODAL AND AGREEMENT CALCULATIONS
    # =============================================================================

    # Universal search calculations
    MODAL_AGREEMENT_NORMALIZER = (
        3.0  # Used in: active_modalities / 3.0 * average_confidence
    )


# =============================================================================
# MATHEMATICAL EXPRESSION FUNCTIONS
# =============================================================================


class MathExpressions:
    """Centralized mathematical expressions and calculations"""

    # =============================================================================
    # PERCENTAGE AND RATIO CALCULATIONS
    # =============================================================================

    @staticmethod
    def calculate_percentage(numerator: float, denominator: float) -> float:
        """Calculate percentage: (numerator / denominator) * 100"""
        return (numerator / max(1, denominator)) * MATH.PERCENTAGE_MULTIPLIER

    @staticmethod
    def calculate_utilization_percentage(current: float, limit: float) -> float:
        """Calculate utilization percentage with 100% cap"""
        return min(100.0, (current / limit) * 100)

    @staticmethod
    def calculate_hit_rate(hits: int, total_requests: int) -> float:
        """Calculate cache hit rate percentage"""
        return (hits / max(1, total_requests)) * 100

    # =============================================================================
    # CONFIDENCE AND QUALITY CALCULATIONS
    # =============================================================================

    @staticmethod
    def calculate_weighted_confidence(
        individual_scores: list, weights: list = None
    ) -> float:
        """Calculate weighted confidence score with boundary constraints"""
        if not individual_scores:
            return 0.0

        if weights is None:
            weights = [1.0] * len(individual_scores)

        # Adjust weights to match scores length
        if len(weights) != len(individual_scores):
            weights = weights[: len(individual_scores)] + [1.0] * (
                len(individual_scores) - len(weights)
            )

        weighted_sum = sum(
            score * weight for score, weight in zip(individual_scores, weights)
        )
        total_weight = sum(weights)

        return min(1.0, max(0.0, weighted_sum / total_weight))

    @staticmethod
    def calculate_average_confidence(scores: list) -> float:
        """Calculate average confidence from list of scores"""
        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def boost_confidence(base_confidence: float, multiplier: float = None) -> float:
        """Boost confidence with multiplier, capped at 1.0"""
        multiplier = multiplier or MATH.CONFIDENCE_BOOST_MULTIPLIER
        return min(1.0, base_confidence * multiplier)

    # =============================================================================
    # NORMALIZATION CALCULATIONS
    # =============================================================================

    @staticmethod
    def normalize_text_length(text_length: int) -> float:
        """Normalize text length for analysis"""
        return min(1.0, text_length / MATH.TEXT_LENGTH_NORMALIZER)

    @staticmethod
    def normalize_word_count(word_count: int) -> float:
        """Normalize word count for analysis"""
        return min(1.0, word_count / MATH.WORD_COUNT_NORMALIZER)

    @staticmethod
    def normalize_entropy(entropy_score: float) -> float:
        """Normalize entropy score for analysis"""
        return entropy_score / MATH.ENTROPY_NORMALIZER

    @staticmethod
    def normalize_sentence_length(avg_sentence_length: float) -> float:
        """Normalize average sentence length"""
        return min(1.0, avg_sentence_length / MATH.SENTENCE_LENGTH_NORMALIZER)

    @staticmethod
    def normalize_document_count(document_count: int) -> float:
        """Normalize document count for confidence calculation"""
        return min(1.0, document_count / MATH.DOCUMENT_COUNT_NORMALIZER)

    # =============================================================================
    # SIZE AND SCALING CALCULATIONS
    # =============================================================================

    @staticmethod
    def calculate_chunk_overlap(chunk_size: int, overlap_ratio: float = None) -> int:
        """Calculate chunk overlap size"""
        ratio = overlap_ratio or MATH.CHUNK_OVERLAP_RATIO
        return max(50, int(chunk_size * ratio))

    @staticmethod
    def calculate_small_chunk_overlap(optimal_chunk_size: int) -> int:
        """Calculate overlap for small chunks"""
        return max(50, int(optimal_chunk_size * MATH.SMALL_CHUNK_OVERLAP_RATIO))

    @staticmethod
    def scale_batch_size(
        total_items: int, max_batch: int = 10, min_batch: int = 1
    ) -> int:
        """Scale batch size based on total items"""
        return min(max_batch, max(min_batch, total_items // 10))

    @staticmethod
    def scale_entities_per_chunk(
        entity_density: float, max_entities: int = 50, min_entities: int = 5
    ) -> int:
        """Scale max entities per chunk based on density"""
        return min(max_entities, max(min_entities, int(entity_density * 100)))

    # =============================================================================
    # MEMORY AND STORAGE CALCULATIONS
    # =============================================================================

    @staticmethod
    def bytes_to_mb(bytes_value: int) -> float:
        """Convert bytes to megabytes"""
        return bytes_value / (1024 * 1024)

    @staticmethod
    def calculate_projected_usage_mb(current_usage: int, item_size: int) -> float:
        """Calculate projected memory usage in MB"""
        return (current_usage + item_size) / (1024 * 1024)

    @staticmethod
    def estimate_string_size(text: str) -> int:
        """Estimate string size in bytes"""
        return len(text.encode("utf-8"))

    @staticmethod
    def estimate_sample_size(item_length: int, sample_size: int) -> int:
        """Estimate size based on sample for large items"""
        return sample_size * max(1, item_length // 10)

    # =============================================================================
    # STATISTICAL AND DIVERSITY CALCULATIONS
    # =============================================================================

    @staticmethod
    def calculate_vocabulary_diversity(
        vocabulary_size: int, total_tokens: int
    ) -> float:
        """Calculate vocabulary diversity ratio"""
        return vocabulary_size / max(1, total_tokens)

    @staticmethod
    def calculate_pattern_density(pattern_count: int, vocabulary_size: int) -> float:
        """Calculate pattern density ratio"""
        return pattern_count / max(1, vocabulary_size)

    @staticmethod
    def calculate_entity_density(entity_count: int, total_length: int) -> float:
        """Calculate entity density ratio"""
        return entity_count / total_length if total_length > 0 else 0.0

    @staticmethod
    def calculate_graph_density(actual_edges: int, max_edges: int) -> float:
        """Calculate graph density ratio"""
        return actual_edges / max_edges if max_edges > 0 else 0.0

    # =============================================================================
    # TIME AND PERFORMANCE CALCULATIONS
    # =============================================================================

    @staticmethod
    def calculate_total_seconds(start_time, end_time) -> float:
        """Calculate total seconds between datetime objects"""
        return (end_time - start_time).total_seconds()

    @staticmethod
    def calculate_age_hours(timestamp, current_time) -> float:
        """Calculate age in hours from timestamp"""
        return (current_time - timestamp).total_seconds() / 3600

    @staticmethod
    def calculate_average_processing_time(
        total_time: float, item_count: int, max_divisor: int = None
    ) -> float:
        """Calculate average processing time per item"""
        divisor = max_divisor or item_count
        return total_time / max(divisor, 1)

    @staticmethod
    def calculate_exponential_backoff(
        base_delay: float, attempt_count: int, max_delay: float = 30.0
    ) -> float:
        """Calculate exponential backoff delay"""
        return min(max_delay, base_delay * (2**attempt_count))

    # =============================================================================
    # CROSS-MODAL AND AGREEMENT CALCULATIONS
    # =============================================================================

    @staticmethod
    def calculate_cross_modal_agreement(
        active_modalities: int, average_confidence: float
    ) -> float:
        """Calculate cross-modal agreement score"""
        return min(
            1.0,
            active_modalities / MATH.MODAL_AGREEMENT_NORMALIZER * average_confidence,
        )

    @staticmethod
    def calculate_synthesis_score(
        avg_confidence: float,
        cross_modal_agreement: float,
        quality_ratio: float,
        confidence_weight: float = None,
        agreement_weight: float = None,
        quality_weight: float = None,
    ) -> float:
        """Calculate result synthesis score"""
        conf_weight = confidence_weight or MATH.MODALITY_WEIGHT
        agree_weight = agreement_weight or MATH.PERFORMANCE_WEIGHT
        qual_weight = quality_weight or MATH.QUALITY_WEIGHT

        synthesis_score = (
            avg_confidence * conf_weight
            + cross_modal_agreement * agree_weight
            + quality_ratio * qual_weight
        )
        return min(1.0, synthesis_score)

    # =============================================================================
    # THRESHOLD AND BOUNDARY CALCULATIONS
    # =============================================================================

    @staticmethod
    def calculate_quality_threshold(
        base_confidence: float, factor: float = None
    ) -> float:
        """Calculate quality threshold using centralized factor"""
        factor = factor or MATH.QUALITY_BOUND_FACTOR
        return max(
            base_confidence * MATH.MIN_THRESHOLD_FACTOR,
            min(base_confidence, base_confidence * factor),
        )

    @staticmethod
    def calculate_relationship_strength_threshold(base_threshold: float) -> float:
        """Calculate relationship strength threshold"""
        return base_threshold * MATH.RELATIONSHIP_STRENGTH_FACTOR

    @staticmethod
    def calculate_quality_validation_threshold(entity_threshold: float) -> float:
        """Calculate quality validation threshold"""
        return entity_threshold * MATH.QUALITY_VALIDATION_FACTOR

    @staticmethod
    def validate_precision_tolerance(
        total: float, expected: float = 1.0, tolerance: float = None
    ) -> bool:
        """Validate value within precision tolerance"""
        tolerance = tolerance or MATH.PRECISION_TOLERANCE
        return (expected - tolerance) <= total <= (expected + tolerance)


# Global instances for easy access
MATH = MathConstants()
EXPR = MathExpressions()
