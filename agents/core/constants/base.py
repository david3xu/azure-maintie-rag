"""
Mathematical Base Constants - Foundation Layer
=============================================

This module contains core mathematical constants and scaling factors that serve as the
foundation for all other constants in the system. These are the most stable constants
and should never be auto-generated as they represent fundamental mathematical relationships.

Organization:
- Mathematical conversion factors (bytes, time, percentages)
- Base system values (timeouts, counts, sizes, thresholds)
- Scaling factors to derive related constants and reduce duplication

These constants support the zero-hardcoded-values philosophy by providing a mathematical
foundation that other constants derive from.
"""

# Remove unused imports


class MathematicalConstants:
    """Core mathematical constants used throughout the system"""

    # Byte conversion factors
    BYTES_PER_KB = 1024
    BYTES_PER_MB = BYTES_PER_KB * 1024  # 1,048,576
    BYTES_PER_GB = BYTES_PER_MB * 1024  # 1,073,741,824

    # Time conversion factors
    MS_PER_SECOND = 1000.0
    SECONDS_PER_MINUTE = 60.0

    # Percentage and scoring
    PERCENTAGE_MULTIPLIER = 100.0
    CONFIDENCE_MIN = 0.0
    CONFIDENCE_MAX = 1.0

    # Base scaling factors to reduce duplication
    BASE_TIMEOUT = 30  # Base timeout in seconds
    BASE_RESULT_COUNT = 10  # Base result count for searches
    BASE_CHUNK_SIZE = 500  # Base chunk size in characters
    BASE_CONFIDENCE = 0.7  # Base confidence threshold


class MathematicalFoundationConstants:
    """Enhanced mathematical foundation constants for model integration"""

    # Perfect mathematical values
    PERFECT_SCORE = 1.0
    ZERO_THRESHOLD = 0.0
    MILLISECONDS_PER_SECOND = 1000.0

    # Exponential backoff
    EXPONENTIAL_BACKOFF_BASE = 2.0
    BASE_DELAY_SECONDS = 1.0

    # Random seed for reproducibility
    RANDOM_SEED = 42


class BaseScalingFactors:
    """Scaling factors to derive related constants from base values"""

    # Timeout scaling
    AZURE_TIMEOUT_FACTOR = 2.0  # 60 = 30 * 2
    MAX_TIMEOUT_FACTOR = 20.0  # 600 = 30 * 20

    # Result count scaling
    SEARCH_RESULT_FACTOR = 2.0  # 20 = 10 * 2
    ENTITY_RESULT_FACTOR = 5.0  # 50 = 10 * 5

    # Chunk size scaling
    STANDARD_CHUNK_FACTOR = 2.0  # 1000 = 500 * 2
    LARGE_CHUNK_FACTOR = 4.0  # 2000 = 500 * 4

    # Confidence scaling
    HIGH_CONFIDENCE_OFFSET = 0.1  # 0.8 = 0.7 + 0.1
    STATISTICAL_CONFIDENCE_OFFSET = 0.05  # 0.75 = 0.7 + 0.05


# Utility functions for mathematical operations
def derive_timeout(base_timeout: int, factor: float) -> int:
    """Derive timeout values from base timeout and scaling factor"""
    return int(base_timeout * factor)


def derive_chunk_size(base_size: int, factor: float) -> int:
    """Derive chunk sizes from base size and scaling factor"""
    return int(base_size * factor)


def derive_confidence(base_confidence: float, offset: float) -> float:
    """Derive confidence thresholds from base confidence and offset"""
    return base_confidence + offset


# Export all constants for easy access
__all__ = [
    "MathematicalConstants",
    "BaseScalingFactors",
    "derive_timeout",
    "derive_chunk_size",
    "derive_confidence",
]
