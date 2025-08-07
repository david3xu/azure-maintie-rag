"""
Security and Validation Constants
=================================

This module contains security-related constants and data structure validation constants.
Security constants are mostly static for consistency, while some validation constants
could be learned from system behavior.

Key Areas:
1. Security Constants - cryptographic standards and access levels
2. Data Model Constants - required keys and structure definitions
3. Validation and Thresholds - error handling and quality gates

AUTO-GENERATION POTENTIAL:
- SecurityConstants: LOW (security requires consistency)
- DataModelConstants: LOW (structural consistency requirements)
- Various validation thresholds: MEDIUM (could adapt to system patterns)
"""


class SecurityConstants:
    """Security constants - mostly static for consistency"""

    # AUTO-GENERATION POTENTIAL: LOW (security requires consistency)

    # Cryptographic Standards - STATIC for security consistency
    DEFAULT_HASH_ALGORITHM = "sha256"  # STATIC: security standard
    HASH_ENCODING = "utf-8"  # STATIC: encoding standard
    CACHE_KEY_SEPARATOR = "|"  # STATIC: consistent key format
    JSON_SORT_KEYS = True  # STATIC: consistent serialization


class DataModelConstants:
    """Required keys and structure definitions for centralized data models"""

    # AUTO-GENERATION POTENTIAL: LOW (structural consistency requirements)
    # These maintain API consistency and data structure integrity
    pass  # Currently empty but reserved for future data model validation keys


# Export all constants
__all__ = [
    "SecurityConstants",
    "DataModelConstants",
]
