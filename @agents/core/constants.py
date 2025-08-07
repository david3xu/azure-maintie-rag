"""
Core Constants - Simplified
===========================

Essential constants for the PydanticAI agent system.
Only includes the minimal constants needed for operation.
"""

# Azure Service Constants
class AzureConstants:
    """Azure service configuration constants"""
    DEFAULT_API_VERSION = "2024-10-21"
    DEFAULT_TIMEOUT_SECONDS = 30
    MAX_RETRIES = 3
    
# Processing Constants  
class ProcessingConstants:
    """Core processing parameters"""
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200
    MAX_CONCURRENT_REQUESTS = 10
    
# Confidence Thresholds
class ConfidenceConstants:
    """Confidence scoring thresholds"""
    MIN_CONFIDENCE = 0.0
    DEFAULT_THRESHOLD = 0.8
    HIGH_CONFIDENCE = 0.9
    MAX_CONFIDENCE = 1.0

# Export all constants
__all__ = ["AzureConstants", "ProcessingConstants", "ConfidenceConstants"]