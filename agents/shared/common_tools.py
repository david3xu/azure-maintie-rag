"""
Common Utility Functions

Truly shared utility functions that are used across multiple agents
but don't require agent context or toolset registration.
"""

import time
import hashlib
import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone


def generate_cache_key(*args: Any) -> str:
    """
    Generate a consistent cache key from input arguments.
    
    Args:
        *args: Arguments to generate cache key from
        
    Returns:
        str: SHA256 hash as cache key
    """
    # Convert arguments to string representation
    key_parts = []
    for arg in args:
        if isinstance(arg, (dict, list)):
            key_parts.append(json.dumps(arg, sort_keys=True))
        else:
            key_parts.append(str(arg))
    
    # Create combined key
    combined_key = "|".join(key_parts)
    
    # Generate hash
    return hashlib.sha256(combined_key.encode()).hexdigest()


def get_current_timestamp() -> str:
    """
    Get current UTC timestamp in ISO format.
    
    Returns:
        str: Current timestamp in ISO format
    """
    return datetime.now(timezone.utc).isoformat()


def calculate_confidence_score(
    individual_scores: List[float], 
    weights: Optional[List[float]] = None
) -> float:
    """
    Calculate weighted confidence score from individual scores.
    
    Args:
        individual_scores: List of individual confidence scores (0.0-1.0)
        weights: Optional weights for each score (defaults to equal weights)
        
    Returns:
        float: Weighted confidence score (0.0-1.0)
    """
    if not individual_scores:
        return 0.0
    
    # Default to equal weights if not provided
    if weights is None:
        weights = [1.0] * len(individual_scores)
    
    # Ensure weights and scores have same length
    if len(weights) != len(individual_scores):
        weights = weights[:len(individual_scores)] + [1.0] * (len(individual_scores) - len(weights))
    
    # Calculate weighted average
    weighted_sum = sum(score * weight for score, weight in zip(individual_scores, weights))
    total_weight = sum(weights)
    
    if total_weight == 0:
        return 0.0
    
    return min(1.0, max(0.0, weighted_sum / total_weight))


def format_processing_time(start_time: float, end_time: Optional[float] = None) -> Dict[str, Any]:
    """
    Format processing time information.
    
    Args:
        start_time: Start time (from time.time())
        end_time: End time (defaults to current time)
        
    Returns:
        Dict with processing time information
    """
    if end_time is None:
        end_time = time.time()
    
    processing_time = end_time - start_time
    
    return {
        "processing_time_seconds": round(processing_time, 3),
        "processing_time_ms": round(processing_time * 1000, 1),
        "start_timestamp": datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
        "end_timestamp": datetime.fromtimestamp(end_time, timezone.utc).isoformat(),
        "performance_category": (
            "excellent" if processing_time < 1.0 else
            "good" if processing_time < 3.0 else
            "acceptable" if processing_time < 10.0 else
            "slow"
        )
    }


def validate_confidence_threshold(value: float, min_threshold: float = 0.0, max_threshold: float = 1.0) -> bool:
    """
    Validate that a confidence value is within acceptable thresholds.
    
    Args:
        value: Confidence value to validate
        min_threshold: Minimum acceptable threshold
        max_threshold: Maximum acceptable threshold
        
    Returns:
        bool: True if value is within thresholds
    """
    return min_threshold <= value <= max_threshold


def extract_domain_from_path(file_path: str) -> str:
    """
    Extract domain name from file path.
    
    Args:
        file_path: File or directory path
        
    Returns:
        str: Extracted domain name
    """
    import os
    
    # Get the directory name
    dir_name = os.path.basename(os.path.dirname(file_path)) if os.path.isfile(file_path) else os.path.basename(file_path)
    
    # Convert to standard domain format
    domain_name = dir_name.lower().replace('-', '_').replace(' ', '_')
    
    # Remove common prefixes/suffixes
    for prefix in ['data_', 'raw_', 'processed_']:
        if domain_name.startswith(prefix):
            domain_name = domain_name[len(prefix):]
    
    for suffix in ['_data', '_raw', '_processed']:
        if domain_name.endswith(suffix):
            domain_name = domain_name[:-len(suffix)]
    
    return domain_name or "general"


def merge_dictionaries(*dicts: Dict[str, Any], deep_merge: bool = True) -> Dict[str, Any]:
    """
    Merge multiple dictionaries with optional deep merging.
    
    Args:
        *dicts: Dictionaries to merge
        deep_merge: Whether to perform deep merge for nested dicts
        
    Returns:
        Dict: Merged dictionary
    """
    result = {}
    
    for d in dicts:
        if not isinstance(d, dict):
            continue
            
        for key, value in d.items():
            if key not in result:
                result[key] = value
            elif deep_merge and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dictionaries(result[key], value, deep_merge=True)
            else:
                result[key] = value
    
    return result


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length (including suffix)
        suffix: Suffix to append if truncated
        
    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
    
    truncated_length = max_length - len(suffix)
    if truncated_length <= 0:
        return suffix[:max_length]
    
    return text[:truncated_length] + suffix


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely parse JSON string with fallback default.
    
    Args:
        json_str: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def normalize_entity_name(name: str) -> str:
    """
    Normalize entity name for consistent comparison.
    
    Args:
        name: Entity name to normalize
        
    Returns:
        str: Normalized entity name
    """
    # Convert to lowercase and strip whitespace
    normalized = name.strip().lower()
    
    # Replace multiple spaces with single space
    import re
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Remove special characters (keep alphanumeric, spaces, and common separators)
    normalized = re.sub(r'[^\w\s\-_.]', '', normalized)
    
    return normalized


# Export all utility functions
__all__ = [
    "generate_cache_key",
    "get_current_timestamp",
    "calculate_confidence_score",
    "format_processing_time",
    "validate_confidence_threshold",
    "extract_domain_from_path",
    "merge_dictionaries",
    "truncate_text",
    "safe_json_loads",
    "normalize_entity_name",
]