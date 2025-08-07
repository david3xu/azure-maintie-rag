"""
Shared Utilities - Simplified
=============================

Common utility functions used across agents.
Only includes essential utilities without complexity.
"""

import time
from typing import Dict, Any
from pathlib import Path

def get_current_timestamp() -> str:
    """Get current timestamp as ISO string"""
    return time.strftime('%Y-%m-%dT%H:%M:%S')

def calculate_processing_time(start_time: float) -> float:
    """Calculate processing time from start timestamp"""
    return time.time() - start_time

def validate_file_path(file_path: str) -> bool:
    """Validate that a file path exists and is readable"""
    try:
        path = Path(file_path)
        return path.exists() and path.is_file()
    except Exception:
        return False

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers with default for division by zero"""
    return numerator / denominator if denominator != 0 else default

def merge_metadata(base: Dict[str, Any], additional: Dict[str, Any]) -> Dict[str, Any]:
    """Safely merge metadata dictionaries"""
    result = base.copy()
    result.update(additional)
    return result

# Export utilities
__all__ = [
    "get_current_timestamp",
    "calculate_processing_time", 
    "validate_file_path",
    "safe_divide",
    "merge_metadata"
]