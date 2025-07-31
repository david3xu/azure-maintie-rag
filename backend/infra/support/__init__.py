"""
Core Infrastructure Support Services

This module contains infrastructure support services that were moved from the
services layer to properly align with layer boundaries.

These services handle technical infrastructure concerns:
- Data operations and management
- System cleanup and maintenance  
- Performance monitoring and optimization
- Infrastructure health checks

They do NOT contain business logic - only technical infrastructure support.
"""

# Import infrastructure support clients and managers
from .data_client import DataService  # Keeping class name for backward compatibility
from .cleanup_manager import CleanupService  # Keeping class name for backward compatibility
from .performance_manager import PerformanceService  # Keeping class name for backward compatibility

__all__ = [
    # Infrastructure support services (backward compatible class names)
    'DataService',
    'CleanupService', 
    'PerformanceService'
]