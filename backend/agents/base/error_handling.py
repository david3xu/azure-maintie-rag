"""
Simplified Error Handling System for PydanticAI Agent

This module provides a simplified error handling interface that maintains
resilience patterns while reducing complexity from 10 categories to 3.
"""

from .simple_error_handler import (
    SimpleErrorHandler, 
    ErrorType, 
    ErrorContext, 
    ErrorRecord,
    get_error_handler as get_simple_error_handler,
    resilient_operation
)
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Backward compatibility - map old enums to new simplified types
class ErrorSeverity(Enum):
    """Deprecated: Use ErrorType instead"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Deprecated: Use ErrorType instead"""
    AZURE_SERVICE = "azure_service"
    TOOL_EXECUTION = "tool_execution"
    VALIDATION = "validation" 
    TIMEOUT = "timeout"
    MEMORY = "memory"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    DATA_CORRUPTION = "data_corruption"
    UNKNOWN = "unknown"


class CircuitBreakerState(Enum):
    """Backward compatibility"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# Backward compatibility class
class ErrorHandler(SimpleErrorHandler):
    """
    Backward compatibility wrapper for SimpleErrorHandler.
    
    Maintains the same interface but uses simplified 3-category system internally.
    """
    
    def categorize_error(self, error: Exception, context: ErrorContext) -> ErrorCategory:
        """Deprecated method - maps to simplified error types"""
        error_type = self.classify_error(error, context)
        
        # Map simplified types back to old categories for compatibility
        if error_type == ErrorType.CRITICAL:
            return ErrorCategory.AZURE_SERVICE
        elif error_type == ErrorType.PERMANENT:
            return ErrorCategory.VALIDATION
        else:  # TRANSIENT
            return ErrorCategory.TIMEOUT
    
    def determine_severity(self, error: Exception, context: ErrorContext, category: ErrorCategory) -> ErrorSeverity:
        """Deprecated method - maps to simplified error types"""
        error_type = self.classify_error(error, context)
        
        if error_type == ErrorType.CRITICAL:
            return ErrorSeverity.CRITICAL
        elif error_type == ErrorType.PERMANENT:
            return ErrorSeverity.HIGH
        else:  # TRANSIENT
            return ErrorSeverity.MEDIUM
    
    def get_circuit_breaker(self, operation: str):
        """Backward compatibility method"""
        class CircuitBreakerWrapper:
            def __init__(self, handler, operation):
                self.handler = handler
                self.operation = operation
            
            def should_allow_request(self) -> bool:
                return not self.handler._is_circuit_breaker_open(self.operation)
            
            def record_success(self):
                self.handler.record_success(self.operation)
            
            def record_failure(self):
                self.handler.record_failure(self.operation)
        
        return CircuitBreakerWrapper(self, operation)


# Keep the old interface working by delegating to simplified handler
_global_error_handler = None

def get_error_handler() -> ErrorHandler:
    """Get or create global error handler instance (backward compatible)"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


# Export the new resilient_operation decorator for easier usage
__all__ = [
    'ErrorHandler', 'ErrorType', 'ErrorContext', 'ErrorRecord',
    'ErrorSeverity', 'ErrorCategory', 'CircuitBreakerState',
    'get_error_handler', 'resilient_operation'
]