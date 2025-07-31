"""
Simplified Error Handling System for PydanticAI Agent

This module replaces the complex 10-category error classification 
with 3 essential categories while maintaining resilience patterns.
"""

import asyncio
import logging
import time
import traceback
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import functools

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Simplified error categories"""
    TRANSIENT = "transient"     # Retry automatically (network, timeout, rate limits)
    PERMANENT = "permanent"     # Don't retry, log and fail (validation, auth failures)
    CRITICAL = "critical"       # Immediate escalation (system failures)


@dataclass
class ErrorContext:
    """Essential error context information"""
    operation: str
    parameters: Dict[str, Any]
    retry_count: int = 0
    correlation_id: Optional[str] = None
    start_time: float = None
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = time.time()


@dataclass
class ErrorRecord:
    """Simplified error record"""
    error_id: str
    timestamp: float
    error_type: ErrorType
    error_message: str
    context: ErrorContext
    resolved: bool = False


class SimpleErrorHandler:
    """
    Simplified error handler with essential resilience patterns.
    
    Reduces complexity from 10 error categories to 3 essential types
    while maintaining automatic recovery and circuit breaker functionality.
    """
    
    def __init__(self):
        self.error_records = []
        self.circuit_breakers = {}  # operation -> failure_count
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 60.0  # seconds
        
        logger.info("Simple error handler initialized")
    
    def classify_error(self, error: Exception, context: ErrorContext) -> ErrorType:
        """Classify error into one of 3 essential types"""
        error_message = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Critical errors - system failures requiring immediate attention
        if any(keyword in error_message for keyword in ['memory', 'system', 'critical']):
            return ErrorType.CRITICAL
        
        # Permanent errors - don't retry
        if any(keyword in error_message for keyword in [
            'validation', 'auth', '401', '403', 'permission', 'invalid'
        ]):
            return ErrorType.PERMANENT
            
        # Transient errors - safe to retry
        # Default assumption: most errors are transient and worth retrying
        return ErrorType.TRANSIENT
    
    async def handle_error(
        self, 
        error: Exception, 
        context: ErrorContext,
        max_retries: int = 3,
        apply_backoff: bool = True
    ) -> Dict[str, Any]:
        """
        Handle error with simplified recovery logic
        
        Returns:
            Dict with recovery status and retry recommendation
        """
        error_id = f"{context.operation}_{int(time.time() * 1000)}"
        error_type = self.classify_error(error, context)
        
        # Create error record
        error_record = ErrorRecord(
            error_id=error_id,
            timestamp=time.time(),
            error_type=error_type,
            error_message=str(error),
            context=context
        )
        
        self.error_records.append(error_record)
        
        # Log with appropriate level
        if error_type == ErrorType.CRITICAL:
            logger.critical(f"CRITICAL error in {context.operation}: {error}")
        elif error_type == ErrorType.PERMANENT:
            logger.error(f"PERMANENT error in {context.operation}: {error}")
        else:  # TRANSIENT
            logger.warning(f"TRANSIENT error in {context.operation}: {error} (retry {context.retry_count})")
        
        # Determine recovery strategy
        should_retry = self._should_retry(error_type, context, max_retries)
        recovery_successful = False
        
        if should_retry and error_type == ErrorType.TRANSIENT and apply_backoff:
            # Simple backoff for transient errors (optional for performance testing)
            await self._apply_backoff(context.retry_count)
            recovery_successful = True
        elif should_retry:
            # Mark as recoverable but don't apply backoff
            recovery_successful = True
        
        return {
            'error_id': error_id,
            'error_type': error_type.value,
            'should_retry': should_retry,
            'recovery_successful': recovery_successful,
            'recommendations': self._get_recommendations(error_type)
        }
    
    def _should_retry(self, error_type: ErrorType, context: ErrorContext, max_retries: int) -> bool:
        """Determine if operation should be retried"""
        
        # Never retry permanent or critical errors
        if error_type in [ErrorType.PERMANENT, ErrorType.CRITICAL]:
            return False
        
        # Check retry limit
        if context.retry_count >= max_retries:
            return False
        
        # Check circuit breaker
        if self._is_circuit_breaker_open(context.operation):
            return False
        
        return True
    
    def _is_circuit_breaker_open(self, operation: str) -> bool:
        """Simple circuit breaker check"""
        if operation not in self.circuit_breakers:
            self.circuit_breakers[operation] = {'failures': 0, 'last_failure': 0}
        
        cb = self.circuit_breakers[operation]
        
        # Reset if timeout expired
        if time.time() - cb['last_failure'] > self.circuit_breaker_timeout:
            cb['failures'] = 0
        
        return cb['failures'] >= self.circuit_breaker_threshold
    
    def record_success(self, operation: str):
        """Record successful operation (resets circuit breaker)"""
        if operation in self.circuit_breakers:
            self.circuit_breakers[operation]['failures'] = 0
    
    def record_failure(self, operation: str):
        """Record failed operation (increments circuit breaker)"""
        if operation not in self.circuit_breakers:
            self.circuit_breakers[operation] = {'failures': 0, 'last_failure': 0}
        
        cb = self.circuit_breakers[operation]
        cb['failures'] += 1
        cb['last_failure'] = time.time()
    
    async def _apply_backoff(self, retry_count: int):
        """Apply exponential backoff with jitter"""
        import random
        base_delay = min(60, 2 ** retry_count)  # Cap at 60 seconds
        jitter = random.uniform(0.1, 0.3) * base_delay
        delay = base_delay + jitter
        
        logger.debug(f"Applying backoff delay: {delay:.2f}s")
        await asyncio.sleep(delay)
    
    def _get_recommendations(self, error_type: ErrorType) -> list:
        """Get actionable recommendations based on error type"""
        if error_type == ErrorType.CRITICAL:
            return [
                "Check system resources and health",
                "Review system logs for root cause",
                "Consider immediate escalation"
            ]
        elif error_type == ErrorType.PERMANENT:
            return [
                "Verify input parameters and configuration",
                "Check authentication and permissions",
                "Review API documentation"
            ]
        else:  # TRANSIENT
            return [
                "Will retry automatically with backoff",
                "Monitor for patterns if recurring",
                "Consider rate limiting if needed"
            ]
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get simplified error statistics"""
        now = time.time()
        recent_errors = [e for e in self.error_records if now - e.timestamp < 3600]  # Last hour
        
        stats = {
            'total_errors': len(self.error_records),
            'recent_errors': len(recent_errors),
            'error_rate_per_hour': len(recent_errors),
            'errors_by_type': {},
            'circuit_breaker_states': {}
        }
        
        # Count by error type
        for error in recent_errors:
            error_type = error.error_type.value
            stats['errors_by_type'][error_type] = stats['errors_by_type'].get(error_type, 0) + 1
        
        # Circuit breaker states
        for operation, cb in self.circuit_breakers.items():
            is_open = cb['failures'] >= self.circuit_breaker_threshold
            stats['circuit_breaker_states'][operation] = 'open' if is_open else 'closed'
        
        return stats


# Global error handler instance
_global_error_handler: Optional[SimpleErrorHandler] = None


def get_error_handler() -> SimpleErrorHandler:
    """Get or create global error handler instance"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = SimpleErrorHandler()
    return _global_error_handler


def resilient_operation(
    operation_name: str,
    max_retries: int = 3,
    timeout: float = 30.0
):
    """
    Simplified decorator for resilient operations
    
    Usage:
        @resilient_operation("tri_modal_search", max_retries=3, timeout=30.0)
        async def search_function(query: str):
            # Function implementation
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            error_handler = get_error_handler()
            
            context = ErrorContext(
                operation=operation_name,
                parameters=kwargs,
                correlation_id=kwargs.get('correlation_id')
            )
            
            for attempt in range(max_retries + 1):
                context.retry_count = attempt
                
                # Check circuit breaker
                if error_handler._is_circuit_breaker_open(operation_name):
                    raise Exception(f"Circuit breaker open for {operation_name}")
                
                try:
                    # Execute with timeout
                    result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                    error_handler.record_success(operation_name)
                    return result
                    
                except Exception as e:
                    error_handler.record_failure(operation_name)
                    
                    if attempt == max_retries:
                        # Final attempt failed
                        await error_handler.handle_error(e, context, max_retries)
                        raise
                    
                    # Handle error and check if we should retry
                    error_result = await error_handler.handle_error(e, context, max_retries)
                    if not error_result['should_retry']:
                        raise
            
        return wrapper
    return decorator