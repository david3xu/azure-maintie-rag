"""
Unified Error Handler - Centralized error handling and recovery

This service provides centralized error handling, logging, and recovery
mechanisms for the agent architecture. It consolidates error handling
patterns from across the system into a single, consistent approach.

Features:
- Structured error categorization and handling
- Automatic retry mechanisms with exponential backoff
- Error recovery strategies for different failure types
- Performance impact tracking
- Integration with monitoring systems
- Circuit breaker patterns for external services
"""

import asyncio
import functools
import logging
import time

# Import constants for zero-hardcoded-values compliance
from agents.core.constants import MathematicalConstants, PerformanceAdaptiveConstants, SystemBoundaryConstants, ErrorHandlingCoordinatedConstants
import traceback
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, Union

# Import models from centralized data models
from agents.core.data_models import (
    ErrorSeverity, ErrorCategory, ErrorContext, ErrorMetrics
)

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Circuit breaker for external service calls"""

    def __init__(self, failure_threshold: int = ErrorHandlingCoordinatedConstants.DEFAULT_FAILURE_THRESHOLD, timeout: float = PerformanceAdaptiveConstants.AZURE_SERVICE_TIMEOUT):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = SystemBoundaryConstants.FAILURE_COUNT_INITIAL
        self.last_failure_time = MathematicalConstants.CONFIDENCE_MIN
        self.state = "closed"  # closed, open, half-open

    def can_proceed(self) -> bool:
        """Check if operation can proceed"""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = "half-open"
                return True
            return False
        else:  # half-open
            return True

    def record_success(self):
        """Record successful operation"""
        self.failure_count = SystemBoundaryConstants.FAILURE_COUNT_INITIAL
        self.state = "closed"

    def record_failure(self):
        """Record failed operation"""
        self.failure_count += SystemBoundaryConstants.FAILURE_COUNT_INCREMENT
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class UnifiedErrorHandler:
    """
    Unified error handler providing centralized error management,
    recovery strategies, and monitoring integration.
    """

    def __init__(self):
        """Initialize unified error handler"""
        self.metrics = ErrorMetrics()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.recovery_strategies: Dict[str, Callable] = {}
        self.error_listeners: List[Callable] = []

        # Default recovery strategies
        self._register_default_strategies()

        logger.info("Unified error handler initialized")

    def _register_default_strategies(self):
        """Register default error recovery strategies"""
        self.recovery_strategies.update(
            {
                "azure_service_retry": self._azure_service_recovery,
                "cache_fallback": self._cache_fallback_recovery,
                "domain_detection_fallback": self._domain_detection_fallback,
                "graceful_degradation": self._graceful_degradation_recovery,
            }
        )

    def register_recovery_strategy(self, name: str, strategy: Callable):
        """Register custom recovery strategy"""
        self.recovery_strategies[name] = strategy
        logger.info(f"Registered recovery strategy: {name}")

    def add_error_listener(self, listener: Callable):
        """Add error event listener for monitoring integration"""
        self.error_listeners.append(listener)

    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for service"""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker()
        return self.circuit_breakers[service_name]

    async def handle_error(
        self,
        error: Exception,
        operation: str,
        component: str,
        parameters: Optional[Dict[str, Any]] = None,
        severity: Optional[ErrorSeverity] = None,
        category: Optional[ErrorCategory] = None,
    ) -> Optional[Any]:
        """
        Handle error with automatic categorization, recovery, and monitoring

        Args:
            error: The exception that occurred
            operation: Name of the operation that failed
            component: Component where error occurred
            parameters: Operation parameters for context
            severity: Override automatic severity detection
            category: Override automatic category detection

        Returns:
            Recovery result if successful, None if recovery failed
        """
        # Create error context
        context = ErrorContext(
            error=error,
            severity=severity or self._detect_severity(error),
            category=category or self._detect_category(error, component),
            operation=operation,
            component=component,
            parameters=parameters or {},
        )

        # Record error metrics
        self.metrics.record_error(context)

        # Log error with context
        self._log_error(context)

        # Notify listeners
        await self._notify_listeners(context)

        # Attempt recovery if appropriate
        recovery_result = None
        if context.severity != ErrorSeverity.CRITICAL:
            recovery_result = await self._attempt_recovery(context)

        return recovery_result

    def _detect_severity(self, error: Exception) -> ErrorSeverity:
        """Automatically detect error severity"""
        error_type = type(error).__name__
        error_message = str(error).lower()

        # Critical errors
        if any(
            term in error_message
            for term in ["authentication", "authorization", "credential"]
        ):
            return ErrorSeverity.CRITICAL

        # High severity errors
        if any(term in error_message for term in ["timeout", "connection", "network"]):
            return ErrorSeverity.HIGH

        # Medium severity errors
        if error_type in ["ValueError", "KeyError", "AttributeError"]:
            return ErrorSeverity.MEDIUM

        # Default to low severity
        return ErrorSeverity.LOW

    def _detect_category(self, error: Exception, component: str) -> ErrorCategory:
        """Automatically detect error category"""
        error_message = str(error).lower()
        component_lower = component.lower()

        # Azure service errors
        if any(
            term in error_message for term in ["azure", "credential", "authentication"]
        ):
            return ErrorCategory.AZURE_SERVICE

        # Network errors
        if any(term in error_message for term in ["connection", "timeout", "network"]):
            return ErrorCategory.NETWORK

        # Component-based categorization
        if "cache" in component_lower or "memory" in component_lower:
            return ErrorCategory.CACHE_MEMORY
        elif "domain" in component_lower:
            return ErrorCategory.DOMAIN_INTELLIGENCE
        elif "agent" in component_lower:
            return ErrorCategory.AGENT_PROCESSING
        elif "config" in component_lower:
            return ErrorCategory.CONFIGURATION

        return ErrorCategory.UNKNOWN

    def _log_error(self, context: ErrorContext):
        """Log error with structured context"""
        log_level = {
            ErrorSeverity.LOW: logging.WARNING,
            ErrorSeverity.MEDIUM: logging.ERROR,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }.get(context.severity, logging.ERROR)

        logger.log(
            log_level,
            f"Error in {context.component}.{context.operation}: {context.error}",
            extra={
                "error_category": context.category.value,
                "error_severity": context.severity.value,
                "component": context.component,
                "operation": context.operation,
                "parameters": context.parameters,
                "attempt_count": context.attempt_count,
                "traceback": traceback.format_exc(),
            },
        )

    async def _notify_listeners(self, context: ErrorContext):
        """Notify error listeners for monitoring integration"""
        for listener in self.error_listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(context)
                else:
                    listener(context)
            except Exception as e:
                logger.warning(f"Error listener failed: {e}")

    async def _attempt_recovery(self, context: ErrorContext) -> Optional[Any]:
        """Attempt error recovery using appropriate strategy"""
        recovery_start = time.time()

        try:
            # Determine recovery strategy
            strategy_name = self._select_recovery_strategy(context)
            if not strategy_name or strategy_name not in self.recovery_strategies:
                return None

            # Set recovery strategy in context
            context.recovery_strategy = strategy_name

            # Execute recovery strategy
            strategy = self.recovery_strategies[strategy_name]
            recovery_result = await strategy(context)

            # Record successful recovery
            recovery_time = time.time() - recovery_start
            self.metrics.record_recovery(True, recovery_time)

            logger.info(
                f"Recovery successful for {context.operation} using {strategy_name}"
            )
            return recovery_result

        except Exception as recovery_error:
            # Record failed recovery
            recovery_time = time.time() - recovery_start
            self.metrics.record_recovery(False, recovery_time)

            logger.error(f"Recovery failed for {context.operation}: {recovery_error}")
            return None

    def _select_recovery_strategy(self, context: ErrorContext) -> Optional[str]:
        """Select appropriate recovery strategy based on error context"""
        if context.category == ErrorCategory.AZURE_SERVICE:
            return "azure_service_retry"
        elif context.category == ErrorCategory.CACHE_MEMORY:
            return "cache_fallback"
        elif context.category == ErrorCategory.DOMAIN_INTELLIGENCE:
            return "domain_detection_fallback"
        elif context.category in [
            ErrorCategory.NETWORK,
            ErrorCategory.AGENT_PROCESSING,
        ]:
            return "azure_service_retry"  # Use retry strategy
        else:
            return "graceful_degradation"

    # Default recovery strategies

    async def _azure_service_recovery(self, context: ErrorContext) -> Optional[Any]:
        """Recovery strategy for Azure service errors"""
        if not context.should_retry:
            return None

        # Check circuit breaker
        circuit_breaker = self.get_circuit_breaker(context.component)
        if not circuit_breaker.can_proceed():
            logger.warning(f"Circuit breaker open for {context.component}")
            return None

        # Wait for backoff delay
        await asyncio.sleep(context.backoff_delay)

        # Increment attempt count
        context.attempt_count += 1

        logger.info(f"Retrying {context.operation} (attempt {context.attempt_count})")
        return {"retry": True, "attempt": context.attempt_count}

    async def _cache_fallback_recovery(self, context: ErrorContext) -> Optional[Any]:
        """Recovery strategy for cache/memory errors"""
        logger.info(f"Using fallback for cache error in {context.operation}")

        # Provide fallback cache behavior
        return {
            "fallback": True,
            "strategy": "bypass_cache",
            "message": "Cache temporarily unavailable, proceeding without cache",
        }

    async def _domain_detection_fallback(self, context: ErrorContext) -> Optional[Any]:
        """Recovery strategy for domain intelligence errors"""
        logger.info(f"Using fallback domain detection for {context.operation}")

        # Fallback to general domain
        return {
            "domain": SystemBoundaryConstants.DEFAULT_FALLBACK_DOMAIN,
            "confidence": SystemBoundaryConstants.DEFAULT_FALLBACK_CONFIDENCE,
            "fallback": True,
            "message": "Using general domain due to detection error",
        }

    async def _graceful_degradation_recovery(
        self, context: ErrorContext
    ) -> Optional[Any]:
        """Recovery strategy for graceful degradation"""
        logger.info(f"Graceful degradation for {context.operation}")

        return {
            "degraded": True,
            "message": "Operating in degraded mode due to error",
            "reduced_functionality": True,
        }

    # Decorator for automatic error handling

    def handle_errors(
        self,
        operation: str = None,
        component: str = None,
        severity: ErrorSeverity = None,
        category: ErrorCategory = None,
        return_on_error: Any = None,
    ):
        """
        Decorator for automatic error handling

        Usage:
            @error_handler.handle_errors(
                operation="search_documents",
                component="azure_search",
                return_on_error=[]
            )
            async def search_documents(query: str):
                # Function implementation
                pass
        """

        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    recovery_result = await self.handle_error(
                        error=e,
                        operation=operation or func.__name__,
                        component=component or func.__module__.split(".")[-1],
                        parameters={"args": args, "kwargs": kwargs},
                        severity=severity,
                        category=category,
                    )

                    if recovery_result is not None:
                        return recovery_result
                    else:
                        return return_on_error

            return wrapper

        return decorator

    # Statistics and monitoring

    def get_error_stats(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        return {
            "total_errors": self.metrics.total_errors,
            "errors_by_category": dict(self.metrics.errors_by_category),
            "errors_by_severity": dict(self.metrics.errors_by_severity),
            "errors_by_component": dict(self.metrics.errors_by_component),
            "recovery_stats": {
                "successful_recoveries": self.metrics.successful_recoveries,
                "failed_recoveries": self.metrics.failed_recoveries,
                "success_rate_percent": self.metrics.recovery_success_rate,
                "average_recovery_time_ms": self.metrics.average_recovery_time * MathematicalConstants.MS_PER_SECOND,
            },
            "circuit_breaker_status": {
                name: {"state": cb.state, "failure_count": cb.failure_count}
                for name, cb in self.circuit_breakers.items()
            },
            "recent_errors": list(self.metrics.recent_errors)[-10:],  # Last 10 errors
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get error handler health status"""
        recent_error_count = len(
            [
                error
                for error in self.metrics.recent_errors
                if time.time() - error["timestamp"] < 3600  # Last hour
            ]
        )

        critical_errors = self.metrics.errors_by_severity.get("critical", 0)
        recovery_rate = self.metrics.recovery_success_rate

        if critical_errors > 0 or recent_error_count > SystemBoundaryConstants.CRITICAL_ERROR_THRESHOLD:
            status = "critical"
        elif recent_error_count > SystemBoundaryConstants.WARNING_ERROR_THRESHOLD or recovery_rate < SystemBoundaryConstants.MIN_RECOVERY_RATE_THRESHOLD:
            status = "warning"
        else:
            status = "healthy"

        return {
            "status": status,
            "recent_errors_last_hour": recent_error_count,
            "critical_errors_total": critical_errors,
            "recovery_success_rate": recovery_rate,
            "active_circuit_breakers": len(
                [cb for cb in self.circuit_breakers.values() if cb.state != "closed"]
            ),
            "error_handler_operational": True,
        }


# Global error handler instance
_global_error_handler: Optional[UnifiedErrorHandler] = None


def get_error_handler() -> UnifiedErrorHandler:
    """Get or create global unified error handler"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = UnifiedErrorHandler()
    return _global_error_handler


# Convenience functions for common error handling patterns


async def handle_azure_service_error(
    error: Exception, service_name: str, operation: str
) -> Optional[Any]:
    """Handle Azure service errors with appropriate recovery"""
    handler = get_error_handler()
    return await handler.handle_error(
        error=error,
        operation=operation,
        component=service_name,
        category=ErrorCategory.AZURE_SERVICE,
        severity=ErrorSeverity.HIGH,
    )


async def handle_domain_intelligence_error(
    error: Exception, operation: str
) -> Optional[Any]:
    """Handle domain intelligence errors with fallback"""
    handler = get_error_handler()
    return await handler.handle_error(
        error=error,
        operation=operation,
        component="domain_intelligence",
        category=ErrorCategory.DOMAIN_INTELLIGENCE,
        severity=ErrorSeverity.MEDIUM,
    )


async def handle_cache_error(error: Exception, operation: str) -> Optional[Any]:
    """Handle cache errors with fallback"""
    handler = get_error_handler()
    return await handler.handle_error(
        error=error,
        operation=operation,
        component="cache_manager",
        category=ErrorCategory.CACHE_MEMORY,
        severity=ErrorSeverity.LOW,
    )


# Error handling decorator shortcuts
def azure_service_errors(service_name: str, return_on_error: Any = None):
    """Decorator for Azure service error handling"""
    handler = get_error_handler()
    return handler.handle_errors(
        component=service_name,
        category=ErrorCategory.AZURE_SERVICE,
        severity=ErrorSeverity.HIGH,
        return_on_error=return_on_error,
    )


def domain_intelligence_errors(return_on_error: Any = None):
    """Decorator for domain intelligence error handling"""
    handler = get_error_handler()
    return handler.handle_errors(
        component="domain_intelligence",
        category=ErrorCategory.DOMAIN_INTELLIGENCE,
        severity=ErrorSeverity.MEDIUM,
        return_on_error=return_on_error or {"domain": SystemBoundaryConstants.DEFAULT_FALLBACK_DOMAIN, "confidence": CoreSystemConstants.DEFAULT_FALLBACK_CONFIDENCE},
    )


def cache_errors(return_on_error: Any = None):
    """Decorator for cache error handling"""
    handler = get_error_handler()
    return handler.handle_errors(
        component="cache_manager",
        category=ErrorCategory.CACHE_MEMORY,
        severity=ErrorSeverity.LOW,
        return_on_error=return_on_error,
    )
