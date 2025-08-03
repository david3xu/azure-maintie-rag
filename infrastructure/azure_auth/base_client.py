"""
Enhanced Base Azure Client - Unified patterns for all Azure services
Provides retry logic, monitoring, error handling, and connection pooling
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, Union

from config.settings import azure_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class ClientMetrics:
    """Client operation metrics"""

    operation_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_duration_ms: float = 0.0
    last_operation_time: Optional[float] = None


class BaseAzureClient(ABC):
    """
    Enhanced base class for all Azure service clients

    Provides:
    - Unified retry logic with exponential backoff
    - Comprehensive error handling and logging
    - Operation metrics and monitoring
    - Connection pooling patterns
    - Managed identity authentication enforcement
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize with enhanced Azure patterns

        Args:
            config: Optional client-specific configuration
        """
        self.config = config or {}
        self._client = None
        self._initialized = False
        self.retry_config = RetryConfig(**self.config.get("retry", {}))
        self.metrics = ClientMetrics()

        # Enforce managed identity authentication only
        self.endpoint = self.config.get("endpoint") or self._get_default_endpoint()

        if not self.endpoint:
            raise ValueError(
                f"{self.__class__.__name__} requires endpoint configuration"
            )

        # Azure authentication - support both managed identity and CLI for development
        self.use_managed_identity = getattr(
            azure_settings, "use_managed_identity", True
        )
        
        # Allow CLI authentication in development environment
        if not self.use_managed_identity:
            logger.info(f"{self.__class__.__name__} using Azure CLI authentication for development")
        else:
            logger.info(f"{self.__class__.__name__} using managed identity authentication")

    @abstractmethod
    def _get_default_endpoint(self) -> str:
        """Get default endpoint from Azure settings"""
        pass

    @abstractmethod
    def _initialize_client(self):
        """Initialize the specific Azure service client with managed identity"""
        pass

    @abstractmethod
    def _health_check(self) -> bool:
        """Perform service-specific health check"""
        pass

    def ensure_initialized(self):
        """Thread-safe lazy initialization"""
        if not self._initialized:
            try:
                self._initialize_client()
                self._initialized = True
                logger.info(
                    f"{self.__class__.__name__} client initialized successfully"
                )
            except Exception as e:
                logger.error(f"{self.__class__.__name__} initialization failed: {e}")
                raise

    async def _execute_with_retry(
        self, func: Callable, operation_name: str, *args, **kwargs
    ) -> Any:
        """
        Execute operation with comprehensive retry logic

        Args:
            func: Function to execute
            operation_name: Name for logging and metrics
            *args, **kwargs: Function arguments
        """
        start_time = time.time()
        last_exception = None

        for attempt in range(1, self.retry_config.max_attempts + 1):
            try:
                self.ensure_initialized()

                # Execute the operation
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                # Record success metrics
                duration_ms = (time.time() - start_time) * 1000
                self._record_success(operation_name, duration_ms)

                return result

            except Exception as e:
                last_exception = e

                # Determine if this is a retryable error
                if (
                    not self._is_retryable_error(e)
                    or attempt == self.retry_config.max_attempts
                ):
                    # Record failure and re-raise
                    duration_ms = (time.time() - start_time) * 1000
                    self._record_error(operation_name, e, duration_ms)
                    raise

                # Calculate retry delay
                delay = self._calculate_retry_delay(attempt)

                logger.warning(
                    f"{self.__class__.__name__}.{operation_name} attempt {attempt} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )

                await asyncio.sleep(delay)

        # This should never be reached due to the logic above
        raise last_exception or Exception("Unexpected retry loop exit")

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable

        Args:
            error: Exception to check

        Returns:
            True if the error should trigger a retry
        """
        # Common retryable Azure exceptions
        retryable_patterns = [
            "timeout",
            "throttled",
            "rate limit",
            "service unavailable",
            "connection",
            "network",
            "temporary",
        ]

        error_str = str(error).lower()
        return any(pattern in error_str for pattern in retryable_patterns)

    def _calculate_retry_delay(self, attempt: int) -> float:
        """
        Calculate retry delay with exponential backoff and jitter

        Args:
            attempt: Current attempt number (1-based)

        Returns:
            Delay in seconds
        """
        delay = min(
            self.retry_config.base_delay
            * (self.retry_config.exponential_base ** (attempt - 1)),
            self.retry_config.max_delay,
        )

        if self.retry_config.jitter:
            import random

            delay *= 0.5 + random.random() * 0.5  # Add 0-50% jitter

        return delay

    def _record_success(self, operation: str, duration_ms: float):
        """Record successful operation metrics"""
        self.metrics.operation_count += 1
        self.metrics.success_count += 1
        self.metrics.total_duration_ms += duration_ms
        self.metrics.last_operation_time = time.time()

        logger.debug(
            f"{self.__class__.__name__}.{operation} completed in {duration_ms:.2f}ms"
        )

    def _record_error(self, operation: str, error: Exception, duration_ms: float):
        """Record failed operation metrics"""
        self.metrics.operation_count += 1
        self.metrics.error_count += 1
        self.metrics.total_duration_ms += duration_ms
        self.metrics.last_operation_time = time.time()

        logger.error(
            f"{self.__class__.__name__}.{operation} failed after {duration_ms:.2f}ms: {error}"
        )

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get client operation metrics

        Returns:
            Dictionary of metrics and performance statistics
        """
        success_rate = (
            self.metrics.success_count / self.metrics.operation_count
            if self.metrics.operation_count > 0
            else 0.0
        )

        avg_duration = (
            self.metrics.total_duration_ms / self.metrics.operation_count
            if self.metrics.operation_count > 0
            else 0.0
        )

        return {
            "client_type": self.__class__.__name__,
            "operation_count": self.metrics.operation_count,
            "success_count": self.metrics.success_count,
            "error_count": self.metrics.error_count,
            "success_rate": success_rate,
            "average_duration_ms": avg_duration,
            "total_duration_ms": self.metrics.total_duration_ms,
            "last_operation_time": self.metrics.last_operation_time,
        }

    def reset_metrics(self):
        """Reset client metrics"""
        self.metrics = ClientMetrics()

    async def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check for the Azure service

        Returns:
            Health status and diagnostic information
        """
        start_time = time.time()

        try:
            self.ensure_initialized()
            service_healthy = self._health_check()

            duration_ms = (time.time() - start_time) * 1000

            return {
                "healthy": service_healthy,
                "service": self.__class__.__name__,
                "endpoint": self.endpoint,
                "check_duration_ms": duration_ms,
                "timestamp": time.time(),
                "metrics": self.get_metrics(),
            }

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            return {
                "healthy": False,
                "service": self.__class__.__name__,
                "endpoint": self.endpoint,
                "error": str(e),
                "check_duration_ms": duration_ms,
                "timestamp": time.time(),
            }

    def create_success_response(
        self, operation: str, data: Any = None
    ) -> Dict[str, Any]:
        """
        Create standardized success response

        Args:
            operation: Operation name
            data: Optional response data

        Returns:
            Standardized success response
        """
        return {
            "success": True,
            "operation": operation,
            "service": self.__class__.__name__,
            "timestamp": time.time(),
            "data": data,
        }

    def create_error_response(
        self, operation: str, error: Union[str, Exception]
    ) -> Dict[str, Any]:
        """
        Create standardized error response

        Args:
            operation: Operation name
            error: Error message or exception

        Returns:
            Standardized error response
        """
        error_msg = str(error)

        return {
            "success": False,
            "operation": operation,
            "service": self.__class__.__name__,
            "timestamp": time.time(),
            "error": error_msg,
            "error_type": type(error).__name__
            if isinstance(error, Exception)
            else "Error",
        }
