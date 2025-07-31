"""
Enhanced Observability with Correlation IDs and Structured Context

This module provides comprehensive observability capabilities including
correlation ID tracking, structured logging, performance monitoring,
and distributed tracing support.
"""

import asyncio
import time
import uuid
import contextvars
import functools
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import json

import logging

# Context variable for correlation ID tracking across async calls
correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar('correlation_id', default='')
operation_context_var: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar('operation_context', default={})

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of operations for monitoring"""
    AGENT_REASONING = "agent_reasoning"
    SEARCH_EXECUTION = "search_execution"
    PATTERN_EXTRACTION = "pattern_extraction"
    API_REQUEST = "api_request"
    DATABASE_QUERY = "database_query"
    EXTERNAL_SERVICE = "external_service"
    BACKGROUND_TASK = "background_task"


class LogLevel(Enum):
    """Enhanced log levels for structured logging"""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class OperationMetrics:
    """Metrics for operation tracking"""
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    performance_met: bool = True
    resource_usage: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationContext:
    """Context information for operations"""
    operation_name: str
    operation_type: OperationType
    correlation_id: str
    parent_operation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


class ObservableOperation:
    """
    Context manager for observable operations with full tracing and correlation.
    
    Provides structured logging, performance monitoring, and correlation ID
    tracking across distributed operations.
    """
    
    def __init__(
        self,
        operation_name: str,
        operation_type: OperationType = OperationType.BACKGROUND_TASK,
        correlation_id: Optional[str] = None,
        parent_operation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        performance_threshold: float = 3.0
    ):
        self.operation_id = str(uuid.uuid4())
        self.context = OperationContext(
            operation_name=operation_name,
            operation_type=operation_type,
            correlation_id=correlation_id or self._get_or_create_correlation_id(),
            parent_operation_id=parent_operation_id,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
            tags=tags or []
        )
        self.metrics = OperationMetrics(start_time=time.time())
        self.performance_threshold = performance_threshold
        self.child_operations: List[str] = []
        
    def _get_or_create_correlation_id(self) -> str:
        """Get existing correlation ID or create new one"""
        existing_id = correlation_id_var.get('')
        if existing_id:
            return existing_id
        return str(uuid.uuid4())
    
    async def __aenter__(self) -> 'ObservableOperation':
        """Start observable operation with full context"""
        
        # Set context variables for child operations
        correlation_id_var.set(self.context.correlation_id)
        operation_context_var.set({
            'operation_id': self.operation_id,
            'operation_name': self.context.operation_name,
            'correlation_id': self.context.correlation_id,
            'parent_operation_id': self.context.parent_operation_id
        })
        
        # Log operation start with structured context
        await self._log_operation_start()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Complete operation with comprehensive logging"""
        
        self.metrics.end_time = time.time()
        self.metrics.duration = self.metrics.end_time - self.metrics.start_time
        self.metrics.success = exc_type is None
        self.metrics.performance_met = self.metrics.duration < self.performance_threshold
        
        if exc_type:
            self.metrics.error_message = str(exc_val)
            await self._log_operation_error(exc_type, exc_val, exc_tb)
        else:
            await self._log_operation_success()
        
        # Update performance tracking
        await self._update_performance_metrics()
    
    async def _log_operation_start(self):
        """Log operation start with full context"""
        
        logger.info(
            f"Operation started: {self.context.operation_name}",
            extra={
                'operation_id': self.operation_id,
                'operation_name': self.context.operation_name,
                'operation_type': self.context.operation_type.value,
                'correlation_id': self.context.correlation_id,
                'parent_operation_id': self.context.parent_operation_id,
                'user_id': self.context.user_id,
                'session_id': self.context.session_id,
                'start_time': self.metrics.start_time,
                'metadata': self.context.metadata,
                'tags': self.context.tags,
                'event_type': 'operation_start'
            }
        )
    
    async def _log_operation_success(self):
        """Log successful operation completion"""
        
        logger.info(
            f"Operation completed successfully: {self.context.operation_name}",
            extra={
                'operation_id': self.operation_id,
                'operation_name': self.context.operation_name,
                'operation_type': self.context.operation_type.value,
                'correlation_id': self.context.correlation_id,
                'parent_operation_id': self.context.parent_operation_id,
                'user_id': self.context.user_id,
                'session_id': self.context.session_id,
                'duration': self.metrics.duration,
                'performance_met': self.metrics.performance_met,
                'performance_threshold': self.performance_threshold,
                'child_operations': self.child_operations,
                'metadata': self.context.metadata,
                'tags': self.context.tags,
                'event_type': 'operation_success'
            }
        )
    
    async def _log_operation_error(self, exc_type, exc_val, exc_tb):
        """Log operation failure with detailed error context"""
        
        logger.error(
            f"Operation failed: {self.context.operation_name}",
            extra={
                'operation_id': self.operation_id,
                'operation_name': self.context.operation_name,
                'operation_type': self.context.operation_type.value,
                'correlation_id': self.context.correlation_id,
                'parent_operation_id': self.context.parent_operation_id,
                'user_id': self.context.user_id,
                'session_id': self.context.session_id,
                'duration': self.metrics.duration,
                'error_type': str(exc_type.__name__) if exc_type else None,
                'error_message': str(exc_val) if exc_val else None,
                'performance_met': self.metrics.performance_met,
                'child_operations': self.child_operations,
                'metadata': self.context.metadata,
                'tags': self.context.tags,
                'event_type': 'operation_error'
            },
            exc_info=True
        )
    
    async def _update_performance_metrics(self):
        """Update global performance metrics"""
        
        # This would integrate with monitoring systems like Prometheus, DataDog, etc.
        # For now, log performance metrics
        
        if not self.metrics.performance_met:
            logger.warning(
                f"Performance threshold exceeded: {self.context.operation_name}",
                extra={
                    'operation_id': self.operation_id,
                    'correlation_id': self.context.correlation_id,
                    'duration': self.metrics.duration,
                    'threshold': self.performance_threshold,
                    'operation_type': self.context.operation_type.value,
                    'event_type': 'performance_violation'
                }
            )
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to operation context"""
        self.context.metadata[key] = value
    
    def add_tag(self, tag: str):
        """Add tag to operation"""
        if tag not in self.context.tags:
            self.context.tags.append(tag)
    
    def log_checkpoint(self, checkpoint_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Log a checkpoint within the operation"""
        
        checkpoint_time = time.time()
        elapsed = checkpoint_time - self.metrics.start_time
        
        logger.debug(
            f"Operation checkpoint: {checkpoint_name}",
            extra={
                'operation_id': self.operation_id,
                'operation_name': self.context.operation_name,
                'correlation_id': self.context.correlation_id,
                'checkpoint_name': checkpoint_name,
                'elapsed_time': elapsed,
                'checkpoint_metadata': metadata or {},
                'event_type': 'operation_checkpoint'
            }
        )
    
    def create_child_operation(
        self,
        child_operation_name: str,
        operation_type: OperationType = OperationType.BACKGROUND_TASK,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'ObservableOperation':
        """Create child operation with proper correlation"""
        
        child_op = ObservableOperation(
            operation_name=child_operation_name,
            operation_type=operation_type,
            correlation_id=self.context.correlation_id,
            parent_operation_id=self.operation_id,
            user_id=self.context.user_id,
            session_id=self.context.session_id,
            metadata=metadata
        )
        
        self.child_operations.append(child_op.operation_id)
        return child_op


def observable_operation(
    operation_name: Optional[str] = None,
    operation_type: OperationType = OperationType.BACKGROUND_TASK,
    performance_threshold: float = 3.0,
    include_args: bool = False,
    include_result: bool = False
):
    """
    Decorator for making functions/methods observable with correlation tracking.
    
    Usage:
        @observable_operation("process_query", OperationType.AGENT_REASONING)
        async def process_query(self, query: str) -> str:
            # Function implementation
            return result
    """
    
    def decorator(func: Callable) -> Callable:
        func_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Extract correlation ID from context if available
                correlation_id = correlation_id_var.get('') or str(uuid.uuid4())
                
                # Prepare metadata
                metadata = {}
                if include_args:
                    metadata['args'] = str(args)
                    metadata['kwargs'] = {k: str(v) for k, v in kwargs.items()}
                
                async with ObservableOperation(
                    operation_name=func_name,
                    operation_type=operation_type,
                    correlation_id=correlation_id,
                    metadata=metadata,
                    performance_threshold=performance_threshold
                ) as op:
                    
                    op.log_checkpoint("function_start")
                    result = await func(*args, **kwargs)
                    op.log_checkpoint("function_complete")
                    
                    if include_result:
                        op.add_metadata('result', str(result)[:500])  # Truncate long results
                    
                    return result
            
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # For sync functions, create simpler logging
                correlation_id = correlation_id_var.get('') or str(uuid.uuid4())
                
                start_time = time.time()
                
                logger.info(
                    f"Sync operation started: {func_name}",
                    extra={
                        'operation_name': func_name,
                        'correlation_id': correlation_id,
                        'operation_type': operation_type.value,
                        'event_type': 'sync_operation_start'
                    }
                )
                
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    logger.info(
                        f"Sync operation completed: {func_name}",
                        extra={
                            'operation_name': func_name,
                            'correlation_id': correlation_id,
                            'duration': duration,
                            'performance_met': duration < performance_threshold,
                            'event_type': 'sync_operation_success'
                        }
                    )
                    
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    
                    logger.error(
                        f"Sync operation failed: {func_name}",
                        extra={
                            'operation_name': func_name,
                            'correlation_id': correlation_id,
                            'duration': duration,
                            'error': str(e),
                            'event_type': 'sync_operation_error'
                        },
                        exc_info=True
                    )
                    raise
            
            return sync_wrapper
    
    return decorator


@asynccontextmanager
async def correlation_context(correlation_id: str):
    """Context manager to set correlation ID for a block of operations"""
    
    token = correlation_id_var.set(correlation_id)
    try:
        yield correlation_id
    finally:
        correlation_id_var.reset(token)


class ObservabilityManager:
    """
    Manager for observability features including metrics collection,
    health monitoring, and performance tracking.
    """
    
    def __init__(self):
        self.operation_metrics: Dict[str, List[float]] = {}
        self.error_counts: Dict[str, int] = {}
        self.performance_violations: List[Dict[str, Any]] = []
        
    def record_operation_metrics(
        self,
        operation_name: str,
        duration: float,
        success: bool,
        performance_met: bool
    ):
        """Record metrics for an operation"""
        
        if operation_name not in self.operation_metrics:
            self.operation_metrics[operation_name] = []
        
        self.operation_metrics[operation_name].append(duration)
        
        if not success:
            self.error_counts[operation_name] = self.error_counts.get(operation_name, 0) + 1
        
        if not performance_met:
            self.performance_violations.append({
                'operation_name': operation_name,
                'duration': duration,
                'timestamp': time.time()
            })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring"""
        
        summary = {
            'operations': {},
            'global_metrics': {
                'total_operations': sum(len(metrics) for metrics in self.operation_metrics.values()),
                'total_errors': sum(self.error_counts.values()),
                'performance_violations': len(self.performance_violations)
            }
        }
        
        for op_name, durations in self.operation_metrics.items():
            if durations:
                summary['operations'][op_name] = {
                    'count': len(durations),
                    'avg_duration': sum(durations) / len(durations),
                    'max_duration': max(durations),
                    'min_duration': min(durations),
                    'error_count': self.error_counts.get(op_name, 0),
                    'error_rate': self.error_counts.get(op_name, 0) / len(durations)
                }
        
        return summary
    
    def clear_metrics(self):
        """Clear collected metrics (for testing or reset)"""
        self.operation_metrics.clear()
        self.error_counts.clear()
        self.performance_violations.clear()


# Global observability manager instance
observability_manager = ObservabilityManager()


def get_current_correlation_id() -> str:
    """Get current correlation ID from context"""
    return correlation_id_var.get('')


def get_current_operation_context() -> Dict[str, Any]:
    """Get current operation context"""
    return operation_context_var.get({})