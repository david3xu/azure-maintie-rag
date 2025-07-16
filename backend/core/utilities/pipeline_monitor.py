"""
Pipeline monitoring and instrumentation module
Provides granular tracking of sub-steps in the RAG pipeline for production debugging and performance analysis
"""

import time
import logging
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import uuid
from contextlib import contextmanager

from config.settings import settings


logger = logging.getLogger(__name__)


@dataclass
class SubStepMetrics:
    """Metrics for a single sub-step in the pipeline"""
    step_name: str
    component: str
    start_time: float
    end_time: float
    duration_ms: float
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None
    cache_hit: bool = False
    api_calls: int = 0
    memory_usage_mb: Optional[float] = None
    custom_metrics: Optional[Dict[str, Any]] = None


@dataclass
class PipelineMetrics:
    """Complete metrics for a pipeline execution"""
    query_id: str
    query: str
    pipeline_type: str
    total_duration_ms: float
    total_steps: int
    successful_steps: int
    failed_steps: int
    total_api_calls: int
    cache_hits: int
    memory_peak_mb: Optional[float] = None
    sub_steps: List[SubStepMetrics] = None
    timestamp: str = None
    confidence_score: Optional[float] = None
    sources_count: Optional[int] = None
    safety_warnings_count: Optional[int] = None


class PipelineMonitor:
    """Monitors and tracks detailed metrics for RAG pipeline execution"""

    def __init__(self, enable_detailed_logging: bool = True, save_metrics: bool = True):
        """Initialize pipeline monitor"""
        self.enable_detailed_logging = enable_detailed_logging
        self.save_metrics = save_metrics
        self.current_query_id: Optional[str] = None
        self.current_metrics: Optional[PipelineMetrics] = None
        self.sub_steps: List[SubStepMetrics] = []
        self.start_time: Optional[float] = None

        # Create metrics directory
        if self.save_metrics:
            self.metrics_dir = Path("data/metrics")
            self.metrics_dir.mkdir(parents=True, exist_ok=True)

        logger.info("PipelineMonitor initialized")

    def start_query(self, query: str, pipeline_type: str = "structured") -> str:
        """Start monitoring a new query execution"""
        self.query_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.sub_steps = []

        self.current_metrics = PipelineMetrics(
            query_id=self.query_id,
            query=query,
            pipeline_type=pipeline_type,
            total_duration_ms=0.0,
            total_steps=0,
            successful_steps=0,
            failed_steps=0,
            total_api_calls=0,
            cache_hits=0,
            sub_steps=[],
            timestamp=datetime.now().isoformat()
        )

        if self.enable_detailed_logging:
            logger.info(f"🔍 Starting query monitoring: {self.query_id[:8]}... | Query: {query[:50]}...")

        return self.query_id

    def end_query(self, confidence_score: Optional[float] = None,
                  sources_count: Optional[int] = None,
                  safety_warnings_count: Optional[int] = None) -> PipelineMetrics:
        """End query monitoring and return final metrics"""
        if not self.current_metrics:
            raise ValueError("No active query to end")

        end_time = time.time()
        total_duration = (end_time - self.start_time) * 1000  # Convert to ms

        # Calculate summary metrics
        successful_steps = sum(1 for step in self.sub_steps if step.success)
        failed_steps = len(self.sub_steps) - successful_steps
        total_api_calls = sum(step.api_calls for step in self.sub_steps)
        cache_hits = sum(1 for step in self.sub_steps if step.cache_hit)

        # Update metrics
        self.current_metrics.total_duration_ms = total_duration
        self.current_metrics.total_steps = len(self.sub_steps)
        self.current_metrics.successful_steps = successful_steps
        self.current_metrics.failed_steps = failed_steps
        self.current_metrics.total_api_calls = total_api_calls
        self.current_metrics.cache_hits = cache_hits
        self.current_metrics.sub_steps = self.sub_steps.copy()
        self.current_metrics.confidence_score = confidence_score
        self.current_metrics.sources_count = sources_count
        self.current_metrics.safety_warnings_count = safety_warnings_count

        # Log summary
        if self.enable_detailed_logging:
            logger.info(f"✅ Query completed: {self.query_id[:8]}... | "
                       f"Duration: {total_duration:.1f}ms | "
                       f"Steps: {successful_steps}/{len(self.sub_steps)} | "
                       f"API calls: {total_api_calls} | "
                       f"Cache hits: {cache_hits}")

        # Save metrics if enabled
        if self.save_metrics:
            self._save_metrics()

        return self.current_metrics

    @contextmanager
    def track_sub_step(self, step_name: str, component: str,
                      input_data: Optional[Any] = None,
                      track_input_size: bool = True,
                      track_output_size: bool = True):
        """Context manager for tracking individual sub-steps"""
        if not self.current_metrics:
            raise ValueError("No active query - call start_query() first")

        step_start = time.time()
        step_metrics = SubStepMetrics(
            step_name=step_name,
            component=component,
            start_time=step_start,
            end_time=0.0,
            duration_ms=0.0,
            input_size=len(str(input_data)) if input_data and track_input_size else None,
            custom_metrics={}
        )

        if self.enable_detailed_logging:
            logger.info(f"  🔄 {step_name} | Component: {component}")

        try:
            yield step_metrics
            step_metrics.success = True

        except Exception as e:
            step_metrics.success = False
            step_metrics.error_message = str(e)
            if self.enable_detailed_logging:
                logger.error(f"  ❌ {step_name} failed: {e}")
            raise

        finally:
            step_metrics.end_time = time.time()
            step_metrics.duration_ms = (step_metrics.end_time - step_metrics.start_time) * 1000

            # Add to sub-steps
            self.sub_steps.append(step_metrics)

            if self.enable_detailed_logging and step_metrics.success:
                logger.info(f"  ✅ {step_name} completed in {step_metrics.duration_ms:.1f}ms")

    def add_custom_metric(self, step_name: str, metric_name: str, value: Any):
        """Add custom metric to the most recent sub-step"""
        if not self.sub_steps:
            return

        # Find the most recent step with this name
        for step in reversed(self.sub_steps):
            if step.step_name == step_name:
                if step.custom_metrics is None:
                    step.custom_metrics = {}
                step.custom_metrics[metric_name] = value
                break

    def track_api_call(self, step_name: str, api_name: str, duration_ms: float, success: bool = True):
        """Track individual API calls within sub-steps"""
        if not self.sub_steps:
            return

        # Find the most recent step with this name
        for step in reversed(self.sub_steps):
            if step.step_name == step_name:
                step.api_calls += 1
                if step.custom_metrics is None:
                    step.custom_metrics = {}

                api_key = f"api_call_{step.api_calls}"
                step.custom_metrics[api_key] = {
                    "api_name": api_name,
                    "duration_ms": duration_ms,
                    "success": success
                }
                break

    def track_cache_hit(self, step_name: str):
        """Mark a sub-step as having a cache hit"""
        if not self.sub_steps:
            return

        # Find the most recent step with this name
        for step in reversed(self.sub_steps):
            if step.step_name == step_name:
                step.cache_hit = True
                break

    def _save_metrics(self):
        """Save metrics to file"""
        if not self.current_metrics:
            return

        try:
            # Convert to dict for JSON serialization
            metrics_dict = asdict(self.current_metrics)

            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pipeline_metrics_{timestamp}_{self.query_id[:8]}.json"
            filepath = self.metrics_dir / filename

            with open(filepath, 'w') as f:
                json.dump(metrics_dict, f, indent=2, default=str)

            logger.debug(f"Metrics saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for the current query"""
        if not self.current_metrics:
            return {}

        # Calculate step-wise performance
        step_performance = {}
        for step in self.sub_steps:
            if step.step_name not in step_performance:
                step_performance[step.step_name] = {
                    "total_calls": 0,
                    "total_duration_ms": 0.0,
                    "avg_duration_ms": 0.0,
                    "success_rate": 0.0,
                    "api_calls": 0,
                    "cache_hits": 0
                }

            perf = step_performance[step.step_name]
            perf["total_calls"] += 1
            perf["total_duration_ms"] += step.duration_ms
            perf["api_calls"] += step.api_calls
            if step.cache_hit:
                perf["cache_hits"] += 1

        # Calculate averages
        for step_name, perf in step_performance.items():
            perf["avg_duration_ms"] = perf["total_duration_ms"] / perf["total_calls"]
            perf["success_rate"] = 1.0  # Will be updated if we track failures per step

        return {
            "query_id": self.current_metrics.query_id,
            "total_duration_ms": self.current_metrics.total_duration_ms,
            "step_performance": step_performance,
            "summary": {
                "total_steps": self.current_metrics.total_steps,
                "successful_steps": self.current_metrics.successful_steps,
                "failed_steps": self.current_metrics.failed_steps,
                "total_api_calls": self.current_metrics.total_api_calls,
                "cache_hits": self.current_metrics.cache_hits,
                "confidence_score": self.current_metrics.confidence_score
            }
        }


# Global monitor instance
_global_monitor: Optional[PipelineMonitor] = None


def get_monitor() -> PipelineMonitor:
    """Get global monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PipelineMonitor()
    return _global_monitor


def reset_monitor():
    """Reset global monitor instance"""
    global _global_monitor
    _global_monitor = None