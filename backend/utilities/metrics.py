"""Universal metrics collection for any domain."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics
import json
from pathlib import Path


@dataclass
class MetricValue:
    """Individual metric value with timestamp."""
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Universal metrics collector that works across all domains."""

    def __init__(self, domain: str = "universal"):
        """Initialize metrics collector."""
        self.domain = domain
        self.metrics: Dict[str, List[MetricValue]] = {}
        self.start_time = datetime.now()

    def record_metric(
        self,
        name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []

        metric_value = MetricValue(
            value=value,
            metadata=metadata or {}
        )
        self.metrics[name].append(metric_value)

    def record_extraction_metrics(
        self,
        num_texts: int,
        num_entities: int,
        num_relations: int,
        duration: float,
        avg_confidence: float
    ) -> None:
        """Record extraction-specific metrics."""
        self.record_metric("extraction.num_texts", num_texts)
        self.record_metric("extraction.num_entities", num_entities)
        self.record_metric("extraction.num_relations", num_relations)
        self.record_metric("extraction.duration_seconds", duration)
        self.record_metric("extraction.avg_confidence", avg_confidence)
        self.record_metric("extraction.entities_per_text", num_entities / max(num_texts, 1))
        self.record_metric("extraction.relations_per_text", num_relations / max(num_texts, 1))
        self.record_metric("extraction.texts_per_second", num_texts / max(duration, 0.001))

    def record_query_metrics(
        self,
        query_type: str,
        num_results: int,
        duration: float,
        confidence: float,
        cache_hit: bool = False
    ) -> None:
        """Record query-specific metrics."""
        self.record_metric("query.duration_seconds", duration, {"type": query_type})
        self.record_metric("query.num_results", num_results, {"type": query_type})
        self.record_metric("query.confidence", confidence, {"type": query_type})
        self.record_metric("query.cache_hit", 1.0 if cache_hit else 0.0, {"type": query_type})

    def record_llm_metrics(
        self,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration: float,
        cost: float = 0.0
    ) -> None:
        """Record LLM usage metrics."""
        metadata = {"model": model_name}
        self.record_metric("llm.prompt_tokens", prompt_tokens, metadata)
        self.record_metric("llm.completion_tokens", completion_tokens, metadata)
        self.record_metric("llm.total_tokens", prompt_tokens + completion_tokens, metadata)
        self.record_metric("llm.duration_seconds", duration, metadata)
        self.record_metric("llm.cost", cost, metadata)

    def get_metric_summary(self, name: str) -> Optional[Dict[str, Any]]:
        """Get summary statistics for a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return None

        values = [m.value for m in self.metrics[name]]

        return {
            'name': name,
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'total': sum(values),
            'latest': values[-1],
            'first_recorded': self.metrics[name][0].timestamp.isoformat(),
            'last_recorded': self.metrics[name][-1].timestamp.isoformat()
        }

    def get_all_summaries(self) -> Dict[str, Any]:
        """Get summary statistics for all metrics."""
        summaries = {}

        for metric_name in self.metrics:
            summaries[metric_name] = self.get_metric_summary(metric_name)

        return {
            'domain': self.domain,
            'collection_started': self.start_time.isoformat(),
            'metrics': summaries
        }

    def get_recent_metrics(self, minutes: int = 5) -> Dict[str, List[MetricValue]]:
        """Get metrics from the last N minutes."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        recent_metrics = {}

        for metric_name, values in self.metrics.items():
            recent_values = [v for v in values if v.timestamp >= cutoff]
            if recent_values:
                recent_metrics[metric_name] = recent_values

        return recent_metrics

    def export_metrics(self, file_path: str) -> None:
        """Export all metrics to a file."""
        export_data = {
            'domain': self.domain,
            'exported_at': datetime.now().isoformat(),
            'collection_started': self.start_time.isoformat(),
            'metrics': {}
        }

        for metric_name, values in self.metrics.items():
            export_data['metrics'][metric_name] = [
                {
                    'value': v.value,
                    'timestamp': v.timestamp.isoformat(),
                    'metadata': v.metadata
                }
                for v in values
            ]

        file_path_obj = Path(file_path)
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path_obj, 'w') as f:
            json.dump(export_data, f, indent=2)

    def clear_metrics(self, older_than_hours: Optional[int] = None) -> None:
        """Clear metrics, optionally only those older than specified hours."""
        if older_than_hours is None:
            self.metrics.clear()
        else:
            cutoff = datetime.now() - timedelta(hours=older_than_hours)
            for metric_name in list(self.metrics.keys()):
                self.metrics[metric_name] = [
                    v for v in self.metrics[metric_name] if v.timestamp >= cutoff
                ]
                if not self.metrics[metric_name]:
                    del self.metrics[metric_name]