"""
Azure Application Insights Integration for Knowledge Extraction Monitoring
Real-time telemetry and performance tracking
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import time
from ...config.settings import azure_settings

logger = logging.getLogger(__name__)

class AzureKnowledgeMonitor:
    """Enterprise monitoring for knowledge extraction pipeline"""

    def __init__(self):
        # Initialize telemetry client (simplified for now)
        self.telemetry_client = self._initialize_telemetry_client()

        # Custom metrics mapping
        self.custom_metrics = {
            "extraction_quality_score": "knowledge_extraction.quality.score",
            "entities_extracted": "knowledge_extraction.entities.count",
            "relations_extracted": "knowledge_extraction.relations.count",
            "processing_duration": "knowledge_extraction.processing.duration_ms",
            "azure_openai_tokens_used": "knowledge_extraction.azure_openai.tokens",
            "confidence_distribution": "knowledge_extraction.confidence.distribution"
        }

        # Performance tracking
        self.performance_metrics = {
            "start_time": None,
            "end_time": None,
            "tokens_used": 0,
            "api_calls": 0
        }

    def _initialize_telemetry_client(self):
        """Initialize Application Insights telemetry client"""
        try:
            # For now, return a mock client
            # In production, this would use the actual Azure Application Insights SDK
            return MockTelemetryClient()
        except Exception as e:
            logger.warning(f"Application Insights initialization failed: {e}")
            return None

    async def track_extraction_quality(self, quality_results: Dict[str, Any]) -> None:
        """Track quality metrics in Azure Application Insights"""

        if not self.telemetry_client:
            logger.warning("Telemetry client not available, skipping quality tracking")
            return

        try:
            # Track enterprise quality score
            self.telemetry_client.track_metric(
                self.custom_metrics["extraction_quality_score"],
                quality_results.get("enterprise_quality_score", 0.0),
                properties={
                    "domain": quality_results.get("domain", "unknown"),
                    "quality_tier": quality_results.get("quality_tier", "unknown")
                }
            )

            # Track extraction counts
            self.telemetry_client.track_metric(
                self.custom_metrics["entities_extracted"],
                quality_results.get("entity_count", 0)
            )

            self.telemetry_client.track_metric(
                self.custom_metrics["relations_extracted"],
                quality_results.get("relation_count", 0)
            )

            # Track quality recommendations as custom events
            if "recommendations" in quality_results:
                self.telemetry_client.track_event(
                    "knowledge_extraction_recommendations",
                    properties={
                        "recommendations": quality_results["recommendations"],
                        "quality_score": quality_results.get("enterprise_quality_score", 0.0)
                    }
                )

            # Flush telemetry
            self.telemetry_client.flush()

        except Exception as e:
            logger.error(f"Failed to track extraction quality: {e}")

    async def track_azure_openai_usage(
        self,
        tokens_used: int,
        api_calls: int,
        processing_time_ms: float
    ) -> None:
        """Track Azure OpenAI usage and costs"""

        if not self.telemetry_client:
            return

        try:
            self.telemetry_client.track_metric(
                self.custom_metrics["azure_openai_tokens_used"],
                tokens_used,
                properties={
                    "api_calls": str(api_calls),
                    "avg_tokens_per_call": str(tokens_used / api_calls if api_calls > 0 else 0)
                }
            )

            self.telemetry_client.track_metric(
                self.custom_metrics["processing_duration"],
                processing_time_ms
            )

        except Exception as e:
            logger.error(f"Failed to track Azure OpenAI usage: {e}")

    def start_performance_tracking(self) -> None:
        """Start performance tracking for extraction process"""
        self.performance_metrics["start_time"] = time.time()
        self.performance_metrics["tokens_used"] = 0
        self.performance_metrics["api_calls"] = 0

    def end_performance_tracking(self) -> Dict[str, Any]:
        """End performance tracking and return metrics"""
        self.performance_metrics["end_time"] = time.time()

        duration_ms = (self.performance_metrics["end_time"] - self.performance_metrics["start_time"]) * 1000

        return {
            "duration_ms": duration_ms,
            "tokens_used": self.performance_metrics["tokens_used"],
            "api_calls": self.performance_metrics["api_calls"],
            "avg_tokens_per_call": self.performance_metrics["tokens_used"] / max(self.performance_metrics["api_calls"], 1)
        }

    def track_token_usage(self, tokens: int) -> None:
        """Track token usage during extraction"""
        self.performance_metrics["tokens_used"] += tokens

    def track_api_call(self) -> None:
        """Track API call during extraction"""
        self.performance_metrics["api_calls"] += 1

    async def track_extraction_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Track extraction errors for monitoring and alerting"""

        if not self.telemetry_client:
            return

        try:
            self.telemetry_client.track_exception(
                exception=error,
                properties={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "context": str(context),
                    "timestamp": datetime.now().isoformat()
                }
            )

            self.telemetry_client.flush()

        except Exception as e:
            logger.error(f"Failed to track extraction error: {e}")

    async def track_business_metrics(self, metrics: Dict[str, Any]) -> None:
        """Track business-specific metrics for knowledge extraction"""

        if not self.telemetry_client:
            return

        try:
            for metric_name, metric_value in metrics.items():
                self.telemetry_client.track_metric(
                    f"knowledge_extraction.business.{metric_name}",
                    metric_value,
                    properties={
                        "metric_category": "business",
                        "timestamp": datetime.now().isoformat()
                    }
                )

            self.telemetry_client.flush()

        except Exception as e:
            logger.error(f"Failed to track business metrics: {e}")

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get a summary of monitoring capabilities and status"""
        return {
            "telemetry_available": self.telemetry_client is not None,
            "custom_metrics_count": len(self.custom_metrics),
            "performance_tracking_active": self.performance_metrics["start_time"] is not None,
            "monitoring_features": [
                "quality_score_tracking",
                "token_usage_monitoring",
                "performance_metrics",
                "error_tracking",
                "business_metrics"
            ]
        }


class MockTelemetryClient:
    """Mock telemetry client for development and testing"""

    def __init__(self):
        self.metrics = []
        self.events = []
        self.exceptions = []

    def track_metric(self, name: str, value: float, properties: Optional[Dict[str, Any]] = None):
        """Track a custom metric"""
        self.metrics.append({
            "name": name,
            "value": value,
            "properties": properties or {},
            "timestamp": datetime.now().isoformat()
        })
        logger.info(f"Tracked metric: {name} = {value}")

    def track_event(self, name: str, properties: Optional[Dict[str, Any]] = None):
        """Track a custom event"""
        self.events.append({
            "name": name,
            "properties": properties or {},
            "timestamp": datetime.now().isoformat()
        })
        logger.info(f"Tracked event: {name}")

    def track_exception(self, exception: Exception, properties: Optional[Dict[str, Any]] = None):
        """Track an exception"""
        self.exceptions.append({
            "exception": str(exception),
            "properties": properties or {},
            "timestamp": datetime.now().isoformat()
        })
        logger.error(f"Tracked exception: {exception}")

    def flush(self):
        """Flush telemetry data"""
        logger.info(f"Flushed telemetry: {len(self.metrics)} metrics, {len(self.events)} events, {len(self.exceptions)} exceptions")

    def get_telemetry_summary(self) -> Dict[str, Any]:
        """Get summary of tracked telemetry"""
        return {
            "metrics_count": len(self.metrics),
            "events_count": len(self.events),
            "exceptions_count": len(self.exceptions),
            "latest_metrics": self.metrics[-5:] if self.metrics else [],
            "latest_events": self.events[-5:] if self.events else [],
            "latest_exceptions": self.exceptions[-5:] if self.exceptions else []
        }