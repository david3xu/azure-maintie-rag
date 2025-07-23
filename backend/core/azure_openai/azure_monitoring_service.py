"""
Azure Knowledge Monitor - Simplified Architecture
Data-driven configuration with optional Application Insights integration
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class AzureKnowledgeMonitor:
    """
    Simplified Azure Knowledge Extraction Monitor

    Azure Architecture Components:
    - Lightweight performance tracking
    - Optional Application Insights integration
    - Configuration-driven telemetry enablement
    - Graceful degradation without dependencies
    """

    def __init__(self):
        """Initialize simplified monitoring with configuration-driven features"""
        # Core performance tracking
        self.performance_metrics = {
            "pipeline_start_time": None,
            "pipeline_end_time": None,
            "tokens_consumed": 0,
            "api_calls_made": 0,
            "documents_processed": 0
        }

        # Optional Application Insights client (configuration-driven)
        self.app_insights_client = self._initialize_optional_app_insights()
        self.telemetry_enabled = bool(self.app_insights_client)

        logger.info(f"Azure Knowledge Monitor initialized - Telemetry: {'Enabled' if self.telemetry_enabled else 'Disabled'}")

    def _initialize_optional_app_insights(self) -> Optional[Any]:
        """
        Optional Application Insights Integration

        Configuration-driven initialization:
        - Loads from azure_settings if available
        - Gracefully degrades if not configured
        - No hard dependencies on Application Insights
        """
        try:
            from config.settings import azure_settings
            connection_string = getattr(azure_settings, 'azure_application_insights_connection_string', None)

            if not connection_string:
                logger.info("Application Insights not configured - operating in lightweight mode")
                return None

            from core.azure_monitoring.app_insights_client import AzureApplicationInsightsClient
            return AzureApplicationInsightsClient(connection_string=connection_string)

        except (ImportError, AttributeError) as e:
            logger.info(f"Application Insights optional dependency not available: {e}")
            return None

    def start_performance_tracking(self) -> None:
        """
        Performance Tracking Start Interface

        Azure Service Integration:
        - Initialize performance timer
        - Optional telemetry event tracking
        - Reset performance counters
        """
        self.performance_metrics["pipeline_start_time"] = time.time()
        self.performance_metrics["tokens_consumed"] = 0
        self.performance_metrics["api_calls_made"] = 0
        self.performance_metrics["documents_processed"] = 0

        if self.telemetry_enabled:
            self.app_insights_client.track_event(
                name="knowledge_extraction_started",
                properties={"timestamp": datetime.now().isoformat()}
            )

        logger.info("Performance tracking started")

    def end_performance_tracking(self) -> Dict[str, Any]:
        """
        Performance Tracking Completion Interface

        Azure Service Integration:
        - Calculate pipeline duration
        - Return performance metrics
        - Optional telemetry completion tracking
        """
        self.performance_metrics["pipeline_end_time"] = time.time()

        if self.performance_metrics["pipeline_start_time"]:
            duration_ms = (self.performance_metrics["pipeline_end_time"] -
                          self.performance_metrics["pipeline_start_time"]) * 1000
        else:
            duration_ms = 0

        performance_summary = {
            "duration_ms": duration_ms,
            "tokens_used": self.performance_metrics["tokens_consumed"],
            "api_calls": self.performance_metrics["api_calls_made"],
            "documents_processed": self.performance_metrics["documents_processed"]
        }

        if self.telemetry_enabled:
            self.app_insights_client.track_event(
                name="knowledge_extraction_completed",
                measurements=performance_summary
            )

        logger.info(f"Performance tracking completed - Duration: {duration_ms:.1f}ms")
        return performance_summary

    def track_azure_openai_usage(self, operation_type: str, tokens_used: int,
                                response_time_ms: float, model_deployment: str) -> None:
        """
        Azure OpenAI Service Usage Tracking

        Azure Service Integration:
        - Token consumption aggregation
        - API call counting
        - Optional dependency tracking
        """
        self.performance_metrics["tokens_consumed"] += tokens_used
        self.performance_metrics["api_calls_made"] += 1

        if self.telemetry_enabled:
            self.app_insights_client.track_dependency(
                name="azure_openai",
                data=f"{operation_type} - {model_deployment}",
                dependency_type="Azure OpenAI",
                duration=response_time_ms,
                success=True,
                properties={
                    "tokens_used": tokens_used,
                    "operation_type": operation_type,
                    "model_deployment": model_deployment
                }
            )

        logger.debug(f"Azure OpenAI usage tracked - Operation: {operation_type}, Tokens: {tokens_used}")

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """
        Monitoring Service Status Summary

        Service Health Architecture:
        - Current monitoring configuration
        - Performance metrics summary
        - Telemetry service status
        """
        return {
            "telemetry_available": self.telemetry_enabled,
            "monitoring_features": [
                "performance_tracking",
                "azure_openai_usage",
                "lightweight_metrics"
            ] + (["application_insights"] if self.telemetry_enabled else []),
            "current_metrics": dict(self.performance_metrics)
        }


# Maintain backward compatibility with existing imports
AzureEnterpriseKnowledgeMonitor = AzureKnowledgeMonitor