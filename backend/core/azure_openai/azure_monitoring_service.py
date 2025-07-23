"""
Azure Enterprise Application Insights Architecture
Complete telemetry integration for Universal RAG knowledge extraction pipeline
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from core.azure_monitoring.app_insights_client import AzureApplicationInsightsClient
from config.settings import azure_settings

logger = logging.getLogger(__name__)


class AzureEnterpriseKnowledgeMonitor:
    """
    Enterprise Azure Application Insights Integration

    Architecture Components:
    - Application Insights Telemetry Orchestration
    - Knowledge Extraction Performance Monitoring
    - Azure OpenAI Service Usage Tracking
    - Enterprise Quality Metrics Collection
    - Operational Excellence Monitoring Pipeline
    """

    def __init__(self):
        """
        Initialize Enterprise Application Insights Architecture

        Service Integration Pattern:
        - Azure Application Insights client orchestration
        - Enterprise telemetry configuration management
        - Performance metrics tracking initialization
        - Custom metrics mapping for knowledge extraction pipeline
        """
        # Enterprise Application Insights client integration
        self.app_insights_client = self._initialize_application_insights_client()

        # Enterprise telemetry configuration
        self.telemetry_enabled = bool(self.app_insights_client and self.app_insights_client.enabled)

        # Knowledge extraction custom metrics mapping
        self.custom_metrics = {
            "extraction_quality_score": "knowledge_extraction.quality.score",
            "entities_extracted": "knowledge_extraction.entities.count",
            "relations_extracted": "knowledge_extraction.relations.count",
            "processing_duration": "knowledge_extraction.processing.duration_ms",
            "azure_openai_tokens_used": "knowledge_extraction.azure_openai.tokens",
            "confidence_distribution": "knowledge_extraction.confidence.distribution",
            "search_index_operations": "knowledge_extraction.search.operations",
            "cosmos_db_operations": "knowledge_extraction.cosmos.operations"
        }

        # Performance tracking architecture
        self.performance_metrics = {
            "pipeline_start_time": None,
            "pipeline_end_time": None,
            "tokens_consumed": 0,
            "api_calls_made": 0,
            "documents_processed": 0,
            "knowledge_entities_created": 0,
            "knowledge_relations_created": 0
        }

        # Enterprise monitoring status
        self.monitoring_status = {
            "telemetry_active": self.telemetry_enabled,
            "metrics_tracked": len(self.custom_metrics),
            "performance_monitoring": True,
            "azure_integration": "application_insights"
        }

        logger.info(f"Enterprise Knowledge Monitor initialized - Telemetry: {'Enabled' if self.telemetry_enabled else 'Disabled'}")

    def _initialize_application_insights_client(self) -> Optional[AzureApplicationInsightsClient]:
        """
        Azure Application Insights Client Orchestration

        Enterprise Integration Architecture:
        - Connection string configuration from Azure Key Vault or app settings
        - Sampling rate configuration for cost optimization
        - OpenTelemetry integration for distributed tracing
        - Graceful degradation when telemetry is not configured
        """
        try:
            # Enterprise connection string management
            connection_string = azure_settings.azure_application_insights_connection_string
            sampling_rate = azure_settings.effective_telemetry_sampling_rate

            if not connection_string:
                logger.warning("Application Insights connection string not configured - telemetry disabled")
                return None

            # Initialize Azure Application Insights client
            app_insights_client = AzureApplicationInsightsClient(
                connection_string=connection_string,
                sampling_rate=sampling_rate
            )

            logger.info("Azure Application Insights client initialized successfully")
            return app_insights_client

        except Exception as e:
            logger.error(f"Application Insights client initialization failed: {e}")
            return None

    def start_knowledge_extraction_monitoring(self, extraction_context: Dict[str, Any]) -> None:
        """
        Enterprise Knowledge Extraction Pipeline Monitoring Initialization

        Monitoring Architecture:
        - Pipeline performance tracking initiation
        - Context-aware telemetry setup
        - Azure service correlation tracking
        - Custom metrics initialization
        """
        self.performance_metrics["pipeline_start_time"] = time.time()
        self.performance_metrics["documents_processed"] = 0
        self.performance_metrics["tokens_consumed"] = 0
        self.performance_metrics["api_calls_made"] = 0

        if self.telemetry_enabled:
            # Track knowledge extraction pipeline start event
            self.app_insights_client.track_event(
                name="knowledge_extraction_pipeline_started",
                properties={
                    "domain": extraction_context.get("domain", "general"),
                    "source_path": extraction_context.get("source_path", "unknown"),
                    "migration_id": extraction_context.get("migration_id", "unknown"),
                    "azure_environment": azure_settings.azure_environment,
                    "extraction_tier": azure_settings.extraction_quality_tier
                }
            )

        logger.info("Knowledge extraction monitoring started")

    def track_azure_openai_usage(self, operation_type: str, tokens_used: int, response_time_ms: float,
                                 model_deployment: str) -> None:
        """
        Azure OpenAI Service Usage Tracking

        Service Monitoring Architecture:
        - Token consumption tracking for cost management
        - Response time monitoring for performance optimization
        - Model deployment usage analytics
        - Rate limiting compliance monitoring
        """
        self.performance_metrics["tokens_consumed"] += tokens_used
        self.performance_metrics["api_calls_made"] += 1

        if self.telemetry_enabled:
            # Track Azure OpenAI service dependency
            self.app_insights_client.track_dependency(
                name="azure_openai_completion",
                data=f"{operation_type} - {model_deployment}",
                dependency_type="Azure OpenAI",
                duration=response_time_ms,
                success=True,
                properties={
                    "operation_type": operation_type,
                    "tokens_used": tokens_used,
                    "model_deployment": model_deployment,
                    "azure_openai_endpoint": azure_settings.azure_openai_endpoint
                }
            )

            # Track token consumption metric
            self.app_insights_client.track_metric(
                name=self.custom_metrics["azure_openai_tokens_used"],
                value=tokens_used,
                properties={
                    "operation": operation_type,
                    "model": model_deployment
                }
            )

    def track_knowledge_extraction_results(self, extraction_results: Dict[str, Any]) -> None:
        """
        Knowledge Extraction Quality Metrics Tracking

        Quality Monitoring Architecture:
        - Entity extraction count and confidence scoring
        - Relation extraction quality assessment
        - Knowledge graph completeness metrics
        - Enterprise quality tier compliance validation
        """
        entities_count = len(extraction_results.get("entities", {}))
        relations_count = len(extraction_results.get("relations", []))
        quality_score = extraction_results.get("extraction_metadata", {}).get("confidence", 0.0)

        # Update performance counters
        self.performance_metrics["knowledge_entities_created"] += entities_count
        self.performance_metrics["knowledge_relations_created"] += relations_count

        if self.telemetry_enabled:
            # Track extraction quality metrics
            self.app_insights_client.track_metric(
                name=self.custom_metrics["extraction_quality_score"],
                value=quality_score,
                properties={
                    "domain": extraction_results.get("domain", "general"),
                    "quality_tier": azure_settings.extraction_quality_tier
                }
            )

            self.app_insights_client.track_metric(
                name=self.custom_metrics["entities_extracted"],
                value=entities_count
            )

            self.app_insights_client.track_metric(
                name=self.custom_metrics["relations_extracted"],
                value=relations_count
            )

            # Track extraction success event
            self.app_insights_client.track_event(
                name="knowledge_extraction_completed",
                properties={
                    "entities_extracted": entities_count,
                    "relations_extracted": relations_count,
                    "quality_score": quality_score,
                    "extraction_success": True
                },
                measurements={
                    "extraction_quality": quality_score,
                    "knowledge_density": entities_count + relations_count
                }
            )

    def track_azure_search_operations(self, operation_type: str, index_name: str,
                                     documents_processed: int, operation_duration_ms: float,
                                     operation_success: bool) -> None:
        """
        Azure Cognitive Search Operations Monitoring

        Search Service Architecture:
        - Index operations performance tracking
        - Document processing throughput monitoring
        - Search service availability validation
        - Indexing pipeline success rate tracking
        """
        if self.telemetry_enabled:
            # Track Azure Search dependency
            self.app_insights_client.track_dependency(
                name="azure_cognitive_search",
                data=f"{operation_type} - {index_name}",
                dependency_type="Azure Cognitive Search",
                duration=operation_duration_ms,
                success=operation_success,
                properties={
                    "operation_type": operation_type,
                    "index_name": index_name,
                    "documents_processed": documents_processed,
                    "search_service_name": azure_settings.azure_search_service
                }
            )

            # Track search operations metric
            self.app_insights_client.track_metric(
                name=self.custom_metrics["search_index_operations"],
                value=documents_processed,
                properties={
                    "operation": operation_type,
                    "index": index_name,
                    "success": operation_success
                }
            )

    def track_azure_cosmos_operations(self, operation_type: str, graph_name: str,
                                     entities_processed: int, relations_processed: int,
                                     operation_duration_ms: float, operation_success: bool) -> None:
        """
        Azure Cosmos DB Gremlin Operations Monitoring

        Graph Database Architecture:
        - Knowledge graph construction monitoring
        - Entity and relation ingestion tracking
        - Cosmos DB performance metrics collection
        - Graph traversal operation optimization tracking
        """
        if self.telemetry_enabled:
            # Track Azure Cosmos DB dependency
            self.app_insights_client.track_dependency(
                name="azure_cosmos_db_gremlin",
                data=f"{operation_type} - {graph_name}",
                dependency_type="Azure Cosmos DB",
                duration=operation_duration_ms,
                success=operation_success,
                properties={
                    "operation_type": operation_type,
                    "graph_name": graph_name,
                    "entities_processed": entities_processed,
                    "relations_processed": relations_processed,
                    "cosmos_account": azure_settings.azure_cosmos_account
                }
            )

            # Track Cosmos operations metric
            self.app_insights_client.track_metric(
                name=self.custom_metrics["cosmos_db_operations"],
                value=entities_processed + relations_processed,
                properties={
                    "operation": operation_type,
                    "graph": graph_name,
                    "success": operation_success
                }
            )

    def track_extraction_error(self, error: Exception, error_context: Dict[str, Any]) -> None:
        """
        Knowledge Extraction Error Tracking and Alerting

        Error Monitoring Architecture:
        - Exception tracking with full context preservation
        - Error correlation with Azure service dependencies
        - Alerting integration for operational response
        - Error pattern analysis for system improvement
        """
        if self.telemetry_enabled:
            # Track exception with full context
            self.app_insights_client.track_event(
                name="knowledge_extraction_error",
                properties={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "error_context": str(error_context),
                    "domain": error_context.get("domain", "unknown"),
                    "operation": error_context.get("operation", "unknown"),
                    "azure_service": error_context.get("azure_service", "unknown"),
                    "timestamp": datetime.now().isoformat()
                }
            )

        logger.error(f"Knowledge extraction error tracked: {type(error).__name__} - {str(error)}")

    def complete_knowledge_extraction_monitoring(self, extraction_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enterprise Knowledge Extraction Pipeline Completion Monitoring

        Pipeline Monitoring Architecture:
        - End-to-end pipeline performance assessment
        - Resource utilization summary
        - Quality metrics aggregation
        - Cost optimization insights generation
        """
        self.performance_metrics["pipeline_end_time"] = time.time()

        # Calculate pipeline performance metrics
        total_duration = (self.performance_metrics["pipeline_end_time"] -
                         self.performance_metrics["pipeline_start_time"]) * 1000  # Convert to ms

        pipeline_metrics = {
            "total_duration_ms": total_duration,
            "documents_processed": self.performance_metrics["documents_processed"],
            "tokens_consumed": self.performance_metrics["tokens_consumed"],
            "api_calls_made": self.performance_metrics["api_calls_made"],
            "entities_created": self.performance_metrics["knowledge_entities_created"],
            "relations_created": self.performance_metrics["knowledge_relations_created"],
            "avg_processing_time_per_document": (
                total_duration / max(self.performance_metrics["documents_processed"], 1)
            )
        }

        if self.telemetry_enabled:
            # Track pipeline completion event
            self.app_insights_client.track_event(
                name="knowledge_extraction_pipeline_completed",
                properties={
                    "domain": extraction_summary.get("domain", "general"),
                    "extraction_success": extraction_summary.get("success", False),
                    "pipeline_stage": "completed"
                },
                measurements=pipeline_metrics
            )

            # Track processing duration metric
            self.app_insights_client.track_metric(
                name=self.custom_metrics["processing_duration"],
                value=total_duration,
                properties={
                    "pipeline_type": "knowledge_extraction",
                    "completion_status": "success" if extraction_summary.get("success") else "failed"
                }
            )

        logger.info(f"Knowledge extraction monitoring completed - Duration: {total_duration:.2f}ms")
        return pipeline_metrics

    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        Enterprise Monitoring Status for Service Health Validation

        Service Health Architecture:
        - Telemetry service connectivity status
        - Performance monitoring operational state
        - Custom metrics configuration validation
        - Azure Application Insights integration health
        """
        return {
            "monitoring_service": "azure_application_insights",
            "telemetry_enabled": self.telemetry_enabled,
            "connection_status": "healthy" if self.telemetry_enabled else "disabled",
            "custom_metrics_count": len(self.custom_metrics),
            "performance_tracking_active": (
                self.performance_metrics["pipeline_start_time"] is not None
            ),
            "azure_services_monitored": [
                "azure_openai",
                "azure_cognitive_search",
                "azure_cosmos_db",
                "knowledge_extraction_pipeline"
            ],
            "monitoring_capabilities": [
                "quality_score_tracking",
                "token_usage_monitoring",
                "performance_metrics",
                "error_tracking",
                "dependency_monitoring",
                "custom_business_metrics"
            ]
        }

    def get_service_health_status(self) -> Dict[str, Any]:
        """
        Service Health Status for Azure Services Manager Integration

        Health Check Architecture:
        - Application Insights connectivity validation
        - Telemetry pipeline operational status
        - Performance metrics collection health
        - Integration with Azure service monitoring
        """
        if self.app_insights_client:
            return self.app_insights_client.get_service_status()
        else:
            return {
                "status": "disabled",
                "service": "application_insights",
                "connection_configured": False,
                "telemetry_active": False
            }