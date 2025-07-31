import logging
from typing import Optional, Dict, Any
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace, metrics
from opentelemetry.trace import Tracer
from opentelemetry.metrics import Meter
from config.settings import azure_settings

logger = logging.getLogger(__name__)

class AzureApplicationInsightsClient:
    _configured = False
    _warning_shown = False
    """Production Application Insights client using Azure Monitor OpenTelemetry."""
    def __init__(self, connection_string: Optional[str] = None, sampling_rate: float = 1.0):
        self.connection_string = connection_string or azure_settings.azure_application_insights_connection_string
        self.sampling_rate = sampling_rate
        self.enabled = bool(self.connection_string)
        if not self.enabled:
            # Only show warning once to avoid spam
            if not AzureApplicationInsightsClient._warning_shown:
                logger.info("Application Insights disabled - no connection string configured")
                AzureApplicationInsightsClient._warning_shown = True
            self.tracer = None
            self.meter = None
            return
        # Only configure once globally
        if not AzureApplicationInsightsClient._configured:
            try:
                configure_azure_monitor(
                    connection_string=self.connection_string,
                    sampling_ratio=self.sampling_rate / 100.0,
                    enable_logging=True,
                    logging_formatter=None
                )
                AzureApplicationInsightsClient._configured = True
            except Exception as e:
                logger.warning(f"Failed to configure Azure Monitor OpenTelemetry: {e}")
                self.enabled = False
                self.tracer = None
                self.meter = None
                return
        # Get tracer and meter
        self.tracer = trace.get_tracer(__name__)
        self.meter = metrics.get_meter(__name__)

    def track_event(self, name: str, properties: Optional[Dict[str, Any]] = None, measurements: Optional[Dict[str, float]] = None):
        if not self.enabled:
            return
        with self.tracer.start_as_current_span(name) as span:
            if properties:
                for k, v in properties.items():
                    span.set_attribute(k, v)
            if measurements:
                for k, v in measurements.items():
                    span.set_attribute(f"measurement.{k}", v)
            logger.debug(f"Tracked event: {name} with properties: {properties} and measurements: {measurements}")

    def track_dependency(self, name: str, data: str, dependency_type: str = "HTTP", duration: float = 0, success: bool = True, properties: Optional[Dict[str, Any]] = None):
        if not self.enabled:
            return
        with self.tracer.start_as_current_span(name) as span:
            span.set_attribute("dependency.type", dependency_type)
            span.set_attribute("dependency.data", data)
            span.set_attribute("dependency.duration", duration)
            span.set_attribute("dependency.success", success)
            if properties:
                for k, v in properties.items():
                    span.set_attribute(k, v)
            logger.debug(f"Tracked dependency: {name} with data: {data}")

    def track_metric(self, name: str, value: float, properties: Optional[Dict[str, Any]] = None):
        if not self.enabled:
            return
        counter = self.meter.create_counter(name)
        counter.add(value, attributes=properties or {})
        logger.debug(f"Tracked metric: {name} = {value}")

    def get_service_status(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self.enabled else "disabled",
            "service": "app_insights",
            "connection_string_configured": bool(self.connection_string),
            "sampling_rate": self.sampling_rate
        }