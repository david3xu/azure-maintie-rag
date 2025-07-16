"""Monitoring integration for Universal RAG system."""

from typing import Dict, List, Any, Optional
from datetime import datetime


class MonitoringClient:
    """Universal monitoring client that works with any domain."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize monitoring client."""
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.events = []  # Simple in-memory store for now

    def log_event(self, event_type: str, data: Dict[str, Any], level: str = "info") -> None:
        """Log an event."""
        if not self.enabled:
            return

        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'level': level,
            'data': data
        }
        self.events.append(event)

    def log_extraction_metrics(self, domain: str, metrics: Dict[str, Any]) -> None:
        """Log extraction metrics."""
        self.log_event('extraction_metrics', {'domain': domain, **metrics})

    def log_query_metrics(self, domain: str, metrics: Dict[str, Any]) -> None:
        """Log query metrics."""
        self.log_event('query_metrics', {'domain': domain, **metrics})

    def log_error(self, error: str, context: Dict[str, Any] = None) -> None:
        """Log an error."""
        self.log_event('error', {'error': error, 'context': context or {}}, level='error')

    def get_recent_events(self, minutes: int = 5) -> List[Dict[str, Any]]:
        """Get recent events."""
        return self.events[-100:]  # Return last 100 events for now