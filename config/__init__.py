"""
Azure Universal RAG Configuration System - Infrastructure Layer

This package provides ONLY infrastructure-level configuration settings.
All business logic models moved to services layer.
All Agent 1 configurations managed in agents layer.

Architecture Compliance:
- Infrastructure Layer: Azure settings, timeouts, environment variables
- Agent 1 generates all domain configurations from data analysis
"""

# Infrastructure layer settings only
from .azure_settings import Settings
from .settings import azure_settings

__version__ = "1.0.0"
__all__ = [
    "Settings",
    "azure_settings"
]
