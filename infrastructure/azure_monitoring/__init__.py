"""
Azure Application Insights Integration
All Azure monitoring functionality using Application Insights client
"""

# Main production client
from .app_insights_client import AzureApplicationInsightsClient

# Aliases for ease of use
AppInsightsClient = AzureApplicationInsightsClient
MonitoringClient = AzureApplicationInsightsClient
AzureMonitoringClient = AzureApplicationInsightsClient

__all__ = [
    "AzureApplicationInsightsClient",
    "AppInsightsClient", 
    "MonitoringClient",
    "AzureMonitoringClient",
]