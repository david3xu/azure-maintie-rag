"""
Settings compatibility module for test imports.
This module provides backward compatibility for test imports that expect config.settings.
"""

# Import infrastructure layer settings only (main.py removed for layer boundary compliance)
from .azure_settings import Settings, settings, azure_settings, AzureSettings

# Export for infrastructure layer only
__all__ = [
    'Settings',
    'AzureSettings', 
    'settings',
    'azure_settings'
]