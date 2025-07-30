"""
Azure service dependencies for FastAPI endpoints
Centralized dependency injection to avoid circular imports
"""

import logging
from typing import Optional
from fastapi import HTTPException

from services.infrastructure_service import InfrastructureService
from services.data_service import DataService
from config.settings import AzureSettings
from core.azure_openai.openai_client import UnifiedAzureOpenAIClient
from integrations.azure_services import AzureServicesManager

logger = logging.getLogger(__name__)

# Global references to be set by main.py
_infrastructure_service: Optional[InfrastructureService] = None
_data_service: Optional[DataService] = None
_azure_services: Optional[AzureServicesManager] = None
_openai_integration: Optional[UnifiedAzureOpenAIClient] = None
_azure_settings: Optional[AzureSettings] = None


def set_azure_services(azure_services: AzureServicesManager):
    """Set the global Azure services instance"""
    global _azure_services
    _azure_services = azure_services


def set_openai_integration(openai_integration: UnifiedAzureOpenAIClient):
    """Set the global OpenAI integration instance"""
    global _openai_integration
    _openai_integration = openai_integration


def set_azure_settings(azure_settings: AzureSettings):
    """Set the global Azure settings instance"""
    global _azure_settings
    _azure_settings = azure_settings


async def get_azure_services() -> AzureServicesManager:
    """Get Azure services instance"""
    if not _azure_services:
        raise HTTPException(status_code=503, detail="Azure services not initialized")
    return _azure_services


async def get_openai_integration() -> UnifiedAzureOpenAIClient:
    """Get Azure OpenAI integration instance"""
    if not _openai_integration:
        raise HTTPException(status_code=503, detail="Azure OpenAI integration not initialized")
    return _openai_integration


async def get_azure_settings() -> AzureSettings:
    """Get Azure settings instance"""
    if not _azure_settings:
        raise HTTPException(status_code=503, detail="Azure settings not initialized")
    return _azure_settings