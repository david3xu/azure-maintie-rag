"""
Azure service dependencies for FastAPI endpoints
Centralized dependency injection to avoid circular imports
"""

import logging
from typing import Optional
from fastapi import HTTPException

from services.infrastructure_service import InfrastructureService
from services.data_service import DataService
from services.query_service import QueryService
from services.workflow_service import WorkflowService
from config.settings import AzureSettings

logger = logging.getLogger(__name__)

# Global references to be set by main.py
_infrastructure_service: Optional[InfrastructureService] = None
_data_service: Optional[DataService] = None
_workflow_service: Optional[WorkflowService] = None
_query_service: Optional[QueryService] = None
_azure_settings: Optional[AzureSettings] = None


def set_infrastructure_service(infrastructure_service: InfrastructureService):
    """Set the global infrastructure service instance"""
    global _infrastructure_service
    _infrastructure_service = infrastructure_service


def set_data_service(data_service: DataService):
    """Set the global data service instance"""
    global _data_service
    _data_service = data_service


def set_workflow_service(workflow_service: WorkflowService):
    """Set the global workflow service instance"""
    global _workflow_service
    _workflow_service = workflow_service


def set_query_service(query_service: QueryService):
    """Set the global query service instance"""
    global _query_service
    _query_service = query_service


def set_azure_settings(azure_settings: AzureSettings):
    """Set the global Azure settings instance"""
    global _azure_settings
    _azure_settings = azure_settings


async def get_infrastructure_service() -> InfrastructureService:
    """Get infrastructure service instance"""
    if not _infrastructure_service:
        raise HTTPException(status_code=503, detail="Infrastructure service not initialized")
    return _infrastructure_service


async def get_data_service() -> DataService:
    """Get data service instance"""
    if not _data_service:
        raise HTTPException(status_code=503, detail="Data service not initialized")
    return _data_service


async def get_workflow_service() -> WorkflowService:
    """Get workflow service instance"""
    if not _workflow_service:
        raise HTTPException(status_code=503, detail="Workflow service not initialized")
    return _workflow_service


async def get_query_service() -> QueryService:
    """Get query service instance with proper dependency injection"""
    if not _query_service:
        # Fallback: create new instance if not initialized
        return QueryService()
    return _query_service


async def get_azure_settings() -> AzureSettings:
    """Get Azure settings instance"""
    if not _azure_settings:
        raise HTTPException(status_code=503, detail="Azure settings not initialized")
    return _azure_settings