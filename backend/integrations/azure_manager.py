"""
Azure Manager
Thin coordination layer that orchestrates the focused services
Replaces the massive integrations/azure_services.py
"""

import logging
from typing import Dict, Any, Optional

from services.infrastructure_service import InfrastructureService
from services.data_service import DataService
from services.cleanup_service import CleanupService

logger = logging.getLogger(__name__)


class AzureManager:
    """
    Lightweight coordinator for Azure services
    Delegates to focused service classes for actual implementation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, storage_factory: Optional[Any] = None):
        """Initialize Azure manager with service delegation"""
        # Initialize focused services
        self.infrastructure = InfrastructureService(config)
        self.data = DataService(self.infrastructure)
        self.cleanup = CleanupService(self.infrastructure)
        
        logger.info("✅ Azure Manager initialized with focused services")
    
    # === INFRASTRUCTURE DELEGATION ===
    
    async def initialize(self) -> None:
        """Initialize all Azure services"""
        return await self.infrastructure.initialize()
    
    def check_all_services_health(self) -> Dict[str, Any]:
        """Check health of all Azure services"""
        return self.infrastructure.check_all_services_health()
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate Azure service configurations"""
        return self.infrastructure.validate_configuration()
    
    def get_service(self, service_name: str):
        """Get service instance by name"""
        return self.infrastructure.get_service(service_name)
    
    def get_rag_storage_client(self):
        """Get RAG storage client"""
        return self.infrastructure.get_rag_storage_client()
    
    def get_ml_storage_client(self):
        """Get ML storage client"""
        return self.infrastructure.get_ml_storage_client()
    
    def get_app_storage_client(self):
        """Get application storage client"""
        return self.infrastructure.get_app_storage_client()
    
    def get_storage_factory(self):
        """Get storage factory instance"""
        return self.infrastructure.get_storage_factory()
    
    # === DATA OPERATIONS DELEGATION ===
    
    async def migrate_data_to_azure(self, source_data_path: str, domain: str) -> Dict[str, Any]:
        """Migrate data to Azure services"""
        return await self.data.migrate_data_to_azure(source_data_path, domain)
    
    # === CLEANUP OPERATIONS DELEGATION ===
    
    async def cleanup_all_azure_data(self, domain: str = "general") -> Dict[str, Any]:
        """Cleanup all Azure data for a domain"""
        return await self.cleanup.cleanup_all_azure_data(domain)
    
    async def cleanup_expired_data(self, retention_days: int = 30) -> Dict[str, Any]:
        """Cleanup data older than retention period"""
        return await self.cleanup.cleanup_expired_data(retention_days)
    
    async def maintenance_health_check(self) -> Dict[str, Any]:
        """Perform maintenance health checks"""
        return await self.cleanup.maintenance_health_check()
    
    # === CONVENIENCE PROPERTIES ===
    
    @property
    def openai_client(self):
        """Get OpenAI client"""
        return self.infrastructure.openai_client
    
    @property
    def search_service(self):
        """Get search service"""
        return self.infrastructure.search_service
    
    @property
    def cosmos_client(self):
        """Get Cosmos client"""
        return self.infrastructure.cosmos_client
    
    @property
    def ml_client(self):
        """Get ML client"""
        return self.infrastructure.ml_client
    
    @property
    def app_insights(self):
        """Get Application Insights client"""
        return self.infrastructure.app_insights


# Maintain backwards compatibility with old class name
AzureServicesManager = AzureManager