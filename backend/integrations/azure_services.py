"""
Azure Services Manager - Thin Delegation Layer
This is a lightweight coordination layer that delegates to focused service classes.
All business logic has been moved to the services layer.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from services.infrastructure_service import InfrastructureService
from services.data_service import DataService
from services.workflow_service import WorkflowService
from config.settings import azure_settings

logger = logging.getLogger(__name__)


class AzureServicesManager:
    """
    Thin delegation layer for Azure services coordination.
    Delegates all business logic to appropriate service classes.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, storage_factory: Optional[Any] = None):
        """Initialize with service delegation"""
        self.config = config or {}
        
        # Initialize focused services
        self.infrastructure = InfrastructureService(config)
        self.data_service = DataService(self.infrastructure)
        self.workflow_service = WorkflowService(self.infrastructure)
        
        # Compatibility attributes
        self.initialized = False
        self.storage_factory = storage_factory or self.infrastructure
        
        logger.info("✅ AzureServicesManager initialized with delegation pattern")
    
    async def initialize(self) -> None:
        """Initialize all services"""
        await self.infrastructure.initialize()
        self.initialized = True
        logger.info("✅ All services initialized")
    
    # === DELEGATION METHODS ===
    
    def get_service(self, service_name: str):
        """Delegate to infrastructure service"""
        return self.infrastructure.get_service(service_name)
    
    def get_rag_storage_client(self):
        """Delegate to infrastructure service"""
        return self.infrastructure.get_rag_storage_client()
    
    def get_ml_storage_client(self):
        """Delegate to infrastructure service"""
        return self.infrastructure.get_ml_storage_client()
    
    def get_app_storage_client(self):
        """Delegate to infrastructure service"""
        return self.infrastructure.get_app_storage_client()
    
    def get_storage_factory(self):
        """Return storage factory for compatibility"""
        return self.storage_factory
    
    def check_all_services_health(self) -> Dict[str, Any]:
        """Delegate to infrastructure service"""
        return self.infrastructure.check_all_services_health()
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Delegate to infrastructure service"""
        return self.infrastructure.validate_configuration()
    
    # === DATA OPERATIONS DELEGATION ===
    
    async def migrate_data_to_azure(self, source_data_path: str, domain: str) -> Dict[str, Any]:
        """Delegate to data service"""
        return await self.data_service.migrate_data_to_azure(source_data_path, domain)
    
    async def validate_domain_data_state(self, domain: str) -> Dict[str, Any]:
        """Delegate to data service"""
        return await self.data_service.validate_domain_data_state(domain)
    
    # === WORKFLOW OPERATIONS DELEGATION ===
    
    async def initialize_rag_system(self, domain_name: str = "general", 
                                   text_files: Optional[List] = None, 
                                   force_rebuild: bool = False) -> Dict[str, Any]:
        """Delegate to workflow service"""
        return await self.workflow_service.initialize_rag_orchestration(
            domain_name, text_files, force_rebuild
        )
    
    # === INTERNAL DELEGATION METHODS (for compatibility) ===
    
    async def _migrate_to_storage(self, source_data_path: str, domain: str, 
                                  migration_context: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate to data service"""
        return await self.data_service._migrate_to_storage(source_data_path, domain, migration_context)
    
    async def _migrate_to_search(self, source_data_path: str, domain: str, 
                                migration_context: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate to data service"""
        return await self.data_service._migrate_to_search(source_data_path, domain, migration_context)
    
    async def _migrate_to_cosmos(self, source_data_path: str, domain: str, 
                                migration_context: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate to data service"""
        return await self.data_service._migrate_to_cosmos(source_data_path, domain, migration_context)
    
    def _calculate_processing_requirement(self, blob_state: Dict, search_state: Dict, 
                                        cosmos_state: Dict, raw_data_state: Dict) -> str:
        """Delegate to data service"""
        return self.data_service._calculate_processing_requirement(
            blob_state, search_state, cosmos_state, raw_data_state
        )
    
    def _validate_raw_data_directory(self) -> Dict[str, Any]:
        """Delegate to data service"""
        return self.data_service._validate_raw_data_directory()
    
    # === VALIDATION DELEGATION METHODS ===
    
    async def _validate_azure_blob_storage(self, domain: str) -> Dict[str, Any]:
        """Delegate to data service"""
        return await self.data_service._validate_azure_blob_storage(domain)
    
    async def _validate_azure_cognitive_search(self, domain: str) -> Dict[str, Any]:
        """Delegate to data service"""
        return await self.data_service._validate_azure_cognitive_search(domain)
    
    async def _validate_azure_cosmos_db(self, domain: str) -> Dict[str, Any]:
        """Delegate to data service"""
        return await self.data_service._validate_azure_cosmos_db(domain)
    
    # === BACKWARDS COMPATIBILITY ===
    
    @property
    def openai_client(self):
        """Get OpenAI client for backwards compatibility"""
        return self.infrastructure.openai_client
    
    @property
    def search_service(self):
        """Get search service for backwards compatibility"""
        return self.infrastructure.search_service
    
    @property
    def cosmos_client(self):
        """Get Cosmos client for backwards compatibility"""
        return self.infrastructure.cosmos_client
    
    @property
    def ml_client(self):
        """Get ML client for backwards compatibility"""
        return self.infrastructure.ml_client
    
    @property
    def app_insights(self):
        """Get Application Insights for backwards compatibility"""
        return self.infrastructure.app_insights