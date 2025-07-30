"""
Infrastructure Service
Handles Azure service management, health checks, and initialization
PRODUCTION-READY: Uses real Azure services with azure_settings configuration
"""

import logging
import time
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Real Azure clients from core
from core.azure_storage import UnifiedStorageClient
from core.azure_search import UnifiedSearchClient
from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient
from core.azure_openai import UnifiedAzureOpenAIClient
from core.azure_openai.embedding import AzureEmbeddingService
from core.azure_monitoring.app_insights_client import AzureApplicationInsightsClient
# Removed duplicate import - using UnifiedAzureOpenAIClient from core

# Azure ML client
from core.azure_ml.ml_client import AzureMLClient

# Configuration
from config.settings import azure_settings

logger = logging.getLogger(__name__)


class InfrastructureService:
    """Production Azure infrastructure service using real Azure clients"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize infrastructure service with real Azure clients"""
        self.config = config or {}
        
        # Real Azure service instances
        self.openai_client = None
        self.vector_service = None
        self.search_service = None
        self.cosmos_client = None
        self.ml_client = None
        self.app_insights = None
        self.storage_client = None
        
        # Service state
        self.initialized = False
        self.service_status = {}
        
        # Initialize real Azure services
        self._initialize_azure_services()
    
    @property
    def search_client(self):
        """Alias for search_service for backward compatibility"""
        return self.search_service
    
    def _initialize_azure_services(self) -> None:
        """Initialize real Azure services with azd compatibility"""
        logger.info("🚀 Initializing Azure services for Universal RAG...")
        
        # Check if this is an azd-managed deployment
        if azure_settings.is_azd_deployment:
            logger.info("🏗️ Detected azd-managed deployment - using managed identity")
        else:
            logger.info("🔧 Using legacy configuration - API keys/connection strings")
        
        try:
            # Real OpenAI client with azd compatibility
            try:
                self.openai_client = UnifiedAzureOpenAIClient()
                endpoint = azure_settings.effective_openai_endpoint
                logger.info(f"✅ Azure OpenAI client initialized - endpoint: {endpoint}")
            except Exception as e:
                logger.error(f"❌ Azure OpenAI initialization failed: {e}")
                self.openai_client = None
            
            # Real Search service with azd compatibility
            try:
                self.search_service = UnifiedSearchClient()
                endpoint = azure_settings.effective_search_endpoint
                logger.info(f"✅ Azure Cognitive Search initialized - endpoint: {endpoint}")
            except Exception as e:
                logger.error(f"❌ Azure Search initialization failed: {e}")
                self.search_service = None
            
            # Real Storage client with azd compatibility
            try:
                self.storage_client = UnifiedStorageClient()
                account = azure_settings.azure_storage_account
                logger.info(f"✅ Azure Blob Storage initialized - account: {account}")
            except Exception as e:
                logger.error(f"❌ Azure Storage initialization failed: {e}")
                self.storage_client = None
            
            # Real Cosmos DB client with azd compatibility
            try:
                self.cosmos_client = AzureCosmosGremlinClient()
                endpoint = azure_settings.azure_cosmos_endpoint
                database = azure_settings.cosmos_database_name
                logger.info(f"✅ Azure Cosmos DB initialized - endpoint: {endpoint}, database: {database}")
            except Exception as e:
                logger.error(f"❌ Azure Cosmos DB initialization failed: {e}")
                self.cosmos_client = None
            
            # Real ML client with azd compatibility
            try:
                self.ml_client = AzureMLClient()
                workspace = azure_settings.azure_ml_workspace_name
                logger.info(f"✅ Azure ML Workspace initialized - workspace: {workspace}")
            except Exception as e:
                logger.error(f"❌ Azure ML initialization failed: {e}")
                self.ml_client = None
            
            # Real Application Insights with azd compatibility
            try:
                self.app_insights = AzureApplicationInsightsClient()
                connection_string = azure_settings.applicationinsights_connection_string
                if connection_string:
                    logger.info("✅ Azure Application Insights initialized - azd managed")
                else:
                    logger.info("✅ Azure Application Insights initialized - legacy config")
            except Exception as e:
                logger.error(f"❌ Application Insights initialization failed: {e}")
                self.app_insights = None
            
            # Real Vector service (integrated with this infrastructure)
            try:
                from services.vector_service import VectorService
                self.vector_service = VectorService(infrastructure_service=self)
                logger.info("✅ Azure OpenAI Vector service initialized")
            except Exception as e:
                logger.error(f"❌ Vector service initialization failed: {e}")
                self.vector_service = None
                
        except Exception as e:
            logger.error(f"Critical error during Azure services initialization: {e}")
    
    async def initialize(self) -> None:
        """Complete async initialization of Azure services"""
        if self.initialized:
            return
        
        logger.info("Starting async initialization of Azure services")
        
        # Perform async initialization for services that support it
        if self.storage_client and hasattr(self.storage_client, 'initialize_async'):
            try:
                await self.storage_client.initialize_async()
                logger.info("✅ Storage client async initialization completed")
            except Exception as e:
                logger.error(f"❌ Storage async initialization failed: {e}")
        
        if self.search_service and hasattr(self.search_service, 'initialize_async'):
            try:
                await self.search_service.initialize_async()
                logger.info("✅ Search service async initialization completed")
            except Exception as e:
                logger.error(f"❌ Search async initialization failed: {e}")
        
        self.initialized = True
        logger.info("✅ Azure infrastructure services fully initialized")
    
    def check_all_services_health(self) -> Dict[str, Any]:
        """Real health check of all Azure services"""
        start_time = time.time()
        health_results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "services": {},
            "summary": {
                "total_services": 0,
                "healthy_services": 0,
                "unhealthy_services": 0,
                "unknown_services": 0
            },
            "azure_settings_status": self._check_azure_settings()
        }
        
        # Real Azure services to check
        services_to_check = {
            "openai": self.openai_client,
            "search": self.search_service,
            "storage": self.storage_client,
            "cosmos": self.cosmos_client,
            "ml": self.ml_client,
            "app_insights": self.app_insights,
            "vector": self.vector_service
        }
        
        for service_name, service_instance in services_to_check.items():
            health_results["services"][service_name] = self._health_check_with_timeout(service_name, service_instance, 10)
            health_results["summary"]["total_services"] += 1
            
            status = health_results["services"][service_name]["status"]
            if status == "healthy":
                health_results["summary"]["healthy_services"] += 1
            elif status == "unhealthy":
                health_results["summary"]["unhealthy_services"] += 1
            else:
                health_results["summary"]["unknown_services"] += 1
        
        # Determine overall status based on real service health
        if health_results["summary"]["unhealthy_services"] > 0:
            health_results["overall_status"] = "degraded"
        if health_results["summary"]["healthy_services"] == 0:
            health_results["overall_status"] = "unhealthy"
        
        health_results["check_duration"] = round(time.time() - start_time, 2)
        return health_results
    
    def _health_check_with_timeout(self, service_name: str, service_instance: Any, timeout_seconds: int) -> Dict[str, Any]:
        """Real health check with timeout for Azure services"""
        try:
            if service_instance is None:
                return {
                    "status": "unhealthy",
                    "error": f"Service {service_name} not initialized",
                    "timestamp": datetime.now().isoformat()
                }
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._perform_real_health_check, service_name, service_instance)
                return future.result(timeout=timeout_seconds)
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": f"Health check failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _perform_real_health_check(self, service_name: str, service_instance: Any) -> Dict[str, Any]:
        """Perform actual health check on real Azure service"""
        try:
            health_check_start = time.time()
            
            # Call real service health check if available
            if hasattr(service_instance, 'health_check'):
                result = service_instance.health_check()
                return {
                    "status": "healthy" if result else "unhealthy",
                    "details": result,
                    "response_time": round(time.time() - health_check_start, 3),
                    "timestamp": datetime.now().isoformat()
                }
            
            # For services without explicit health check, verify they're responsive
            if hasattr(service_instance, 'ensure_initialized'):
                service_instance.ensure_initialized()
            
            return {
                "status": "healthy",
                "message": f"Azure {service_name} service is responsive",
                "response_time": round(time.time() - health_check_start, 3),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _check_azure_settings(self) -> Dict[str, Any]:
        """Verify Azure configuration settings are properly loaded"""
        try:
            settings_status = {
                "openai_configured": bool(getattr(azure_settings, 'openai_api_key', None)),
                "search_configured": bool(getattr(azure_settings, 'azure_search_api_key', None)),
                "storage_configured": bool(getattr(azure_settings, 'azure_storage_connection_string', None)),
                "cosmos_configured": bool(getattr(azure_settings, 'cosmos_primary_key', None)),
                "ml_configured": bool(getattr(azure_settings, 'azure_ml_workspace_name', None))
            }
            
            settings_status["all_configured"] = all(settings_status.values())
            return settings_status
            
        except Exception as e:
            return {"error": f"Failed to check Azure settings: {str(e)}"}
    
    def get_service(self, service_name: str):
        """Get real Azure service instance by name"""
        service_mapping = {
            "openai": self.openai_client,
            "search": self.search_service,
            "storage": self.storage_client,
            "cosmos": self.cosmos_client,
            "ml": self.ml_client,
            "app_insights": self.app_insights,
            "vector": self.vector_service
        }
        return service_mapping.get(service_name)
    
    def get_rag_storage_client(self):
        """Get real RAG storage client"""
        if not self.storage_client:
            raise RuntimeError("Azure Storage client not initialized")
        return self.storage_client
    
    def get_ml_storage_client(self):
        """Get real ML storage client (same as RAG for unified storage)"""
        if not self.storage_client:
            raise RuntimeError("Azure Storage client not initialized")
        return self.storage_client
    
    def get_app_storage_client(self):
        """Get real application storage client"""
        if not self.storage_client:
            raise RuntimeError("Azure Storage client not initialized")
        return self.storage_client
    
    def get_storage_factory(self):
        """Get storage factory (compatibility method)"""
        return self
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate real Azure service configurations"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "services_validated": [],
            "azure_settings_check": self._check_azure_settings()
        }
        
        # Validate required Azure settings
        required_settings = [
            ("openai_api_key", "Azure OpenAI API Key"),
            ("openai_api_base", "Azure OpenAI Endpoint"),
            ("azure_search_service_name", "Azure Search Service Name"),
            ("azure_search_api_key", "Azure Search API Key"),
            ("azure_storage_connection_string", "Azure Storage Connection String"),
            ("cosmos_account_uri", "Cosmos DB Account URI"),
            ("cosmos_primary_key", "Cosmos DB Primary Key")
        ]
        
        for setting_name, display_name in required_settings:
            if not hasattr(azure_settings, setting_name) or not getattr(azure_settings, setting_name):
                validation_results["errors"].append(f"Missing required Azure setting: {display_name}")
                validation_results["valid"] = False
        
        # Validate service connectivity with real Azure services
        for service_name in ["openai", "search", "storage", "cosmos", "ml", "vector"]:
            try:
                service = self.get_service(service_name)
                if service:
                    # Try to perform a simple operation to verify connectivity
                    if hasattr(service, 'ensure_initialized'):
                        service.ensure_initialized()
                    validation_results["services_validated"].append(service_name)
                else:
                    validation_results["warnings"].append(f"Azure {service_name} service not available")
            except Exception as e:
                validation_results["errors"].append(f"Azure {service_name} validation failed: {str(e)}")
                validation_results["valid"] = False
        
        return validation_results