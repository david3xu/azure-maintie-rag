"""
Enhanced Infrastructure Service with Proper Async Initialization
Replaces synchronous blocking operations with parallel async patterns following CODING_STANDARDS.md
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

# Real Azure clients from core
from infra.azure_storage import UnifiedStorageClient
from infra.azure_search import UnifiedSearchClient
from infra.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient
from infra.azure_openai import UnifiedAzureOpenAIClient
from infra.azure_openai.embedding import AzureEmbeddingService
from infra.azure_monitoring.app_insights_client import AzureApplicationInsightsClient
from infra.azure_ml.ml_client import AzureMLClient

# Configuration
from config.settings import azure_settings

logger = logging.getLogger(__name__)


@dataclass
class ServiceInitializationResult:
    """Result of async service initialization"""
    service_name: str
    success: bool
    initialization_time: float
    error_message: Optional[str] = None
    client_instance: Optional[Any] = None


class AsyncInfrastructureService:
    """
    Production Azure infrastructure service with proper async initialization.
    Eliminates blocking operations and uses parallel initialization patterns.
    """
    
    def __init__(self, azure_settings_instance: Optional[Any] = None):
        """
        Initialize service WITHOUT blocking operations.
        Services are initialized asynchronously in parallel via initialize_async().
        """
        self.azure_settings = azure_settings_instance or azure_settings
        
        # Service instances - initialized asynchronously
        self.openai_client: Optional[UnifiedAzureOpenAIClient] = None
        self.vector_service: Optional[Any] = None
        self.search_service: Optional[UnifiedSearchClient] = None
        self.cosmos_client: Optional[AzureCosmosGremlinClient] = None
        self.ml_client: Optional[AzureMLClient] = None
        self.app_insights: Optional[AzureApplicationInsightsClient] = None
        self.storage_client: Optional[UnifiedStorageClient] = None
        
        # Service state tracking
        self.initialized = False
        self.initialization_results: List[ServiceInitializationResult] = []
        self.initialization_start_time: Optional[datetime] = None
        self.initialization_end_time: Optional[datetime] = None
    
    @property
    def search_client(self):
        """Alias for search_service for backward compatibility"""
        return self.search_service
    
    async def initialize_async(self) -> Dict[str, Any]:
        """
        Perform complete async initialization of all Azure services in parallel.
        This is the main initialization method that replaces synchronous patterns.
        """
        if self.initialized:
            logger.info("Infrastructure services already initialized")
            return self._get_initialization_summary()
        
        self.initialization_start_time = datetime.utcnow()
        logger.info("🚀 Starting parallel async initialization of Azure services...")
        
        try:
            # Create initialization tasks for parallel execution
            initialization_tasks = [
                self._initialize_openai_service(),
                self._initialize_search_service(),
                self._initialize_storage_service(),
                self._initialize_cosmos_service(),
                self._initialize_ml_service(),
                self._initialize_app_insights_service(),
            ]
            
            # Execute all initializations in parallel with timeout
            initialization_results = await asyncio.wait_for(
                asyncio.gather(*initialization_tasks, return_exceptions=True),
                timeout=30.0  # 30 second timeout for all services
            )
            
            # Process results and create service instances
            for result in initialization_results:
                if isinstance(result, Exception):
                    logger.error(f"Service initialization failed with exception: {result}")
                    self.initialization_results.append(ServiceInitializationResult(
                        service_name="unknown",
                        success=False,
                        initialization_time=0.0,
                        error_message=str(result)
                    ))
                elif isinstance(result, ServiceInitializationResult):
                    self.initialization_results.append(result)
                    
                    # Set service instance if successful
                    if result.success and result.client_instance:
                        setattr(self, result.service_name, result.client_instance)
            
            # Initialize vector service after openai_client is ready
            if self.openai_client:
                vector_result = await self._initialize_vector_service()
                self.initialization_results.append(vector_result)
                if vector_result.success:
                    self.vector_service = vector_result.client_instance
            
            self.initialization_end_time = datetime.utcnow()
            self.initialized = True
            
            # Log summary
            successful_services = sum(1 for r in self.initialization_results if r.success)
            total_services = len(self.initialization_results)
            total_time = (self.initialization_end_time - self.initialization_start_time).total_seconds()
            
            logger.info(f"✅ Azure services initialization completed: "
                       f"{successful_services}/{total_services} services initialized in {total_time:.2f}s")
            
            return self._get_initialization_summary()
            
        except asyncio.TimeoutError:
            logger.error("❌ Service initialization timed out after 30 seconds")
            raise ServiceInitializationError("Service initialization timeout")
        except Exception as e:
            logger.error(f"❌ Critical error during service initialization: {e}")
            raise ServiceInitializationError(f"Service initialization failed: {str(e)}") from e
    
    async def _initialize_openai_service(self) -> ServiceInitializationResult:
        """Initialize Azure OpenAI service asynchronously"""
        start_time = time.time()
        
        try:
            logger.debug("Initializing Azure OpenAI client...")
            
            # Create client asynchronously (if the client supports async initialization)
            client = UnifiedAzureOpenAIClient()
            
            # Test client connectivity
            # Note: Add actual connectivity test here if needed
            
            initialization_time = time.time() - start_time
            endpoint = self.azure_settings.effective_openai_endpoint
            
            logger.info(f"✅ Azure OpenAI client initialized in {initialization_time:.2f}s - endpoint: {endpoint}")
            
            return ServiceInitializationResult(
                service_name="openai_client",
                success=True,
                initialization_time=initialization_time,
                client_instance=client
            )
            
        except Exception as e:
            initialization_time = time.time() - start_time
            logger.error(f"❌ Azure OpenAI initialization failed in {initialization_time:.2f}s: {e}")
            
            return ServiceInitializationResult(
                service_name="openai_client",
                success=False,
                initialization_time=initialization_time,
                error_message=str(e)
            )
    
    async def _initialize_search_service(self) -> ServiceInitializationResult:
        """Initialize Azure Cognitive Search service asynchronously"""
        start_time = time.time()
        
        try:
            logger.debug("Initializing Azure Cognitive Search client...")
            
            client = UnifiedSearchClient()
            
            # Perform async initialization if client supports it
            if hasattr(client, 'initialize_async'):
                await client.initialize_async()
            
            initialization_time = time.time() - start_time
            endpoint = self.azure_settings.effective_search_endpoint
            
            logger.info(f"✅ Azure Cognitive Search initialized in {initialization_time:.2f}s - endpoint: {endpoint}")
            
            return ServiceInitializationResult(
                service_name="search_service",
                success=True,
                initialization_time=initialization_time,
                client_instance=client
            )
            
        except Exception as e:
            initialization_time = time.time() - start_time
            logger.error(f"❌ Azure Search initialization failed in {initialization_time:.2f}s: {e}")
            
            return ServiceInitializationResult(
                service_name="search_service",
                success=False,
                initialization_time=initialization_time,
                error_message=str(e)
            )
    
    async def _initialize_storage_service(self) -> ServiceInitializationResult:
        """Initialize Azure Blob Storage service asynchronously"""
        start_time = time.time()
        
        try:
            logger.debug("Initializing Azure Blob Storage client...")
            
            client = UnifiedStorageClient()
            
            # Perform async initialization if client supports it
            if hasattr(client, 'initialize_async'):
                await client.initialize_async()
            
            initialization_time = time.time() - start_time
            account = self.azure_settings.azure_storage_account
            
            logger.info(f"✅ Azure Blob Storage initialized in {initialization_time:.2f}s - account: {account}")
            
            return ServiceInitializationResult(
                service_name="storage_client",
                success=True,
                initialization_time=initialization_time,
                client_instance=client
            )
            
        except Exception as e:
            initialization_time = time.time() - start_time
            logger.error(f"❌ Azure Storage initialization failed in {initialization_time:.2f}s: {e}")
            
            return ServiceInitializationResult(
                service_name="storage_client",
                success=False,
                initialization_time=initialization_time,
                error_message=str(e)
            )
    
    async def _initialize_cosmos_service(self) -> ServiceInitializationResult:
        """Initialize Azure Cosmos DB service asynchronously"""
        start_time = time.time()
        
        try:
            logger.debug("Initializing Azure Cosmos DB client...")
            
            client = AzureCosmosGremlinClient()
            
            # Perform async initialization if client supports it
            if hasattr(client, 'initialize_async'):
                await client.initialize_async()
            
            initialization_time = time.time() - start_time
            endpoint = self.azure_settings.azure_cosmos_endpoint
            database = self.azure_settings.cosmos_database_name
            
            logger.info(f"✅ Azure Cosmos DB initialized in {initialization_time:.2f}s - endpoint: {endpoint}, database: {database}")
            
            return ServiceInitializationResult(
                service_name="cosmos_client",
                success=True,
                initialization_time=initialization_time,
                client_instance=client
            )
            
        except Exception as e:
            initialization_time = time.time() - start_time
            logger.error(f"❌ Azure Cosmos DB initialization failed in {initialization_time:.2f}s: {e}")
            
            return ServiceInitializationResult(
                service_name="cosmos_client",
                success=False,
                initialization_time=initialization_time,
                error_message=str(e)
            )
    
    async def _initialize_ml_service(self) -> ServiceInitializationResult:
        """Initialize Azure ML service asynchronously"""
        start_time = time.time()
        
        try:
            logger.debug("Initializing Azure ML client...")
            
            client = AzureMLClient()
            
            # Perform async initialization if client supports it
            if hasattr(client, 'initialize_async'):
                await client.initialize_async()
            
            initialization_time = time.time() - start_time
            workspace = self.azure_settings.azure_ml_workspace_name
            
            logger.info(f"✅ Azure ML Workspace initialized in {initialization_time:.2f}s - workspace: {workspace}")
            
            return ServiceInitializationResult(
                service_name="ml_client",
                success=True,
                initialization_time=initialization_time,
                client_instance=client
            )
            
        except Exception as e:
            initialization_time = time.time() - start_time
            logger.error(f"❌ Azure ML initialization failed in {initialization_time:.2f}s: {e}")
            
            return ServiceInitializationResult(
                service_name="ml_client",
                success=False,
                initialization_time=initialization_time,
                error_message=str(e)
            )
    
    async def _initialize_app_insights_service(self) -> ServiceInitializationResult:
        """Initialize Azure Application Insights service asynchronously"""
        start_time = time.time()
        
        try:
            logger.debug("Initializing Azure Application Insights client...")
            
            client = AzureApplicationInsightsClient()
            
            # Perform async initialization if client supports it
            if hasattr(client, 'initialize_async'):
                await client.initialize_async()
            
            initialization_time = time.time() - start_time
            connection_string = self.azure_settings.applicationinsights_connection_string
            
            if connection_string:
                logger.info(f"✅ Azure Application Insights initialized in {initialization_time:.2f}s - azd managed")
            else:
                logger.info(f"✅ Azure Application Insights initialized in {initialization_time:.2f}s - legacy config")
            
            return ServiceInitializationResult(
                service_name="app_insights",
                success=True,
                initialization_time=initialization_time,
                client_instance=client
            )
            
        except Exception as e:
            initialization_time = time.time() - start_time
            logger.error(f"❌ Application Insights initialization failed in {initialization_time:.2f}s: {e}")
            
            return ServiceInitializationResult(
                service_name="app_insights",
                success=False,
                initialization_time=initialization_time,
                error_message=str(e)
            )
    
    async def _initialize_vector_service(self) -> ServiceInitializationResult:
        """Initialize Vector service asynchronously (depends on OpenAI client)"""
        start_time = time.time()
        
        try:
            logger.debug("Initializing Vector service...")
            
            # Import here to avoid circular dependencies
            from services.vector_service import VectorService
            
            client = VectorService(infrastructure_service=self)
            
            # Perform async initialization if client supports it
            if hasattr(client, 'initialize_async'):
                await client.initialize_async()
            
            initialization_time = time.time() - start_time
            logger.info(f"✅ Vector service initialized in {initialization_time:.2f}s")
            
            return ServiceInitializationResult(
                service_name="vector_service",
                success=True,
                initialization_time=initialization_time,
                client_instance=client
            )
            
        except Exception as e:
            initialization_time = time.time() - start_time
            logger.error(f"❌ Vector service initialization failed in {initialization_time:.2f}s: {e}")
            
            return ServiceInitializationResult(
                service_name="vector_service",
                success=False,
                initialization_time=initialization_time,
                error_message=str(e)
            )
    
    def _get_initialization_summary(self) -> Dict[str, Any]:
        """Get summary of initialization results"""
        if not self.initialization_results:
            return {"status": "not_initialized", "services": []}
        
        successful_services = [r for r in self.initialization_results if r.success]
        failed_services = [r for r in self.initialization_results if not r.success]
        
        total_time = 0.0
        if self.initialization_start_time and self.initialization_end_time:
            total_time = (self.initialization_end_time - self.initialization_start_time).total_seconds()
        
        return {
            "status": "initialized" if self.initialized else "partially_initialized",
            "total_initialization_time": total_time,
            "services": {
                "successful": len(successful_services),
                "failed": len(failed_services),
                "total": len(self.initialization_results)
            },
            "successful_services": [
                {
                    "name": r.service_name,
                    "initialization_time": r.initialization_time
                }
                for r in successful_services
            ],
            "failed_services": [
                {
                    "name": r.service_name,
                    "error": r.error_message,
                    "initialization_time": r.initialization_time
                }
                for r in failed_services
            ]
        }
    
    async def health_check_async(self) -> Dict[str, Any]:
        """Perform comprehensive async health check of all services"""
        if not self.initialized:
            return {
                "status": "unhealthy",
                "reason": "Services not initialized",
                "services": {}
            }
        
        health_tasks = []
        service_names = []
        
        # Create health check tasks for each initialized service
        services_to_check = [
            ("openai", self.openai_client),
            ("search", self.search_service),
            ("storage", self.storage_client),
            ("cosmos", self.cosmos_client),
            ("ml", self.ml_client),
            ("app_insights", self.app_insights),
            ("vector", self.vector_service)
        ]
        
        for service_name, service_client in services_to_check:
            if service_client:
                service_names.append(service_name)
                # Add health check task if service supports it
                if hasattr(service_client, 'health_check_async'):
                    health_tasks.append(service_client.health_check_async())
                else:
                    # Simple availability check
                    health_tasks.append(self._simple_health_check(service_name, service_client))
        
        # Execute all health checks in parallel
        if health_tasks:
            health_results = await asyncio.gather(*health_tasks, return_exceptions=True)
        else:
            health_results = []
        
        # Process results
        services_health = {}
        healthy_count = 0
        
        for i, result in enumerate(health_results):
            service_name = service_names[i] if i < len(service_names) else f"unknown_{i}"
            
            if isinstance(result, Exception):
                services_health[service_name] = {
                    "status": "unhealthy",
                    "error": str(result)
                }
            else:
                services_health[service_name] = result
                if result.get("status") == "healthy":
                    healthy_count += 1
        
        overall_status = "healthy" if healthy_count == len(services_health) else "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "services": services_health,
            "summary": {
                "total_services": len(services_health),
                "healthy_services": healthy_count,
                "unhealthy_services": len(services_health) - healthy_count
            }
        }
    
    async def _simple_health_check(self, service_name: str, service_client: Any) -> Dict[str, Any]:
        """Simple health check for services without async health check method"""
        try:
            # Basic availability check - service exists and is not None
            if service_client is None:
                return {
                    "status": "unhealthy",
                    "reason": "Service client is None"
                }
            
            return {
                "status": "healthy",
                "service_type": type(service_client).__name__
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def shutdown_async(self) -> None:
        """Properly shutdown all services asynchronously"""
        logger.info("🔄 Starting async shutdown of Azure services...")
        
        shutdown_tasks = []
        
        # Create shutdown tasks for services that support async shutdown
        services_to_shutdown = [
            ("openai", self.openai_client),
            ("search", self.search_service),
            ("storage", self.storage_client),
            ("cosmos", self.cosmos_client),
            ("ml", self.ml_client),
            ("app_insights", self.app_insights),
            ("vector", self.vector_service)
        ]
        
        for service_name, service_client in services_to_shutdown:
            if service_client and hasattr(service_client, 'shutdown_async'):
                shutdown_tasks.append(self._shutdown_service_safely(service_name, service_client))
        
        # Execute all shutdowns in parallel
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        # Reset state
        self.initialized = False
        self.openai_client = None
        self.search_service = None
        self.storage_client = None
        self.cosmos_client = None
        self.ml_client = None
        self.app_insights = None
        self.vector_service = None
        
        logger.info("✅ Azure services shutdown completed")
    
    async def _shutdown_service_safely(self, service_name: str, service_client: Any) -> None:
        """Safely shutdown a single service"""
        try:
            await service_client.shutdown_async()
            logger.debug(f"✅ {service_name} service shut down successfully")
        except Exception as e:
            logger.warning(f"⚠️ Error shutting down {service_name} service: {e}")


class ServiceInitializationError(Exception):
    """Exception raised when service initialization fails"""
    pass


# For backward compatibility, provide an alias
InfrastructureService = AsyncInfrastructureService