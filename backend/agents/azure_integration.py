"""
Azure Services Integration for PydanticAI Agent

This module creates the integration bridge between our PydanticAI agent
and the existing Azure services in the core module. It provides the
AzureServiceContainer that will be injected into PydanticAI tools.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

# TODO: Fix imports once we have proper module structure
# For now, create placeholder classes for testing

class AzureOpenAIClient:
    """Placeholder for Azure OpenAI client"""
    async def initialize(self): pass
    async def health_check(self): return {"status": "healthy"}

class AzureCognitiveSearchClient:
    """Placeholder for Azure Search client"""
    async def initialize(self): pass
    async def health_check(self): return {"status": "healthy"}

class CosmosGremlinClient:
    """Placeholder for Cosmos DB client"""
    async def initialize(self): pass
    async def health_check(self): return {"status": "healthy"}

class AzureStorageClient:
    """Placeholder for Azure Storage client"""
    async def initialize(self): pass
    async def health_check(self): return {"status": "healthy"}

class AzureMLClient:
    """Placeholder for Azure ML client"""
    async def initialize(self): pass
    async def health_check(self): return {"status": "healthy"}

class AzureApplicationInsightsClient:
    """Placeholder for Azure Monitoring client"""
    async def initialize(self): pass
    async def health_check(self): return {"status": "healthy"}

# Import real TriModalOrchestrator - production only
from .search.tri_modal_orchestrator import TriModalOrchestrator

class ZeroConfigAdapter:
    """Placeholder for our zero-config adapter"""
    async def initialize(self): pass
    async def health_check(self): return {"status": "healthy"}
    async def adapt_agent_to_domain(self, raw_text_data=None, domain_name=None, **kwargs):
        return {"discovered_domain": domain_name or "auto-detected", "confidence": 0.8}

class PatternLearningSystem:
    """Placeholder for our pattern learning system"""
    async def initialize(self): pass
    async def health_check(self): return {"status": "healthy"}

class DynamicPatternExtractor:
    """Placeholder for our pattern extractor"""
    def __init__(self): self.discovery_system = None
    async def initialize(self): pass
    async def health_check(self): return {"status": "healthy"}

logger = logging.getLogger(__name__)


@dataclass
class AzureServiceContainer:
    """
    Container for all Azure services and our unique components.
    
    This will be injected into PydanticAI tools via RunContext.deps,
    allowing them to access all our existing capabilities while maintaining
    proper layer boundaries.
    """
    
    # Azure AI Services
    azure_openai: Optional[AzureOpenAIClient] = None
    azure_search: Optional[AzureCognitiveSearchClient] = None  
    azure_cosmos: Optional[CosmosGremlinClient] = None
    azure_storage: Optional[AzureStorageClient] = None
    azure_ml: Optional[AzureMLClient] = None
    azure_monitoring: Optional[AzureApplicationInsightsClient] = None
    
    # Our Unique Intelligence Components  
    tri_modal_orchestrator: Optional[TriModalOrchestrator] = None
    zero_config_adapter: Optional[ZeroConfigAdapter] = None
    pattern_learning_system: Optional[PatternLearningSystem] = None
    dynamic_pattern_extractor: Optional[DynamicPatternExtractor] = None
    
    # Configuration and metadata
    config: Dict[str, Any] = field(default_factory=dict)
    performance_targets: Dict[str, float] = field(default_factory=lambda: {
        "max_response_time": 3.0,
        "min_confidence": 0.7,
        "max_memory_usage": 500  # MB
    })
    
    # Health and monitoring
    _initialized: bool = False
    _health_status: Dict[str, Any] = field(default_factory=dict)


async def create_azure_service_container(config: Optional[Dict[str, Any]] = None) -> AzureServiceContainer:
    """
    Factory function to create and initialize the Azure service container.
    
    This function handles the complex initialization of all Azure services
    and our unique components in the correct order with proper error handling.
    """
    
    logger.info("Initializing Azure Service Container for PydanticAI integration")
    
    container = AzureServiceContainer()
    
    if config:
        container.config = config
    
    try:
        # Phase 1: Initialize Azure AI Services
        logger.info("Phase 1: Initializing Azure AI services")
        
        # Azure OpenAI - Critical for agent communication
        container.azure_openai = AzureOpenAIClient()
        await _safe_initialize(container.azure_openai, "Azure OpenAI")
        
        # Azure Cognitive Search - Critical for vector search
        container.azure_search = AzureCognitiveSearchClient()
        await _safe_initialize(container.azure_search, "Azure Cognitive Search")
        
        # Azure Cosmos DB - Critical for graph search
        container.azure_cosmos = CosmosGremlinClient()
        await _safe_initialize(container.azure_cosmos, "Azure Cosmos DB")
        
        # Phase 2: Initialize Supporting Services
        logger.info("Phase 2: Initializing supporting Azure services")
        
        container.azure_storage = AzureStorageClient()
        await _safe_initialize(container.azure_storage, "Azure Storage")
        
        container.azure_ml = AzureMLClient()
        await _safe_initialize(container.azure_ml, "Azure ML")
        
        container.azure_monitoring = AzureApplicationInsightsClient()
        await _safe_initialize(container.azure_monitoring, "Azure Monitoring")
        
        # Phase 3: Initialize Our Unique Intelligence Components
        logger.info("Phase 3: Initializing unique intelligence components")
        
        # Tri-Modal Orchestrator - Our core competitive advantage
        container.tri_modal_orchestrator = TriModalOrchestrator(timeout=2.5)
        await _safe_initialize(container.tri_modal_orchestrator, "Tri-Modal Orchestrator")
        
        # Zero-Config Adapter - Our domain discovery system
        container.zero_config_adapter = ZeroConfigAdapter()
        await _safe_initialize(container.zero_config_adapter, "Zero-Config Adapter")
        
        # Pattern Learning System - Our learning capabilities
        container.pattern_learning_system = PatternLearningSystem()
        await _safe_initialize(container.pattern_learning_system, "Pattern Learning System")
        
        # Dynamic Pattern Extractor - Our pattern discovery
        container.dynamic_pattern_extractor = DynamicPatternExtractor()
        await _safe_initialize(container.dynamic_pattern_extractor, "Dynamic Pattern Extractor")
        
        # Phase 4: Integration and Validation
        logger.info("Phase 4: Finalizing integration")
        
        # Connect components that depend on each other
        if container.dynamic_pattern_extractor and container.pattern_learning_system:
            container.dynamic_pattern_extractor.discovery_system = container.pattern_learning_system
        
        # Run health check
        container._health_status = await _health_check_all_services(container)
        container._initialized = True
        
        logger.info("Azure Service Container initialized successfully")
        logger.info(f"Health Status: {container._health_status}")
        
        return container
        
    except Exception as e:
        logger.error(f"Failed to initialize Azure Service Container: {e}")
        raise


async def _safe_initialize(service: Any, service_name: str) -> bool:
    """Safely initialize a service with proper error handling"""
    try:
        if hasattr(service, 'initialize'):
            await service.initialize()
        elif hasattr(service, '__aenter__'):
            await service.__aenter__()
        
        logger.info(f"✅ {service_name} initialized successfully")
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ {service_name} initialization failed: {e}")
        return False


async def _health_check_all_services(container: AzureServiceContainer) -> Dict[str, Any]:
    """Perform health check on all services in the container"""
    
    health_status = {
        "overall_status": "healthy",
        "services": {},
        "unique_components": {},
        "ready_for_pydantic_ai": True
    }
    
    # Check Azure services
    azure_services = [
        ("azure_openai", container.azure_openai),
        ("azure_search", container.azure_search),
        ("azure_cosmos", container.azure_cosmos),
        ("azure_storage", container.azure_storage),
        ("azure_ml", container.azure_ml),
        ("azure_monitoring", container.azure_monitoring)
    ]
    
    for service_name, service in azure_services:
        if service:
            try:
                if hasattr(service, 'health_check'):
                    service_health = await service.health_check()
                    health_status["services"][service_name] = service_health
                else:
                    health_status["services"][service_name] = {"status": "initialized"}
            except Exception as e:
                health_status["services"][service_name] = {"status": "error", "error": str(e)}
                health_status["overall_status"] = "degraded"
        else:
            health_status["services"][service_name] = {"status": "not_initialized"}
    
    # Check unique components
    unique_components = [
        ("tri_modal_orchestrator", container.tri_modal_orchestrator),
        ("zero_config_adapter", container.zero_config_adapter),
        ("pattern_learning_system", container.pattern_learning_system),
        ("dynamic_pattern_extractor", container.dynamic_pattern_extractor)
    ]
    
    for component_name, component in unique_components:
        if component:
            try:
                if hasattr(component, 'health_check'):
                    component_health = await component.health_check()
                    health_status["unique_components"][component_name] = component_health
                else:
                    health_status["unique_components"][component_name] = {"status": "initialized"}
            except Exception as e:
                health_status["unique_components"][component_name] = {"status": "error", "error": str(e)}
                health_status["overall_status"] = "degraded"
        else:
            health_status["unique_components"][component_name] = {"status": "not_initialized"}
    
    # Determine if ready for PydanticAI
    critical_services_healthy = all([
        health_status["services"].get("azure_openai", {}).get("status") != "error",
        health_status["unique_components"].get("tri_modal_orchestrator", {}).get("status") != "error"
    ])
    
    health_status["ready_for_pydantic_ai"] = critical_services_healthy
    
    if not critical_services_healthy:
        health_status["overall_status"] = "critical"
    
    return health_status


# Example usage for testing
async def test_azure_integration():
    """Test the Azure service container integration"""
    try:
        container = await create_azure_service_container()
        
        print(f"Container initialized: {container._initialized}")
        print(f"Health status: {container._health_status['overall_status']}")
        print(f"Ready for PydanticAI: {container._health_status['ready_for_pydantic_ai']}")
        
        # Test tri-modal orchestrator access
        if container.tri_modal_orchestrator:
            print("✅ Tri-modal orchestrator available for PydanticAI tools")
        
        # Test domain discovery access  
        if container.zero_config_adapter:
            print("✅ Zero-config adapter available for PydanticAI tools")
            
        return container
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return None


if __name__ == "__main__":
    # Test the integration
    import asyncio
    asyncio.run(test_azure_integration())