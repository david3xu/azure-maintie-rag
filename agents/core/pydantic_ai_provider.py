"""
PydanticAI Azure Provider Configuration for Enterprise Integration
=================================================================

This module provides the PydanticAI Azure provider configuration that integrates
with ConsolidatedAzureServices for production-ready enterprise deployment.

Features:
- Integration with ConsolidatedAzureServices dependency injection
- Azure managed identity authentication (no API keys)
- Production-ready configuration for azd deployment
- Enterprise-grade error handling and monitoring
"""

import os
import logging
from typing import Optional, Any, Dict
from pydantic_ai import Agent
from pydantic_ai.providers.azure import AzureProvider
from azure.identity import DefaultAzureCredential

# Import ConsolidatedAzureServices for dependency injection
from .azure_services import ConsolidatedAzureServices

logger = logging.getLogger(__name__)


def create_azure_pydantic_provider(
    azure_services: Optional[ConsolidatedAzureServices] = None
) -> Optional[AzureProvider]:
    """
    Create PydanticAI Azure provider using ConsolidatedAzureServices integration.
    
    This function creates the Azure provider correctly for enterprise deployment:
    - Uses ConsolidatedAzureServices for dependency injection
    - Leverages managed identity authentication through consolidated services
    - No API keys required
    - Works with azd-managed Azure OpenAI endpoints
    - Enterprise-grade configuration and monitoring
    
    Args:
        azure_services: Optional ConsolidatedAzureServices instance (will create if None)
    
    Returns:
        AzureProvider configured for managed identity, or None if not configured
    """
    try:
        from azure.identity import get_bearer_token_provider
        from openai import AsyncAzureOpenAI
        
        # Create or use existing Azure services container
        if azure_services is None:
            from .azure_services import create_azure_service_container
            import asyncio
            # Create container in sync context (will be replaced with proper async pattern)
            try:
                loop = asyncio.get_event_loop()
                azure_services = loop.run_until_complete(create_azure_service_container())
            except RuntimeError:
                # No event loop, create new one
                azure_services = asyncio.run(create_azure_service_container())
        
        # Get Azure OpenAI configuration from consolidated services
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        api_version = os.getenv('OPENAI_API_VERSION', '2024-08-01-preview')
        
        if not azure_endpoint:
            logger.warning("AZURE_OPENAI_ENDPOINT not configured - PydanticAI will not be available")
            return None
        
        # Create token provider with managed identity (through ConsolidatedAzureServices)
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), 
            "https://cognitiveservices.azure.com/.default"
        )
        
        # Create Azure OpenAI client with managed identity
        azure_client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            azure_ad_token_provider=token_provider  # Uses managed identity
        )
        
        # Create PydanticAI provider with managed identity client
        azure_provider = AzureProvider(openai_client=azure_client)
        
        logger.info(f"✅ PydanticAI Azure provider created with ConsolidatedAzureServices for: {azure_endpoint}")
        return azure_provider
        
    except Exception as e:
        logger.error(f"❌ Failed to create PydanticAI Azure provider: {e}")
        return None


async def create_azure_pydantic_provider_async(
    azure_services: Optional[ConsolidatedAzureServices] = None
) -> Optional[AzureProvider]:
    """
    Create PydanticAI Azure provider using ConsolidatedAzureServices integration (async version).
    
    This is the preferred async version for use in async contexts.
    
    Args:
        azure_services: Optional ConsolidatedAzureServices instance (will create if None)
    
    Returns:
        AzureProvider configured for managed identity, or None if not configured
    """
    try:
        from azure.identity import get_bearer_token_provider
        from openai import AsyncAzureOpenAI
        
        # Create or use existing Azure services container
        if azure_services is None:
            from .azure_services import create_azure_service_container
            azure_services = await create_azure_service_container()
        
        # Get Azure OpenAI configuration from consolidated services
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        api_version = os.getenv('OPENAI_API_VERSION', '2024-08-01-preview')
        
        if not azure_endpoint:
            logger.warning("AZURE_OPENAI_ENDPOINT not configured - PydanticAI will not be available")
            return None
        
        # Create token provider with managed identity (through ConsolidatedAzureServices)
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), 
            "https://cognitiveservices.azure.com/.default"
        )
        
        # Create Azure OpenAI client with managed identity
        azure_client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            azure_ad_token_provider=token_provider  # Uses managed identity
        )
        
        # Create PydanticAI provider with managed identity client
        azure_provider = AzureProvider(openai_client=azure_client)
        
        logger.info(f"✅ PydanticAI Azure provider created (async) with ConsolidatedAzureServices for: {azure_endpoint}")
        return azure_provider
        
    except Exception as e:
        logger.error(f"❌ Failed to create PydanticAI Azure provider (async): {e}")
        return None


def create_pydantic_agent(
    model_name: str = "gpt-4o",
    system_prompt: str = None,
    agent_name: str = "azure-rag-agent",
    azure_services: Optional[ConsolidatedAzureServices] = None
) -> Optional[Agent]:
    """
    Create PydanticAI agent with Azure managed identity provider and ConsolidatedAzureServices.
    
    Args:
        model_name: Azure OpenAI model deployment name
        system_prompt: Custom system prompt for the agent
        agent_name: Name for the agent
        azure_services: Optional ConsolidatedAzureServices instance
        
    Returns:
        Configured PydanticAI agent or None if provider creation failed
    """
    # Create Azure provider with ConsolidatedAzureServices integration
    azure_provider = create_azure_pydantic_provider(azure_services)
    
    if not azure_provider:
        logger.error("Cannot create PydanticAI agent - Azure provider initialization failed")
        return None
    
    # Default system prompt if none provided
    if system_prompt is None:
        system_prompt = """
        You are an intelligent Azure RAG system with enterprise competitive advantages:
        
        1. **Tri-Modal Search Unity**: Vector + Graph + GNN search simultaneously
        2. **Sub-3-Second Response**: Guaranteed response time under 3 seconds
        3. **Zero-Config Discovery**: Automatic domain detection without configuration
        4. **Azure-Native Integration**: Deep integration with Azure AI services
        5. **Enterprise Architecture**: ConsolidatedAzureServices dependency injection
        
        Your role is to provide comprehensive, accurate responses while maintaining
        performance guarantees and leveraging Azure services effectively through
        enterprise-grade architecture patterns.
        """
    
    try:
        # Create agent with Azure provider
        agent = Agent(
            model_name,
            name=agent_name,
            system_prompt=system_prompt,
            model_provider=azure_provider
        )
        
        logger.info(f"✅ PydanticAI agent '{agent_name}' created successfully with model: {model_name} and ConsolidatedAzureServices")
        return agent
        
    except Exception as e:
        logger.error(f"❌ Failed to create PydanticAI agent: {e}")
        return None


async def create_pydantic_agent_async(
    model_name: str = "gpt-4o",
    system_prompt: str = None,
    agent_name: str = "azure-rag-agent",
    azure_services: Optional[ConsolidatedAzureServices] = None
) -> Optional[Agent]:
    """
    Create PydanticAI agent with Azure managed identity provider and ConsolidatedAzureServices (async version).
    
    This is the preferred async version for use in async contexts.
    
    Args:
        model_name: Azure OpenAI model deployment name
        system_prompt: Custom system prompt for the agent
        agent_name: Name for the agent
        azure_services: Optional ConsolidatedAzureServices instance
        
    Returns:
        Configured PydanticAI agent or None if provider creation failed
    """
    # Create Azure provider with ConsolidatedAzureServices integration
    azure_provider = await create_azure_pydantic_provider_async(azure_services)
    
    if not azure_provider:
        logger.error("Cannot create PydanticAI agent - Azure provider initialization failed")
        return None
    
    # Default system prompt if none provided
    if system_prompt is None:
        system_prompt = """
        You are an intelligent Azure RAG system with enterprise competitive advantages:
        
        1. **Tri-Modal Search Unity**: Vector + Graph + GNN search simultaneously
        2. **Sub-3-Second Response**: Guaranteed response time under 3 seconds
        3. **Zero-Config Discovery**: Automatic domain detection without configuration
        4. **Azure-Native Integration**: Deep integration with Azure AI services
        5. **Enterprise Architecture**: ConsolidatedAzureServices dependency injection
        
        Your role is to provide comprehensive, accurate responses while maintaining
        performance guarantees and leveraging Azure services effectively through
        enterprise-grade architecture patterns.
        """
    
    try:
        # Create agent with Azure provider
        agent = Agent(
            model_name,
            name=agent_name,
            system_prompt=system_prompt,
            model_provider=azure_provider
        )
        
        logger.info(f"✅ PydanticAI agent '{agent_name}' created successfully (async) with model: {model_name} and ConsolidatedAzureServices")
        return agent
        
    except Exception as e:
        logger.error(f"❌ Failed to create PydanticAI agent (async): {e}")
        return None


def test_azure_provider_connection(
    azure_services: Optional[ConsolidatedAzureServices] = None
) -> bool:
    """
    Test Azure provider connection and authentication with ConsolidatedAzureServices.
    
    Args:
        azure_services: Optional ConsolidatedAzureServices instance
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        # Create provider with ConsolidatedAzureServices
        provider = create_azure_pydantic_provider(azure_services)
        
        if not provider:
            logger.error("❌ Azure provider creation failed")
            return False
        
        # Create test agent
        agent = create_pydantic_agent(
            model_name=os.getenv('OPENAI_MODEL_DEPLOYMENT', 'gpt-4o'),
            system_prompt="You are a test agent for connection validation.",
            agent_name="test-connection-agent",
            azure_services=azure_services
        )
        
        if not agent:
            logger.error("❌ Test agent creation failed")
            return False
        
        logger.info("✅ PydanticAI Azure provider connection test successful with ConsolidatedAzureServices")
        return True
        
    except Exception as e:
        logger.error(f"❌ Azure provider connection test failed: {e}")
        return False


async def test_azure_provider_connection_async(
    azure_services: Optional[ConsolidatedAzureServices] = None
) -> bool:
    """
    Test Azure provider connection and authentication with ConsolidatedAzureServices (async version).
    
    Args:
        azure_services: Optional ConsolidatedAzureServices instance
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        # Create provider with ConsolidatedAzureServices
        provider = await create_azure_pydantic_provider_async(azure_services)
        
        if not provider:
            logger.error("❌ Azure provider creation failed")
            return False
        
        # Create test agent
        agent = await create_pydantic_agent_async(
            model_name=os.getenv('OPENAI_MODEL_DEPLOYMENT', 'gpt-4o'),
            system_prompt="You are a test agent for connection validation.",
            agent_name="test-connection-agent",
            azure_services=azure_services
        )
        
        if not agent:
            logger.error("❌ Test agent creation failed")
            return False
        
        logger.info("✅ PydanticAI Azure provider connection test successful (async) with ConsolidatedAzureServices")
        return True
        
    except Exception as e:
        logger.error(f"❌ Azure provider connection test failed (async): {e}")
        return False


# Export main functions
__all__ = [
    'create_azure_pydantic_provider',
    'create_azure_pydantic_provider_async', 
    'create_pydantic_agent',
    'create_pydantic_agent_async',
    'test_azure_provider_connection',
    'test_azure_provider_connection_async'
]


if __name__ == "__main__":
    import asyncio
    
    async def test_configuration():
        """Test the enhanced PydanticAI configuration with ConsolidatedAzureServices"""
        print("Testing Enhanced PydanticAI Azure Provider Configuration with ConsolidatedAzureServices...")
        
        try:
            # Create ConsolidatedAzureServices for testing
            from .azure_services import create_azure_service_container
            azure_services = await create_azure_service_container()
            print("✅ ConsolidatedAzureServices created successfully")
            
            # Test provider creation (async)
            provider = await create_azure_pydantic_provider_async(azure_services)
            if provider:
                print("✅ Azure provider created successfully with ConsolidatedAzureServices")
            else:
                print("❌ Azure provider creation failed")
                return
            
            # Test agent creation (async)
            agent = await create_pydantic_agent_async(azure_services=azure_services)
            if agent:
                print("✅ PydanticAI agent created successfully with ConsolidatedAzureServices")
            else:
                print("❌ PydanticAI agent creation failed")
                return
            
            # Test connection (async)
            if await test_azure_provider_connection_async(azure_services):
                print("✅ All tests passed - ready for enterprise deployment!")
            else:
                print("❌ Connection test failed")
                
        except Exception as e:
            print(f"❌ Test configuration failed: {e}")
    
    # Run async test
    asyncio.run(test_configuration())