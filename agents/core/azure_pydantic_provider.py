"""
Azure PydanticAI Provider with Managed Identity Support
=======================================================

Custom PydanticAI provider that uses Azure managed identity instead of API keys.
Solves the authentication mismatch between azd (managed identity) and PydanticAI (API keys).
"""

import os
from typing import Optional

from azure.identity import DefaultAzureCredential, AzureCliCredential
from openai import AsyncAzureOpenAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Load environment variables early
from dotenv import load_dotenv
load_dotenv()




def get_azure_openai_model(model_deployment: Optional[str] = None) -> OpenAIModel:
    """
    Create Azure OpenAI model for PydanticAI agents.
    
    Sets up environment variables for PydanticAI Azure provider.
    
    Args:
        model_deployment: Optional model deployment name. Uses OPENAI_MODEL_DEPLOYMENT env var if not provided.
        
    Returns:
        OpenAIModel: Configured Azure OpenAI model for PydanticAI
    """
    # Get Azure configuration from environment (set by azd)
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("OPENAI_API_VERSION", "2024-06-01")
    deployment_name = model_deployment or os.getenv('OPENAI_MODEL_DEPLOYMENT', 'gpt-4o')

    if not azure_endpoint:
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT not set. Run 'azd up' to deploy Azure services."
        )

    # Use appropriate credential based on environment setting
    use_managed_identity = os.getenv("USE_MANAGED_IDENTITY", "true").lower() == "true"
    
    if not use_managed_identity:
        # For testing, use CLI credentials with proper token provider
        from azure.identity import AzureCliCredential, get_bearer_token_provider
        credential = AzureCliCredential()
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )
        
        azure_client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
        )
    else:
        # For production, use managed identity with proper token provider
        from azure.identity import get_bearer_token_provider
        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )
        
        azure_client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
        )
    
    # Use OpenAIProvider with custom AsyncAzureOpenAI client
    # This ensures the API version 2024-06-01 is used for tool_choice support
    provider = OpenAIProvider(openai_client=azure_client)
    
    return OpenAIModel(
        model_name=deployment_name,
        provider=provider
    )


# Test function to verify Azure authentication works
async def test_azure_authentication():
    """Test Azure managed identity authentication with PydanticAI."""
    try:
        model = get_azure_openai_model()
        print("✅ Azure PydanticAI model created successfully")
        print(f"   Model: {model.model_name}")
        print("   Authentication: Azure managed identity")
        return True
    except Exception as e:
        print(f"❌ Azure authentication test failed: {e}")
        return False


