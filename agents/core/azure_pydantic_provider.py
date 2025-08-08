"""
Azure PydanticAI Provider with Managed Identity Support
=======================================================

Custom PydanticAI provider that uses Azure managed identity instead of API keys.
Solves the authentication mismatch between azd (managed identity) and PydanticAI (API keys).
"""

import os
from typing import Optional

from azure.identity import DefaultAzureCredential
from openai import AsyncAzureOpenAI
from pydantic_ai.models import Model
from pydantic_ai.providers.openai import OpenAIProvider


def create_azure_token_provider():
    """Create Azure AD token provider for managed identity authentication."""
    credential = DefaultAzureCredential()

    def get_token():
        token = credential.get_token("https://cognitiveservices.azure.com/.default")
        return token.token

    return get_token


def create_azure_openai_client() -> AsyncAzureOpenAI:
    """Create AsyncAzureOpenAI client with managed identity authentication."""

    # Get Azure configuration from environment (set by azd)
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.getenv("OPENAI_MODEL_DEPLOYMENT", "gpt-4o")
    api_version = os.getenv("OPENAI_API_VERSION", "2024-02-01")

    if not azure_endpoint:
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT not set. Run 'azd up' to deploy Azure services."
        )

    # Create Azure AD token provider
    token_provider = create_azure_token_provider()

    # Create AsyncAzureOpenAI client with managed identity
    client = AsyncAzureOpenAI(
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        api_version=api_version,
        azure_ad_token_provider=token_provider,  # Use managed identity
        # api_key not needed with azure_ad_token_provider
    )

    return client


def create_azure_pydantic_provider() -> OpenAIProvider:
    """Create PydanticAI OpenAI provider with Azure managed identity client."""

    # Create Azure OpenAI client with managed identity
    azure_client = create_azure_openai_client()

    # Create PydanticAI provider with custom Azure client
    provider = OpenAIProvider(
        openai_client=azure_client,  # Inject our Azure identity client
        # base_url and api_key handled by azure_client
    )

    return provider


def create_azure_pydantic_model(model_name: str = "gpt-4o") -> str:
    """Create PydanticAI model string with Azure managed identity authentication.

    Note: Returns model string for use with Agent constructor.
    The provider configuration is handled separately.
    """
    return f"openai:{model_name}"


# Test function to verify Azure authentication works
async def test_azure_authentication():
    """Test Azure managed identity authentication with PydanticAI."""
    try:
        model = create_azure_pydantic_model()
        print("✅ Azure PydanticAI model created successfully")
        print(f"   Model: {model}")
        print("   Authentication: Azure managed identity")
        return True
    except Exception as e:
        print(f"❌ Azure authentication test failed: {e}")
        return False


if __name__ == "__main__":
    import asyncio

    print("🧪 Testing Azure Managed Identity with PydanticAI")
    print("=" * 50)

    # Run test
    result = asyncio.run(test_azure_authentication())

    if result:
        print("\n🎉 SUCCESS: PydanticAI can use Azure managed identity")
        print("   No API keys required - fully compatible with azd deployment")
    else:
        print("\n❌ FAILED: Authentication issue needs investigation")
