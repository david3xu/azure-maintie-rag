#!/usr/bin/env python3
"""
Test Azure OpenAI connectivity using .env configuration
"""

import os
import asyncio
from pathlib import Path

# Load environment variables from .env file
def load_env():
    """Load environment variables from .env file"""
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("✅ Environment variables loaded from .env")
    else:
        print("❌ .env file not found")

async def test_azure_openai():
    """Test Azure OpenAI connection using .env configuration"""
    print("🔍 Testing Azure OpenAI Connection with .env configuration...")

    # Load .env file
    load_env()

    # Display configuration (without key)
    print(f"✅ Endpoint: {os.environ.get('AZURE_OPENAI_ENDPOINT')}")
    print(f"✅ API Version: {os.environ.get('AZURE_OPENAI_API_VERSION')}")
    print(f"✅ Model Deployment: {os.environ.get('OPENAI_MODEL_DEPLOYMENT')}")

    try:
        # Test with pydantic-ai
        from pydantic_ai.providers.azure import AzureProvider

        azure_provider = AzureProvider(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"]
        )

        print("✅ Azure OpenAI provider created successfully")

        # Test with the configured model deployment
        from pydantic_ai import Agent

        model_name = os.environ.get("OPENAI_MODEL_DEPLOYMENT", "gpt-4.1")
        agent = Agent(
            model=f"azure:{model_name}",
            name='test-agent',
            system_prompt="You are a helpful assistant for testing Azure OpenAI connectivity."
        )

        print(f"✅ PydanticAI agent created with model: {model_name}")

        # Test a simple query
        result = await agent.run("Respond with exactly: 'Azure OpenAI connection successful!'")
        print(f"✅ Test query successful: {result.data}")

        # Test embedding model if available
        try:
            embedding_model = os.environ.get("EMBEDDING_MODEL_DEPLOYMENT")
            if embedding_model:
                print(f"✅ Embedding model available: {embedding_model}")
        except Exception as e:
            print(f"⚠️ Embedding model test skipped: {e}")

        return True

    except Exception as e:
        print(f"❌ Azure OpenAI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_domain_agent_with_azure():
    """Test domain agent with Azure OpenAI"""
    print("\n🎯 Testing Domain Agent with Azure OpenAI...")

    try:
        # Import domain agent
        from agents.domain_intelligence.agent import get_domain_agent

        agent = get_domain_agent()
        print("✅ Domain Intelligence Agent loaded with Azure OpenAI")

        # Test a simple domain detection
        result = await agent.run("What domain is this query about: 'Aircraft hydraulic system maintenance procedures'?")
        print(f"✅ Domain detection test: {result.data}")

        return True

    except Exception as e:
        print(f"❌ Domain agent test failed: {e}")
        return False

if __name__ == "__main__":
    async def main():
        # Test basic Azure OpenAI connectivity
        openai_success = await test_azure_openai()

        if openai_success:
            print("\n🎉 Azure OpenAI connection working!")

            # Test domain agent
            agent_success = await test_domain_agent_with_azure()

            if agent_success:
                print("\n🚀 Domain Intelligence Agent working with Azure OpenAI!")
                return True
            else:
                print("\n⚠️ Azure OpenAI works but domain agent has issues")
                return False
        else:
            print("\n❌ Azure OpenAI connection failed!")
            return False

    success = asyncio.run(main())
    exit(0 if success else 1)
