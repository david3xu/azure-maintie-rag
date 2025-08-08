"""
Authentication Debug Tests
========================

Tests to diagnose Azure authentication issues and verify credential setup.
"""

import os

import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class TestAuthenticationDiagnostics:
    """Debug Azure authentication setup."""

    def test_azure_cli_authentication(self):
        """Test Azure CLI authentication status."""
        try:
            from azure.identity import AzureCliCredential

            credential = AzureCliCredential()

            # Test getting a token for Azure Cognitive Services
            token = credential.get_token("https://cognitiveservices.azure.com/.default")

            assert token is not None
            assert hasattr(token, "token")
            assert len(token.token) > 0

            print("✅ Azure CLI Authentication: Working correctly")
            print(f"   Token type: {type(token.token)}")
            print(f"   Token length: {len(token.token)}")

        except Exception as e:
            print(f"❌ Azure CLI Authentication Failed: {e}")
            error_msg = str(e).lower()
            
            # Check for authentication issues
            if any(auth_error in error_msg for auth_error in [
                'authentication', 'credential', 'not authenticated', 'login required',
                'az login', 'no subscription', 'cli not found', 'please run'
            ]):
                pytest.skip(f"Azure CLI authentication required - run 'az login': {e}")
            
            # Check for network/service issues
            if any(network_error in error_msg for network_error in [
                'connection', 'timeout', 'network', 'dns', 'unreachable'
            ]):
                pytest.skip(f"Network connectivity issue: {e}")
            
            # For other errors, show details and fail
            raise

    def test_default_azure_credential(self):
        """Test DefaultAzureCredential setup."""
        try:
            from azure.identity import DefaultAzureCredential

            credential = DefaultAzureCredential()

            # Test getting a token for Azure Cognitive Services
            token = credential.get_token("https://cognitiveservices.azure.com/.default")

            assert token is not None
            assert hasattr(token, "token")
            assert len(token.token) > 0

            print("✅ DefaultAzureCredential: Working correctly")
            print(f"   Token type: {type(token.token)}")
            print(f"   Token length: {len(token.token)}")

        except Exception as e:
            print(f"❌ DefaultAzureCredential Failed: {e}")
            error_msg = str(e).lower()
            
            # Check for authentication issues
            if any(auth_error in error_msg for auth_error in [
                'authentication', 'credential', 'not authenticated', 'login required',
                'az login', 'no subscription', 'managed identity', 'environment credential'
            ]):
                pytest.skip(f"Azure credential authentication issue: {e}")
            
            # Check for network/service issues
            if any(network_error in error_msg for network_error in [
                'connection', 'timeout', 'network', 'dns', 'unreachable'
            ]):
                pytest.skip(f"Network connectivity issue: {e}")
                
            raise

    def test_environment_variables_diagnostic(self):
        """Diagnostic test for environment variables."""
        required_vars = {
            "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "AZURE_OPENAI_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            "OPENAI_MODEL_DEPLOYMENT": os.getenv("OPENAI_MODEL_DEPLOYMENT"),
            "EMBEDDING_MODEL_DEPLOYMENT": os.getenv("EMBEDDING_MODEL_DEPLOYMENT"),
            "AZURE_COSMOS_ENDPOINT": os.getenv("AZURE_COSMOS_ENDPOINT"),
            "AZURE_SEARCH_ENDPOINT": os.getenv("AZURE_SEARCH_ENDPOINT"),
            "AZURE_STORAGE_ACCOUNT": os.getenv("AZURE_STORAGE_ACCOUNT"),
            "AZURE_ENV_NAME": os.getenv("AZURE_ENV_NAME"),
            "USE_MANAGED_IDENTITY": os.getenv("USE_MANAGED_IDENTITY"),
            "TEST_USE_MANAGED_IDENTITY": os.getenv("TEST_USE_MANAGED_IDENTITY"),
        }

        print("\n🔍 Environment Variables Diagnostic:")
        for var_name, var_value in required_vars.items():
            status = "✅ SET" if var_value else "❌ MISSING"
            # Hide sensitive values but show they're set
            display_value = (
                f"***{var_value[-10:]}"
                if var_value and len(var_value) > 10
                else var_value
            )
            print(f"   {status} {var_name}: {display_value}")

        # Check critical variables
        critical_vars = ["AZURE_OPENAI_ENDPOINT", "OPENAI_MODEL_DEPLOYMENT"]
        missing_critical = [var for var in critical_vars if not required_vars[var]]

        assert (
            not missing_critical
        ), f"Missing critical environment variables: {missing_critical}"
        print("✅ Critical environment variables are set")

    def test_azure_openai_endpoint_format(self):
        """Test Azure OpenAI endpoint format."""
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        assert endpoint, "AZURE_OPENAI_ENDPOINT must be set"

        assert endpoint.startswith(
            "https://"
        ), f"Endpoint must start with https://: {endpoint}"
        assert (
            "openai.azure.com" in endpoint
        ), f"Endpoint must contain openai.azure.com: {endpoint}"

        print(f"✅ Azure OpenAI Endpoint Format: Valid ({endpoint})")

    @pytest.mark.asyncio
    async def test_bearer_token_provider(self):
        """Test Azure bearer token provider for OpenAI."""
        try:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider

            credential = DefaultAzureCredential()
            token_provider = get_bearer_token_provider(
                credential, "https://cognitiveservices.azure.com/.default"
            )

            # Test token provider (it returns a callable that returns the token)
            token = token_provider()

            assert token is not None
            assert isinstance(token, str)
            assert len(token) > 0

            print("✅ Bearer Token Provider: Working correctly")
            print(f"   Token type: {type(token)}")
            print(f"   Token length: {len(token)}")

        except Exception as e:
            print(f"❌ Bearer Token Provider Failed: {e}")
            error_msg = str(e).lower()
            
            # Check for authentication issues
            if any(auth_error in error_msg for auth_error in [
                'authentication', 'credential', 'not authenticated', 'login required',
                'az login', 'no subscription', 'token', 'bearer'
            ]):
                pytest.skip(f"Azure bearer token authentication issue: {e}")
            
            # Check for network/service issues  
            if any(network_error in error_msg for network_error in [
                'connection', 'timeout', 'network', 'dns', 'unreachable'
            ]):
                pytest.skip(f"Network connectivity issue: {e}")
                
            raise
