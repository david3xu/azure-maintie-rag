from config.settings import settings
from pydantic_settings import BaseSettings
import pytest

def test_azure_config():
    """Test real Azure OpenAI configuration"""
    print("🔄 Testing Azure Configuration...")
    try:
        # Verify Azure settings
        assert settings.openai_api_type == "azure", f"Expected 'azure', got {settings.openai_api_type}"
        assert "openai.azure.com" in settings.openai_api_base, "Invalid Azure endpoint"
        assert settings.embedding_dimension == 1536, f"Expected 1536, got {settings.embedding_dimension}"

        print("✅ Azure configuration valid")
        print(f"📊 API Base: {settings.openai_api_base}")
        print(f"📊 Embedding Model: {settings.embedding_model}")
        print(f"📊 Batch Size: {settings.embedding_batch_size}")
    except Exception as e:
        pytest.fail(f"Azure config test failed: {e}")

if __name__ == "__main__":
    test_azure_config()
