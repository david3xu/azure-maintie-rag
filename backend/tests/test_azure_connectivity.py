from openai import AzureOpenAI
from config.settings import settings
import os
from src.retrieval.vector_search import MaintenanceVectorSearch
from src.generation.llm_interface import MaintenanceLLMInterface
import pytest

def test_azure_embedding_api():
    """Test Azure embedding API connectivity"""
    print("🔄 Testing Azure Embedding API Connectivity...")
    try:
        vector_search = MaintenanceVectorSearch()
        test_text = "This is a test sentence."
        embedding = vector_search._get_embedding([test_text])

        assert embedding is not None, "Embedding returned None"
        assert embedding.shape == (1, settings.embedding_dimension), "Embedding has incorrect shape"
        print(f"✅ Embedding received with shape: {embedding.shape}")
    except Exception as e:
        pytest.fail(f"Azure Embedding API test failed: {e}")

def test_azure_chat_api():
    """Test Azure chat API connectivity"""
    print("🔄 Testing Azure Chat API Connectivity...")
    try:
        llm_interface = MaintenanceLLMInterface()
        response = llm_interface.test_connection("Hello, what is your purpose?")

        assert response is not None, "Chat response returned None"
        assert isinstance(response, str), "Chat response is not a string"
        assert len(response) > 0, "Chat response is empty"
        print(f"✅ Chat response received: {response[:50]}...")
    except Exception as e:
        pytest.fail(f"Azure Chat API test failed: {e}")

if __name__ == "__main__":
    test_azure_embedding_api()
    test_azure_chat_api()
