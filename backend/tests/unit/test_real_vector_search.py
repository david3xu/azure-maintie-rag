from src.retrieval.vector_search import MaintenanceVectorSearch
from config.settings import settings # Import settings to access embedding_dimension
from src.models.maintenance_models import MaintenanceDocument # Although not directly used in the provided test snippet, it's good practice to include it if the original guide mentioned it
import pytest # Added import for pytest

def test_real_vector_search():
    """Test actual Azure-based vector search implementation"""
    print("🔄 Testing Real Vector Search (Azure OpenAI)...")

    try:
        # Initialize real vector search
        vector_search = MaintenanceVectorSearch()
        print("✅ Vector search initialized with Azure OpenAI")

        # Test embedding generation (real Azure API)
        test_text = "pump seal maintenance procedure"
        embedding_result = vector_search._get_embedding([test_text])

        assert embedding_result is not None, "Embedding result is None"
        assert embedding_result.shape[1] == settings.embedding_dimension, f"Wrong dimension: {embedding_result.shape[1]}"

        print(f"✅ Azure embedding generated")
        print(f"📊 Embedding shape: {embedding_result.shape}")
        print(f"📊 Expected dimension: {settings.embedding_dimension}")

    except Exception as e:
        pytest.fail(f"Vector search error: {e}\n🔍 Check Azure OpenAI credentials and endpoint")

if __name__ == "__main__":
    test_real_vector_search()
