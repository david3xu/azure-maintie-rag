from src.retrieval.vector_search import MaintenanceVectorSearch
from config.settings import settings # Import settings to access embedding_dimension
from src.models.maintenance_models import MaintenanceDocument # Although not directly used in the provided test snippet, it's good practice to include it if the original guide mentioned it

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

        print(f"✅ Azure embedding generated")
        print(f"📊 Embedding shape: {embedding_result.shape}")
        print(f"📊 Expected dimension: {settings.embedding_dimension}")

        # Verify dimension matches Azure OpenAI ada-002
        assert embedding_result.shape[1] == 1536, f"Wrong dimension: {embedding_result.shape[1]}"

        return True

    except Exception as e:
        print(f"❌ Vector search error: {e}")
        print("🔍 Check Azure OpenAI credentials and endpoint")
        return False

if __name__ == "__main__":
    test_real_vector_search()
