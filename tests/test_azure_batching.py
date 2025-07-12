from src.retrieval.vector_search import MaintenanceVectorSearch
from config.settings import settings

def test_azure_batch_limits():
    """Test Azure OpenAI batch size limitations"""
    print("🔄 Testing Azure Batch Limits...")

    vector_search = MaintenanceVectorSearch()

    # Test small batch (should work)
    small_batch = [f"test query {i}" for i in range(10)]
    try:
        embeddings = vector_search._get_embedding(small_batch)
        print(f"✅ Small batch (10) processed: {embeddings.shape}")
    except Exception as e:
        print(f"❌ Small batch failed: {e}")
        return False

    # Test medium batch (should work)
    medium_batch = [f"test query {i}" for i in range(100)]
    try:
        embeddings = vector_search._get_embedding(medium_batch)
        print(f"✅ Medium batch (100) processed: {embeddings.shape}")
    except Exception as e:
        print(f"❌ Medium batch failed: {e}")
        return False

    # Test large batch (may fail without proper batching)
    large_batch = [f"test query {i}" for i in range(3000)]
    try:
        embeddings = vector_search._get_embedding(large_batch)
        print(f"⚠️ Large batch (3000) processed: {embeddings.shape}")
        print("⚠️ This should not work without proper batching!")
    except Exception as e:
        print(f"✅ Large batch properly failed (expected): {e}")
        print("✅ Batch size limit protection working")

    return True

if __name__ == "__main__":
    test_azure_batch_limits()
