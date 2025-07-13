from src.pipeline.enhanced_rag import MaintIEEnhancedRAG
import json
import pytest # Added import for pytest

def test_real_pipeline():
    """Test actual RAG pipeline with Azure OpenAI"""
    print("🔄 Testing Real RAG Pipeline...")

    try:
        # Initialize real pipeline
        rag = MaintIEEnhancedRAG()

        # Test component initialization
        print("📦 Initializing real components...")
        init_results = rag.initialize_components(force_rebuild=True) # Force rebuild here

        print(f"📊 Initialization Results:")
        for component, status in init_results.items():
            symbol = "✅" if status else "❌"
            print(f"   {symbol} {component}: {status}")

        assert rag.components_initialized is True, "RAG pipeline components not initialized"

        # Test real query processing (if components work)
        test_query = "centrifugal pump bearing failure diagnosis"
        print(f"\n🔍 Testing real query: '{test_query}'")

        response = rag.process_query(test_query)

        assert response is not None, "No response from pipeline"
        assert response.generated_response is not None and len(response.generated_response) > 0, "Empty or missing generated_response"
        assert response.processing_time > 0, "Invalid processing time"
        assert response.confidence_score >= 0 and response.confidence_score <= 1, "Invalid confidence score"
        assert len(response.sources) > 0, "No sources found"

        print(f"✅ Real query processed successfully")
        print(f"📊 Response length: {len(response.generated_response)}")
        print(f"📊 Processing time: {response.processing_time:.2f}s")
        print(f"📊 Confidence: {response.confidence_score:.2f}")
        print(f"📊 Sources: {len(response.sources)}")

        # Validate Azure-specific response (optional, can be removed if not directly asserting Azure origin)
        # if "azure" in str(response.sources).lower():
        #     print("✅ Response generated via Azure OpenAI")

    except Exception as e:
        pytest.fail(f"Real pipeline test failed: {e}")

if __name__ == "__main__":
    test_real_pipeline()
