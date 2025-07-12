from src.pipeline.enhanced_rag import MaintIEEnhancedRAG
import json

def test_real_pipeline():
    """Test actual RAG pipeline with Azure OpenAI"""
    print("🔄 Testing Real RAG Pipeline...")

    # Initialize real pipeline
    rag = MaintIEEnhancedRAG()

    # Test component initialization
    print("📦 Initializing real components...")
    init_results = rag.initialize_components(force_rebuild=True) # Force rebuild here

    print(f"📊 Initialization Results:")
    for component, status in init_results.items():
        symbol = "✅" if status else "❌"
        print(f"   {symbol} {component}: {status}")

    # Test real query processing (if components work)
    if rag.components_initialized:
        test_query = "centrifugal pump bearing failure diagnosis"
        print(f"\n🔍 Testing real query: '{test_query}'")

        try:
            response = rag.process_query(test_query)

            print(f"✅ Real query processed successfully")
            print(f"📊 Response length: {len(response.generated_response)}")
            print(f"📊 Processing time: {response.processing_time:.2f}s")
            print(f"📊 Confidence: {response.confidence_score:.2f}")
            print(f"📊 Sources: {len(response.sources)}")

            # Validate Azure-specific response
            if "azure" in str(response.sources).lower():
                print("✅ Response generated via Azure OpenAI")

            return True

        except Exception as e:
            print(f"❌ Real pipeline error: {e}")
            return False
    else:
        print("⚠️ Components not initialized - check Azure credentials")
        return False

if __name__ == "__main__":
    test_real_pipeline()
