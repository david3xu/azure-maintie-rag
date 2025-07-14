import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.enhanced_rag import MaintIEEnhancedRAG
from src.pipeline.rag_multi_modal import MaintIEMultiModalRAG
from src.pipeline.rag_structured import MaintIEStructuredRAG
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

    except Exception as e:
        pytest.fail(f"Real pipeline test failed: {e}")

def test_multi_modal_rag():
    """Test multi-modal RAG implementation specifically"""
    print("\n🔄 Testing Multi-Modal RAG Implementation...")

    try:
        # Initialize multi-modal RAG
        multi_modal_rag = MaintIEMultiModalRAG()

        # Test initialization
        print("📦 Initializing multi-modal components...")
        init_results = multi_modal_rag.initialize_components(force_rebuild=False)

        print(f"📊 Multi-modal Initialization Results:")
        for component, status in init_results.items():
            symbol = "✅" if status else "❌"
            print(f"   {symbol} {component}: {status}")

        # Test query processing
        test_query = "pump seal failure troubleshooting"
        print(f"\n🔍 Testing multi-modal query: '{test_query}'")

        response = multi_modal_rag.process_query(test_query)

        assert response is not None, "No response from multi-modal RAG"
        assert response.generated_response is not None and len(response.generated_response) > 0, "Empty or missing generated_response"
        assert response.processing_time > 0, "Invalid processing time"
        assert response.confidence_score >= 0 and response.confidence_score <= 1, "Invalid confidence score"
        assert len(response.sources) > 0, "No sources found"

        print(f"✅ Multi-modal query processed successfully")
        print(f"📊 Response length: {len(response.generated_response)}")
        print(f"📊 Processing time: {response.processing_time:.2f}s")
        print(f"📊 Confidence: {response.confidence_score:.2f}")
        print(f"📊 Sources: {len(response.sources)}")

    except Exception as e:
        pytest.fail(f"Multi-modal RAG test failed: {e}")

def test_structured_rag():
    """Test structured RAG implementation specifically"""
    print("\n⚡ Testing Structured RAG Implementation...")

    try:
        # Initialize structured RAG
        structured_rag = MaintIEStructuredRAG()

        # Test initialization
        print("📦 Initializing structured components...")
        init_results = structured_rag.initialize_components(force_rebuild=False)

        print(f"📊 Structured Initialization Results:")
        for component, status in init_results.items():
            symbol = "✅" if status else "❌"
            print(f"   {symbol} {component}: {status}")

        # Test query processing
        test_query = "motor bearing replacement procedure"
        print(f"\n🔍 Testing structured query: '{test_query}'")

        response = structured_rag.process_query(test_query)

        assert response is not None, "No response from structured RAG"
        assert response.generated_response is not None and len(response.generated_response) > 0, "Empty or missing generated_response"
        assert response.processing_time > 0, "Invalid processing time"
        assert response.confidence_score >= 0 and response.confidence_score <= 1, "Invalid confidence score"
        assert len(response.sources) > 0, "No sources found"

        print(f"✅ Structured query processed successfully")
        print(f"📊 Response length: {len(response.generated_response)}")
        print(f"📊 Processing time: {response.processing_time:.2f}s")
        print(f"📊 Confidence: {response.confidence_score:.2f}")
        print(f"📊 Sources: {len(response.sources)}")

    except Exception as e:
        pytest.fail(f"Structured RAG test failed: {e}")

def test_rag_comparison():
    """Test comparison between multi-modal and structured RAG"""
    print("\n🔬 Testing RAG Implementation Comparison...")

    try:
        # Initialize both implementations
        multi_modal_rag = MaintIEMultiModalRAG()
        structured_rag = MaintIEStructuredRAG()

        # Initialize components
        multi_modal_rag.initialize_components(force_rebuild=False)
        structured_rag.initialize_components(force_rebuild=False)

        # Test query
        test_query = "compressor vibration analysis"
        print(f"\n🔍 Comparing implementations for: '{test_query}'")

        # Process with multi-modal
        multi_modal_response = multi_modal_rag.process_query(test_query)

        # Process with structured
        structured_response = structured_rag.process_query(test_query)

        # Compare results
        print(f"📊 Multi-modal Results:")
        print(f"   Processing time: {multi_modal_response.processing_time:.2f}s")
        print(f"   Confidence: {multi_modal_response.confidence_score:.3f}")
        print(f"   Sources: {len(multi_modal_response.sources)}")

        print(f"📊 Structured Results:")
        print(f"   Processing time: {structured_response.processing_time:.2f}s")
        print(f"   Confidence: {structured_response.confidence_score:.3f}")
        print(f"   Sources: {len(structured_response.sources)}")

        # Calculate performance improvement
        if multi_modal_response.processing_time > 0:
            improvement = ((multi_modal_response.processing_time - structured_response.processing_time) / multi_modal_response.processing_time) * 100
            speedup = multi_modal_response.processing_time / structured_response.processing_time
            print(f"📊 Performance: {improvement:.1f}% faster ({speedup:.1f}x speedup)")

        # Validate both responses are valid
        assert multi_modal_response is not None, "Multi-modal response is None"
        assert structured_response is not None, "Structured response is None"
        assert len(multi_modal_response.generated_response) > 0, "Empty multi-modal response"
        assert len(structured_response.generated_response) > 0, "Empty structured response"

        print(f"✅ Both implementations working correctly")

    except Exception as e:
        pytest.fail(f"RAG comparison test failed: {e}")

if __name__ == "__main__":
    test_real_pipeline()
    test_multi_modal_rag()
    test_structured_rag()
    test_rag_comparison()
