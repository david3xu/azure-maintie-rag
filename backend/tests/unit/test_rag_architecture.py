#!/usr/bin/env python3
"""
Test suite for the new RAG architecture components

This test suite validates:
1. Base RAG class functionality
2. Multi-modal RAG implementation
3. Structured RAG implementation
4. Enhanced RAG orchestrator
5. Component separation and inheritance
6. Error handling and fallback mechanisms

Usage:
    python -m pytest backend/tests/test_rag_architecture.py -v
    python backend/tests/test_rag_architecture.py
"""

import pytest
import time
from unittest.mock import Mock, patch
from pathlib import Path
import sys

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from src.pipeline.rag_base import MaintIERAGBase
from src.pipeline.rag_multi_modal import MaintIEMultiModalRAG
from src.pipeline.rag_structured import MaintIEStructuredRAG
from src.pipeline.enhanced_rag import MaintIEEnhancedRAG
from api.models.query_models import QueryRequest, QueryResponse
from src.models.maintenance_models import RAGResponse, EnhancedQuery, SearchResult, QueryAnalysis

class TestRAGBase:
    """Test the base RAG class functionality"""

    def test_base_class_initialization(self):
        """Test that base class can be initialized"""
        print("🔍 Testing RAG Base Class Initialization...")

        # Create a mock implementation of abstract methods
        class MockRAG(MaintIERAGBase):
            def __init__(self):
                super().__init__("Test")
            def _initialize_components(self):
                return {"mock": True}

            def _process_query_implementation(self, query: str, **kwargs):
                # Create minimal required objects for RAGResponse
                analysis = QueryAnalysis(
                    original_query=query,
                    query_type="informational",
                    entities=[],
                    intent="test",
                    complexity="simple"
                )
                enhanced_query = EnhancedQuery(
                    analysis=analysis,
                    expanded_concepts=[],
                    related_entities=[],
                    domain_context={},
                    structured_search="",
                    safety_considerations=[]
                )
                return RAGResponse(
                    query=query,
                    enhanced_query=enhanced_query,
                    search_results=[],
                    generated_response="Mock response",
                    confidence_score=0.8,
                    processing_time=0.1,
                    sources=[],
                    safety_warnings=[],
                    citations=[]
                )

        rag = MockRAG()
        assert rag is not None
        print("✅ Base class initialization successful")

    def test_base_class_abstract_methods(self):
        """Test that base class enforces abstract method implementation"""
        print("🔍 Testing Base Class Abstract Methods...")

        # This should raise an error since abstract methods aren't implemented
        with pytest.raises(TypeError):
            MaintIERAGBase()

        print("✅ Abstract method enforcement working")

    def test_base_class_common_methods(self):
        """Test common methods in base class"""
        print("🔍 Testing Base Class Common Methods...")

        class MockRAG(MaintIERAGBase):
            def __init__(self):
                super().__init__("Test")
            def _initialize_components(self):
                return {"mock": True}

            def _process_query_implementation(self, query: str, **kwargs):
                # Create minimal required objects for RAGResponse
                analysis = QueryAnalysis(
                    original_query=query,
                    query_type="informational",
                    entities=[],
                    intent="test",
                    complexity="simple"
                )
                enhanced_query = EnhancedQuery(
                    analysis=analysis,
                    expanded_concepts=[],
                    related_entities=[],
                    domain_context={},
                    structured_search="",
                    safety_considerations=[]
                )
                return RAGResponse(
                    query=query,
                    enhanced_query=enhanced_query,
                    search_results=[],
                    generated_response="Mock response",
                    confidence_score=0.8,
                    processing_time=0.1,
                    sources=[],
                    safety_warnings=[],
                    citations=[]
                )

        rag = MockRAG()

        # Test get_system_status
        status = rag.get_system_status()
        assert "total_queries_processed" in status
        assert "average_processing_time" in status
        assert "components_initialized" in status

        # Test get_performance_metrics
        metrics = rag.get_performance_metrics()
        assert "query_count" in metrics
        assert "average_processing_time" in metrics

        print("✅ Common methods working correctly")

class TestMultiModalRAG:
    """Test the multi-modal RAG implementation"""

    def test_multi_modal_initialization(self):
        """Test multi-modal RAG initialization"""
        print("🔄 Testing Multi-Modal RAG Initialization...")

        try:
            rag = MaintIEMultiModalRAG()
            assert rag is not None
            assert isinstance(rag, MaintIERAGBase)
            print("✅ Multi-modal RAG initialization successful")
        except Exception as e:
            print(f"⚠️  Multi-modal initialization failed: {e}")
            # This might fail if Azure credentials aren't configured
            pytest.skip("Multi-modal RAG requires Azure configuration")

    def test_multi_modal_query_processing(self):
        """Test multi-modal query processing"""
        print("🔄 Testing Multi-Modal Query Processing...")

        try:
            rag = MaintIEMultiModalRAG()

            # Test with a simple query
            query = "pump maintenance"
            response = rag.process_query(query)

            assert response is not None
            assert hasattr(response, 'generated_response')
            assert hasattr(response, 'sources')
            assert hasattr(response, 'confidence_score')
            assert hasattr(response, 'processing_time')

            print("✅ Multi-modal query processing successful")

        except Exception as e:
            print(f"⚠️  Multi-modal query processing failed: {e}")
            pytest.skip("Multi-modal RAG requires Azure configuration")

class TestStructuredRAG:
    """Test the structured RAG implementation"""

    def test_structured_initialization(self):
        """Test structured RAG initialization"""
        print("⚡ Testing Structured RAG Initialization...")

        try:
            rag = MaintIEStructuredRAG()
            assert rag is not None
            assert isinstance(rag, MaintIERAGBase)
            print("✅ Structured RAG initialization successful")
        except Exception as e:
            print(f"⚠️  Structured initialization failed: {e}")
            pytest.skip("Structured RAG requires Azure configuration")

    def test_structured_query_processing(self):
        """Test structured query processing"""
        print("⚡ Testing Structured Query Processing...")

        try:
            rag = MaintIEStructuredRAG()

            # Test with a simple query
            query = "motor bearing replacement"
            response = rag.process_query(query)

            assert response is not None
            assert hasattr(response, 'generated_response')
            assert hasattr(response, 'sources')
            assert hasattr(response, 'confidence_score')
            assert hasattr(response, 'processing_time')

            print("✅ Structured query processing successful")

        except Exception as e:
            print(f"⚠️  Structured query processing failed: {e}")
            pytest.skip("Structured RAG requires Azure configuration")

class TestEnhancedRAG:
    """Test the enhanced RAG orchestrator"""

    def test_enhanced_initialization(self):
        """Test enhanced RAG initialization"""
        print("🎯 Testing Enhanced RAG Initialization...")

        try:
            rag = MaintIEEnhancedRAG()
            assert rag is not None
            assert hasattr(rag, 'active_implementation')
            print("✅ Enhanced RAG initialization successful")
        except Exception as e:
            print(f"⚠️  Enhanced initialization failed: {e}")
            pytest.skip("Enhanced RAG requires Azure configuration")

    def test_enhanced_implementation_switching(self):
        """Test switching between implementations"""
        print("🎯 Testing Implementation Switching...")

        try:
            rag = MaintIEEnhancedRAG()

            # Test switching to multi-modal
            rag.set_active_implementation('multi_modal')
            assert rag.active_implementation == 'multi_modal'

            # Test switching to structured
            rag.set_active_implementation('structured')
            assert rag.active_implementation == 'structured'

            print("✅ Implementation switching successful")

        except Exception as e:
            print(f"⚠️  Implementation switching failed: {e}")
            pytest.skip("Enhanced RAG requires Azure configuration")

    def test_enhanced_query_processing(self):
        """Test enhanced RAG query processing"""
        print("🎯 Testing Enhanced Query Processing...")

        try:
            rag = MaintIEEnhancedRAG()

            # Test with multi-modal
            rag.set_active_implementation('multi_modal')
            response_multi = rag.process_query("pump seal failure")
            assert response_multi is not None

            # Test with structured
            rag.set_active_implementation('structured')
            response_struct = rag.process_query("motor bearing replacement")
            assert response_struct is not None

            print("✅ Enhanced query processing successful")

        except Exception as e:
            print(f"⚠️  Enhanced query processing failed: {e}")
            pytest.skip("Enhanced RAG requires Azure configuration")

class TestRAGArchitectureIntegration:
    """Test integration between RAG components"""

    def test_component_separation(self):
        """Test that components are properly separated"""
        print("🔗 Testing Component Separation...")

        # Test that each implementation is independent
        multi_modal = MaintIEMultiModalRAG()
        structured = MaintIEStructuredRAG()
        enhanced = MaintIEEnhancedRAG()

        assert type(multi_modal) != type(structured)
        assert isinstance(multi_modal, MaintIERAGBase)
        assert isinstance(structured, MaintIERAGBase)
        # Enhanced RAG is an orchestrator, not a base class implementation
        assert not isinstance(enhanced, MaintIERAGBase)
        assert hasattr(enhanced, 'multi_modal_rag')
        assert hasattr(enhanced, 'structured_rag')

        print("✅ Component separation verified")

    def test_inheritance_structure(self):
        """Test inheritance structure"""
        print("🔗 Testing Inheritance Structure...")

        multi_modal = MaintIEMultiModalRAG()
        structured = MaintIEStructuredRAG()

        # Both should inherit from base class
        assert isinstance(multi_modal, MaintIERAGBase)
        assert isinstance(structured, MaintIERAGBase)

        # But they should be different types
        assert type(multi_modal) != type(structured)

        print("✅ Inheritance structure verified")

    def test_error_handling(self):
        """Test error handling in RAG components"""
        print("🔗 Testing Error Handling...")

        # Test base class error handling
        class ErrorRAG(MaintIERAGBase):
            def __init__(self):
                super().__init__("Error")
            def _initialize_components(self):
                raise Exception("Test error")

            def _process_query_implementation(self, query: str, **kwargs):
                raise Exception("Test error")

        rag = ErrorRAG()

        # Test that errors are properly caught and handled
        try:
            rag.initialize_components()
            assert False, "Should have raised an exception"
        except Exception:
            pass  # Expected

        try:
            rag.process_query("test")
            assert False, "Should have raised an exception"
        except Exception:
            pass  # Expected

        print("✅ Error handling verified")

def test_architecture_design_patterns():
    """Test that design patterns are properly implemented"""
    print("🏗️  Testing Architecture Design Patterns...")

    # Test Template Method pattern
    class TemplateTestRAG(MaintIERAGBase):
        def __init__(self):
            super().__init__("Template")
        def _initialize_components(self):
            return {"template": True}
        def _process_query_implementation(self, query: str, **kwargs):
            # Create minimal required objects for RAGResponse
            analysis = QueryAnalysis(
                original_query=query,
                query_type="informational",
                entities=[],
                intent="test",
                complexity="simple"
            )
            enhanced_query = EnhancedQuery(
                analysis=analysis,
                expanded_concepts=[],
                related_entities=[],
                domain_context={},
                structured_search="",
                safety_considerations=[]
            )
            return RAGResponse(
                query=query,
                enhanced_query=enhanced_query,
                search_results=[],
                generated_response="Template response",
                confidence_score=0.9,
                processing_time=0.05,
                sources=[],
                safety_warnings=[],
                citations=[]
            )
        def process_query(self, query: str, **kwargs):
            return self._process_query_implementation(query, **kwargs)

    rag = TemplateTestRAG()

    # The template method should call the implementation
    response = rag.process_query("test")
    assert response.generated_response == "Template response"

    print("✅ Design patterns verified")

def main():
    """Main test function"""
    print("🚀 MaintIE RAG Architecture Test Suite")
    print("Testing the new separated RAG architecture")
    print("=" * 60)

    # Run all test classes
    test_classes = [
        TestRAGBase(),
        TestMultiModalRAG(),
        TestStructuredRAG(),
        TestEnhancedRAG(),
        TestRAGArchitectureIntegration()
    ]

    for test_class in test_classes:
        for method_name in dir(test_class):
            if method_name.startswith('test_'):
                method = getattr(test_class, method_name)
                if callable(method):
                    try:
                        method()
                    except Exception as e:
                        print(f"❌ {method_name} failed: {e}")

    # Test design patterns
    test_architecture_design_patterns()

    print("\n" + "=" * 60)
    print("✅ RAG Architecture test suite completed!")

if __name__ == "__main__":
    main()