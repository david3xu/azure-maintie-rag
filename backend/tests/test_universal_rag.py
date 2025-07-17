#!/usr/bin/env python3
"""
Universal RAG Test Suite
=======================

Comprehensive tests for the Universal RAG system using only universal components.
Replaces all old domain-specific test files with clean universal tests.

Tests:
- Universal knowledge extraction
- Universal query processing
- Universal API endpoints
- Multi-domain support
- Performance validation
"""

import sys
import os
import pytest
import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# Add backend directory to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Universal RAG imports (only new universal components)
from core.orchestration.enhanced_rag_universal import EnhancedUniversalRAG
from core.orchestration.universal_rag_orchestrator_complete import (
    UniversalRAGOrchestrator, create_universal_rag_from_texts
)
from core.extraction.universal_knowledge_extractor import UniversalKnowledgeExtractor
from core.knowledge.universal_text_processor import UniversalTextProcessor
from core.models.universal_models import (
    UniversalEntity, UniversalRelation, UniversalDocument
)


class TestUniversalRAG:
    """Test suite for Universal RAG system"""

    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing"""
        return [
            "The system requires regular maintenance to ensure optimal performance.",
            "Equipment monitoring helps detect potential failures before they occur.",
            "Safety procedures must be followed when performing maintenance tasks.",
            "Preventive maintenance schedules reduce unexpected downtime significantly.",
            "Proper lubrication is essential for bearing longevity and performance."
        ]

    @pytest.fixture
    def sample_domains(self):
        """Sample domains with texts for multi-domain testing"""
        return {
            "medical": [
                "Patient symptoms include fever and fatigue requiring immediate attention.",
                "Diagnosis involves comprehensive examination and laboratory test results."
            ],
            "legal": [
                "Contract terms must be clearly defined and legally enforceable.",
                "Liability clauses protect parties from unforeseen legal circumstances."
            ],
            "finance": [
                "Investment risk assessment requires careful portfolio analysis.",
                "Market volatility significantly affects asset valuation and returns."
            ]
        }

    @pytest.mark.asyncio
    async def test_universal_knowledge_extractor(self, sample_texts):
        """Test universal knowledge extraction from texts"""
        extractor = UniversalKnowledgeExtractor("test_domain")

        # Test knowledge extraction
        results = await extractor.extract_knowledge_from_texts(sample_texts)

        assert results["success"] is True
        assert "knowledge_summary" in results
        assert results["knowledge_summary"]["total_entities"] > 0
        assert results["knowledge_summary"]["total_relations"] >= 0
        assert "discovered_types" in results
        assert len(results["discovered_types"]["entity_types"]) > 0

    @pytest.mark.asyncio
    async def test_universal_text_processor(self, sample_texts):
        """Test universal text processor"""
        processor = UniversalTextProcessor("test_domain")

        # Test document creation
        documents = await processor.process_texts_to_documents(sample_texts)

        assert len(documents) == len(sample_texts)
        for doc_id, doc in documents.items():
            assert isinstance(doc, UniversalDocument)
            assert doc.text is not None
            assert len(doc.text.strip()) > 0

    @pytest.mark.asyncio
    async def test_universal_rag_orchestrator(self, sample_texts):
        """Test universal RAG orchestrator"""
        orchestrator = await create_universal_rag_from_texts(sample_texts, "test_domain")

        # Test system status
        status = orchestrator.get_system_status()
        assert status["initialized"] is True
        assert status["domain"] == "test_domain"
        assert status["system_stats"]["total_documents"] > 0

        # Test query processing
        query = "How to maintain equipment properly?"
        results = await orchestrator.process_query(query)

        assert results["success"] is True
        assert "response" in results
        assert results["domain"] == "test_domain"
        assert results["processing_time"] > 0

    @pytest.mark.asyncio
    async def test_enhanced_universal_rag(self, sample_texts):
        """Test enhanced universal RAG pipeline"""
        enhanced_rag = EnhancedUniversalRAG("test_domain")

        # Create temporary text files for initialization
        temp_dir = Path("temp_test_data")
        temp_dir.mkdir(exist_ok=True)

        text_files = []
        for i, text in enumerate(sample_texts):
            file_path = temp_dir / f"test_{i}.txt"
            with open(file_path, 'w') as f:
                f.write(text)
            text_files.append(file_path)

        try:
            # Test initialization
            init_results = await enhanced_rag.initialize_components(text_files)
            assert init_results["success"] is True
            assert init_results["components_initialized"] is True
            assert init_results["knowledge_loaded"] is True

            # Test query processing
            query = "What are maintenance best practices?"
            results = await enhanced_rag.process_query(query)

            assert results["success"] is True
            assert "generated_response" in results
            assert results["domain"] == "test_domain"

        finally:
            # Clean up temporary files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_multi_domain_support(self, sample_domains):
        """Test multi-domain support"""
        orchestrators = {}

        # Create orchestrators for each domain
        for domain, texts in sample_domains.items():
            orchestrator = await create_universal_rag_from_texts(texts, domain)
            orchestrators[domain] = orchestrator

            # Verify domain-specific setup
            status = orchestrator.get_system_status()
            assert status["domain"] == domain
            assert status["initialized"] is True

        # Test domain-specific queries
        queries = {
            "medical": "What symptoms require attention?",
            "legal": "What are contract requirements?",
            "finance": "How to assess investment risk?"
        }

        for domain, query in queries.items():
            orchestrator = orchestrators[domain]
            results = await orchestrator.process_query(query)

            assert results["success"] is True
            assert results["domain"] == domain
            assert "response" in results

    @pytest.mark.asyncio
    async def test_dynamic_type_discovery(self, sample_texts):
        """Test dynamic entity and relation type discovery"""
        extractor = UniversalKnowledgeExtractor("test_domain")

        results = await extractor.extract_knowledge_from_texts(sample_texts)

        assert results["success"] is True

        discovered_types = results["discovered_types"]
        entity_types = discovered_types["entity_types"]
        relation_types = discovered_types["relation_types"]

        # Verify dynamic types are discovered (not hardcoded)
        assert len(entity_types) > 0
        assert len(relation_types) >= 0

        # Verify types are strings (not enum values)
        for entity_type in entity_types:
            assert isinstance(entity_type, str)

        for relation_type in relation_types:
            assert isinstance(relation_type, str)

    @pytest.mark.asyncio
    async def test_zero_configuration_setup(self, sample_texts):
        """Test that system works with zero configuration"""
        # This should work without any config files or schemas
        orchestrator = await create_universal_rag_from_texts(sample_texts, "zero_config_test")

        # Test that it works immediately
        query = "Tell me about this content"
        results = await orchestrator.process_query(query)

        assert results["success"] is True

        # Verify no hardcoded types were required
        status = orchestrator.get_system_status()
        assert status["initialized"] is True

    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, sample_texts):
        """Test system performance benchmarks"""
        # Test setup time
        setup_start = time.time()
        orchestrator = await create_universal_rag_from_texts(sample_texts, "performance_test")
        setup_time = time.time() - setup_start

        # Setup should be reasonably fast (< 30 seconds)
        assert setup_time < 30.0

        # Test query time
        query = "What is the main topic?"
        query_start = time.time()
        results = await orchestrator.process_query(query)
        query_time = time.time() - query_start

        assert results["success"] is True
        # Query should be reasonably fast (< 10 seconds)
        assert query_time < 10.0

        # Log performance for monitoring
        print(f"Performance: Setup {setup_time:.2f}s, Query {query_time:.2f}s")

    def test_universal_models(self):
        """Test universal data models"""
        # Test UniversalEntity
        entity = UniversalEntity(
            entity_id="test_entity",
            text="test concept",
            entity_type="test_type",  # Dynamic string, not enum
            confidence=0.9,
            metadata={"test": "data"}
        )

        assert entity.entity_id == "test_entity"
        assert entity.entity_type == "test_type"
        assert entity.confidence == 0.9

        # Test serialization
        entity_dict = entity.to_dict()
        assert entity_dict["entity_type"] == "test_type"

        # Test deserialization
        new_entity = UniversalEntity.from_dict(entity_dict)
        assert new_entity.entity_type == entity.entity_type

        # Test UniversalRelation
        relation = UniversalRelation(
            relation_id="test_relation",
            source_entity_id="entity1",
            target_entity_id="entity2",
            relation_type="test_relation_type",  # Dynamic string, not enum
            confidence=0.8
        )

        assert relation.relation_type == "test_relation_type"

        # Test UniversalDocument
        document = UniversalDocument(
            document_id="test_doc",
            text="This is a test document",
            title="Test Document",
            metadata={"source": "test"}
        )

        document.add_entity(entity)
        document.add_relation(relation)

        assert len(document.entities) == 1
        assert len(document.relations) == 1

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling and edge cases"""
        # Test with empty texts
        empty_orchestrator = await create_universal_rag_from_texts([], "empty_test")
        status = empty_orchestrator.get_system_status()
        # Should handle gracefully (might have 0 entities but still work)

        # Test with very short texts
        short_texts = ["a", "b", "c"]
        short_orchestrator = await create_universal_rag_from_texts(short_texts, "short_test")

        query = "test query"
        results = await short_orchestrator.process_query(query)
        # Should either succeed or fail gracefully
        assert "success" in results

    def test_legacy_compatibility(self):
        """Test that legacy aliases still work"""
        from core.models.maintenance_models import (
            MaintenanceEntity, MaintenanceRelation, MaintenanceDocument
        )

        # These should be aliases to Universal classes
        assert MaintenanceEntity == UniversalEntity
        assert MaintenanceRelation == UniversalRelation
        assert MaintenanceDocument == UniversalDocument

        # Test legacy helper functions
        from core.models.maintenance_models import (
            create_entity_type, create_relation_type
        )

        entity_type = create_entity_type("Test Entity")
        assert entity_type == "test_entity"

        relation_type = create_relation_type("Test Relation")
        assert relation_type == "test_relation"


class TestSystemIntegration:
    """Integration tests for the complete Universal RAG system"""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # Sample workflow: Text → Knowledge → Query → Response
        texts = [
            "Modern systems require automated monitoring and maintenance scheduling.",
            "Predictive analytics help identify potential equipment failures early.",
            "Regular inspections ensure safety and operational reliability."
        ]

        # Step 1: Create system
        orchestrator = await create_universal_rag_from_texts(texts, "e2e_test")

        # Step 2: Verify setup
        status = orchestrator.get_system_status()
        assert status["initialized"] is True
        assert status["system_stats"]["total_documents"] > 0

        # Step 3: Process query
        query = "How can I improve system reliability?"
        results = await orchestrator.process_query(query)

        # Step 4: Verify response
        assert results["success"] is True
        assert "response" in results
        assert results["processing_time"] > 0

        # Step 5: Test multiple queries
        queries = [
            "What monitoring techniques are available?",
            "How to schedule maintenance?",
            "What are safety considerations?"
        ]

        for test_query in queries:
            test_results = await orchestrator.process_query(test_query)
            assert test_results["success"] is True


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])