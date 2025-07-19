"""Test universal RAG models using only MD files from data/raw directory."""

import pytest
import os
from typing import List, Dict, Any
from pathlib import Path

from backend.core.models.universal_rag_models import (
    UniversalEntity, UniversalRelation, UniversalDocument,
    UniversalQueryAnalysis, UniversalEnhancedQuery,
    UniversalSearchResult, UniversalRAGResponse,
    UniversalKnowledgeGraph, UniversalTrainingConfig, UniversalTrainingResult
)


def load_raw_data() -> List[str]:
    """Load raw data from data/raw directory for testing - MD format only"""
    raw_data_path = Path("data/raw")
    if not raw_data_path.exists():
        return []

    documents = []
    for file_path in raw_data_path.rglob("*.md"):
        if file_path.is_file():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(content)
            except Exception:
                continue

    return documents


def create_test_entities_from_raw_data(raw_text: str) -> List[UniversalEntity]:
    """Create test entities from raw MD data without domain assumptions"""
    entities = []

    # Simple word-based entity extraction from MD content (no domain knowledge)
    words = raw_text.split()
    for i, word in enumerate(words[:10]):  # Limit to first 10 words
        if len(word) > 3:  # Only words longer than 3 characters
            entity = UniversalEntity(
                entity_id=f"test_entity_{i}",
                text=word,
                entity_type="unknown",  # No domain assumption
                confidence=0.8
            )
            entities.append(entity)

    return entities


def create_test_relations_from_entities(entities: List[UniversalEntity]) -> List[UniversalRelation]:
    """Create test relations from entities without domain assumptions"""
    relations = []

    if len(entities) < 2:
        return relations

    # Create simple sequential relations (no domain knowledge)
    for i in range(len(entities) - 1):
        relation = UniversalRelation(
            relation_id=f"test_relation_{i}",
            source_entity_id=entities[i].entity_id,
            target_entity_id=entities[i + 1].entity_id,
            relation_type="follows",  # Generic relation type
            confidence=0.7
        )
        relations.append(relation)

    return relations


class TestUniversalModels:
    """Test universal RAG models with MD data only"""

    def test_universal_entity_creation(self):
        """Test universal entity creation with MD data"""
        raw_documents = load_raw_data()

        if not raw_documents:
            pytest.skip("No MD files available in data/raw for testing")

        # Use first MD document for testing
        test_text = raw_documents[0][:100]  # First 100 characters
        entities = create_test_entities_from_raw_data(test_text)

        assert len(entities) > 0
        for entity in entities:
            assert isinstance(entity, UniversalEntity)
            assert entity.entity_id.startswith("test_entity_")
            assert len(entity.text) > 0
            assert entity.entity_type == "unknown"  # No domain assumption
            assert 0 <= entity.confidence <= 1

    def test_universal_relation_creation(self):
        """Test universal relation creation with MD data"""
        raw_documents = load_raw_data()

        if not raw_documents:
            pytest.skip("No MD files available in data/raw for testing")

        # Use first MD document for testing
        test_text = raw_documents[0][:100]  # First 100 characters
        entities = create_test_entities_from_raw_data(test_text)
        relations = create_test_relations_from_entities(entities)

        if len(entities) >= 2:
            assert len(relations) > 0
            for relation in relations:
                assert isinstance(relation, UniversalRelation)
                assert relation.relation_id.startswith("test_relation_")
                assert relation.relation_type == "follows"  # Generic type
                assert 0 <= relation.confidence <= 1

    def test_universal_document_creation(self):
        """Test universal document creation with MD data"""
        raw_documents = load_raw_data()

        if not raw_documents:
            pytest.skip("No MD files available in data/raw for testing")

        # Use first MD document for testing
        test_text = raw_documents[0][:200]  # First 200 characters
        entities = create_test_entities_from_raw_data(test_text)
        relations = create_test_relations_from_entities(entities)

        document = UniversalDocument(
            doc_id="test_doc_1",
            text=test_text,
            title="Test Document from MD Data"
        )

        # Add entities and relations
        for entity in entities:
            document.add_entity(entity)

        for relation in relations:
            document.add_relation(relation)

        assert document.doc_id == "test_doc_1"
        assert len(document.text) > 0
        assert len(document.entities) > 0
        assert len(document.relations) >= 0

    def test_universal_query_analysis(self):
        """Test universal query analysis with MD data"""
        raw_documents = load_raw_data()

        if not raw_documents:
            pytest.skip("No MD files available in data/raw for testing")

        # Create test query from MD data
        test_text = raw_documents[0][:50]  # First 50 characters
        query = f"What is {test_text[:20]}?"  # Create a question

        analysis = UniversalQueryAnalysis(
            query_text=query,
            query_type="factual",  # Generic query type
            confidence=0.8,
            entities_detected=[],  # No domain-specific entities
            concepts_detected=[],  # No domain-specific concepts
            intent="information_seeking"
        )

        assert analysis.query_text == query
        assert analysis.query_type.value == "factual"
        assert 0 <= analysis.confidence <= 1
        assert analysis.intent == "information_seeking"

    def test_universal_training_config(self):
        """Test universal training configuration with MD data"""
        config = UniversalTrainingConfig(
            model_type="gnn",
            domain="test_domain",
            training_data_path="data/raw/",  # MD files only
            model_config={
                "code_path": "./training/",
                "command": "python train.py"
            },
            hyperparameters={
                "learning_rate": 0.001,
                "batch_size": 32
            }
        )

        assert config.model_type == "gnn"
        assert config.domain == "test_domain"
        assert config.training_data_path == "data/raw/"
        assert "learning_rate" in config.hyperparameters
        assert "batch_size" in config.hyperparameters

    def test_universal_knowledge_graph(self):
        """Test universal knowledge graph with MD data"""
        raw_documents = load_raw_data()

        if not raw_documents:
            pytest.skip("No MD files available in data/raw for testing")

        # Create test entities and relations from MD data
        test_text = raw_documents[0][:100]
        entities = create_test_entities_from_raw_data(test_text)
        relations = create_test_relations_from_entities(entities)

        kg = UniversalKnowledgeGraph(domain="test_domain")

        # Add entities and relations
        for entity in entities:
            kg.add_entity(entity)

        for relation in relations:
            kg.add_relation(relation)

        assert kg.domain == "test_domain"
        assert len(kg.entities) > 0
        assert len(kg.relations) >= 0
        assert len(kg.entity_types) > 0

    def test_model_serialization(self):
        """Test model serialization/deserialization"""
        # Create test entity
        entity = UniversalEntity(
            entity_id="test_serialization",
            text="test_text",
            entity_type="test_type",
            confidence=0.9
        )

        # Test to_dict
        entity_dict = entity.to_dict()
        assert entity_dict["entity_id"] == "test_serialization"
        assert entity_dict["text"] == "test_text"
        assert entity_dict["entity_type"] == "test_type"
        assert entity_dict["confidence"] == 0.9

        # Test from_dict
        reconstructed_entity = UniversalEntity.from_dict(entity_dict)
        assert reconstructed_entity.entity_id == entity.entity_id
        assert reconstructed_entity.text == entity.text
        assert reconstructed_entity.entity_type == entity.entity_type
        assert reconstructed_entity.confidence == entity.confidence

    def test_no_domain_knowledge_assumptions(self):
        """Test that no domain knowledge is assumed"""
        # Test that entity types are dynamic
        entity_types = ["component", "person", "resource", "document", "process"]

        for entity_type in entity_types:
            entity = UniversalEntity(
                entity_id=f"test_{entity_type}",
                text=f"test_{entity_type}",
                entity_type=entity_type,
                confidence=0.8
            )
            assert entity.entity_type == entity_type.lower().replace(' ', '_')

        # Test that relation types are dynamic
        relation_types = ["causes", "belongs_to", "contains", "relates_to"]

        for relation_type in relation_types:
            relation = UniversalRelation(
                relation_id=f"test_{relation_type}",
                source_entity_id="source",
                target_entity_id="target",
                relation_type=relation_type,
                confidence=0.8
            )
            assert relation.relation_type == relation_type.lower().replace(' ', '_')