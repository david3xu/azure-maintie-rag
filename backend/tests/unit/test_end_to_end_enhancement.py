"""
End-to-end integration tests for enhanced scheme integration
"""
import pytest
from pathlib import Path
from src.knowledge.data_transformer import MaintIEDataTransformer
from src.models.maintenance_models import EntityType, RelationType


class TestEndToEndEnhancement:

    def test_full_pipeline_with_hierarchy(self):
        """Test complete pipeline with enhanced scheme support"""
        transformer = MaintIEDataTransformer()

        # Run extraction
        stats = transformer.extract_maintenance_knowledge()

        # Verify enhanced features
        assert stats["total_entities"] > 0
        assert stats["total_relations"] > 0

        # Check that hierarchy types are being used
        entity_types = set()
        for entity in transformer.entities.values():
            entity_types.add(entity.entity_type.value)

        # Should have some hierarchy types (not just top-level)
        hierarchy_types = [t for t in entity_types if "/" in t]
        assert len(hierarchy_types) > 0, "No hierarchy types found"

    def test_metadata_preservation(self):
        """Test that metadata is preserved through pipeline"""
        transformer = MaintIEDataTransformer()
        stats = transformer.extract_maintenance_knowledge()

        # Check that entities have enhanced metadata
        if transformer.entities:
            sample_entity = next(iter(transformer.entities.values()))
            assert "color" in sample_entity.metadata
            assert "description" in sample_entity.metadata
            assert "example_terms" in sample_entity.metadata

    def test_enhanced_type_coverage(self):
        """Test that enhanced type mappings cover all scheme types"""
        transformer = MaintIEDataTransformer()

        # Check that we have comprehensive type coverage
        entity_types = transformer.type_mappings["entity"]
        relation_types = transformer.type_mappings["relation"]

        # Should have substantial coverage
        assert len(entity_types) > 50, f"Expected >50 entity types, got {len(entity_types)}"
        assert len(relation_types) > 5, f"Expected >5 relation types, got {len(relation_types)}"

        # Check for specific important types
        important_entity_types = [
            "PhysicalObject/Substance/Gas",
            "PhysicalObject/Substance/Liquid",
            "Activity/MaintenanceActivity/Inspect",
            "State/UndesirableState/FailedState"
        ]

        important_relation_types = [
            "hasPart",
            "hasProperty",
            "isA",
            "contains"
        ]

        for entity_type in important_entity_types:
            assert entity_type in entity_types, f"Missing important entity type: {entity_type}"

        for relation_type in important_relation_types:
            assert relation_type in relation_types, f"Missing important relation type: {relation_type}"

    def test_scheme_processor_integration(self):
        """Test that scheme processor is properly integrated"""
        transformer = MaintIEDataTransformer()

        # Check that scheme processor is initialized
        assert hasattr(transformer, 'scheme_processor')
        assert transformer.scheme_processor is not None

        # Check that metadata manager is initialized
        assert hasattr(transformer, 'metadata_manager')
        assert transformer.metadata_manager is not None

        # Check that we have processed the scheme data
        assert hasattr(transformer, 'scheme_data')
        assert transformer.scheme_data is not None

    def test_backwards_compatibility(self):
        """Test that the enhanced system maintains backwards compatibility"""
        transformer = MaintIEDataTransformer()

        # Test that basic entity types still work
        basic_entity = transformer._map_entity_type("PhysicalObject")
        assert basic_entity == EntityType.PHYSICAL_OBJECT

        # Test that basic relation types still work
        basic_relation = transformer._map_relation_type("hasPart")
        assert basic_relation == RelationType.HAS_PART

        # Test that unknown types fall back gracefully
        unknown_entity = transformer._map_entity_type("UnknownType")
        assert unknown_entity == EntityType.PHYSICAL_OBJECT

        unknown_relation = transformer._map_relation_type("UnknownRelation")
        assert unknown_relation == RelationType.HAS_PART

    def test_performance_impact(self):
        """Test that enhanced features don't significantly impact performance"""
        import time

        transformer = MaintIEDataTransformer()

        # Measure initialization time
        start_time = time.time()
        transformer.extract_maintenance_knowledge()
        end_time = time.time()

        processing_time = end_time - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        assert processing_time < 30.0, f"Processing took too long: {processing_time:.2f}s"

    def test_metadata_quality(self):
        """Test that metadata is of good quality"""
        transformer = MaintIEDataTransformer()

        # Check that metadata has meaningful values
        entity_metadata = transformer.metadata_manager.get_entity_metadata("PhysicalObject/Substance/Gas")
        if entity_metadata:
            assert entity_metadata.color != "", "Color should not be empty"
            assert isinstance(entity_metadata.active, bool), "Active should be boolean"
            assert isinstance(entity_metadata.example_terms, list), "Example terms should be list"

        relation_metadata = transformer.metadata_manager.get_relation_metadata("hasPart")
        if relation_metadata:
            assert relation_metadata.color != "", "Color should not be empty"
            assert isinstance(relation_metadata.active, bool), "Active should be boolean"
            assert isinstance(relation_metadata.example_terms, list), "Example terms should be list"