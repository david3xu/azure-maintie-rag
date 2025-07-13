"""
Unit tests for enhanced scheme integration
"""
import pytest
from pathlib import Path
from src.knowledge.schema_processor import SchemeProcessor
from src.knowledge.data_transformer import MaintIEDataTransformer
from src.models.maintenance_models import EntityType, RelationType


class TestEnhancedSchemeIntegration:

    @pytest.fixture
    def scheme_processor(self):
        scheme_path = Path("data/raw/scheme.json")
        processor = SchemeProcessor(scheme_path)
        processor.load_scheme()  # Load the scheme data
        return processor

    @pytest.fixture
    def transformer(self):
        return MaintIEDataTransformer()

    def test_hierarchy_loading(self, scheme_processor):
        """Test that hierarchy is loaded correctly"""
        scheme_data = scheme_processor.load_scheme()

        assert "entity" in scheme_data
        assert "relation" in scheme_data
        assert len(scheme_processor.get_all_types("entity")) > 0
        assert len(scheme_processor.get_all_types("relation")) > 0

    def test_subtype_mapping(self, transformer):
        """Test that subtypes are mapped correctly"""
        # Test entity subtype
        gas_type = transformer._map_entity_type("PhysicalObject/Substance/Gas")
        assert gas_type == EntityType.GAS

        # Test relation subtype
        patient_type = transformer._map_relation_type("hasParticipant/hasPatient")
        assert patient_type == RelationType.HAS_PATIENT

    def test_metadata_integration(self, transformer):
        """Test that metadata is properly integrated"""
        # Create sample entity data with proper text
        entity_data = {
            "id": "test_entity",
            "type": "PhysicalObject/Substance/Gas",
            "text": "gas",  # Provide text directly
            "start": 0,
            "end": 1,
            "confidence": 0.9
        }

        entity = transformer._create_entity(
            entity_data, "test_doc", "gas leak", ["gas", "leak"], 1.0
        )

        assert entity is not None
        assert entity.entity_type == EntityType.GAS
        assert "color" in entity.metadata
        assert "description" in entity.metadata
        assert "example_terms" in entity.metadata

    def test_enhanced_type_mappings(self, transformer):
        """Test that enhanced type mappings include all hierarchy types"""
        # Check that we have more types than just top-level
        entity_types = transformer.type_mappings["entity"]
        relation_types = transformer.type_mappings["relation"]

        # Should have hierarchy types (containing "/")
        hierarchy_entity_types = [t for t in entity_types.keys() if "/" in t]
        hierarchy_relation_types = [t for t in relation_types.keys() if "/" in t]

        assert len(hierarchy_entity_types) > 0, "No hierarchy entity types found"
        assert len(hierarchy_relation_types) > 0, "No hierarchy relation types found"

    def test_metadata_manager(self, transformer):
        """Test metadata manager functionality"""
        # Test getting entity metadata
        gas_metadata = transformer.metadata_manager.get_entity_metadata("PhysicalObject/Substance/Gas")
        assert gas_metadata is not None
        assert hasattr(gas_metadata, 'color')
        assert hasattr(gas_metadata, 'description')
        assert hasattr(gas_metadata, 'active')

        # Test getting relation metadata
        has_part_metadata = transformer.metadata_manager.get_relation_metadata("hasPart")
        assert has_part_metadata is not None
        assert hasattr(has_part_metadata, 'color')
        assert hasattr(has_part_metadata, 'description')

    def test_active_type_filtering(self, transformer):
        """Test that inactive types are filtered out"""
        active_entity_types = transformer.metadata_manager.get_active_types("entity")
        active_relation_types = transformer.metadata_manager.get_active_types("relation")

        assert len(active_entity_types) > 0
        assert len(active_relation_types) > 0

        # All active types should have active=True in their metadata
        for entity_type in active_entity_types:
            metadata = transformer.metadata_manager.get_entity_metadata(entity_type)
            assert metadata.active is True

        for relation_type in active_relation_types:
            metadata = transformer.metadata_manager.get_relation_metadata(relation_type)
            assert metadata.active is True

    def test_scheme_processor_node_finding(self, scheme_processor):
        """Test finding nodes in the hierarchy"""
        # Test finding an entity node
        gas_node = scheme_processor.find_node("PhysicalObject/Substance/Gas", "entity")
        assert gas_node is not None
        assert gas_node.fullname == "PhysicalObject/Substance/Gas"

        # Test finding a relation node
        has_part_node = scheme_processor.find_node("hasPart", "relation")
        assert has_part_node is not None
        assert has_part_node.fullname == "hasPart"

    def test_comprehensive_entity_mapping(self, transformer):
        """Test comprehensive entity type mapping"""
        # Test various entity types from the hierarchy
        test_cases = [
            ("PhysicalObject/Substance/Gas", EntityType.GAS),
            ("PhysicalObject/Substance/Liquid", EntityType.LIQUID),
            ("PhysicalObject/Organism/Person", EntityType.PERSON),
            ("Activity/MaintenanceActivity/Inspect", EntityType.INSPECT),
            ("State/UndesirableState/FailedState", EntityType.FAILED_STATE),
            ("Process/DesirableProcess", EntityType.DESIRABLE_PROCESS),
        ]

        for input_type, expected_enum in test_cases:
            result = transformer._map_entity_type(input_type)
            assert result == expected_enum, f"Failed for {input_type}: expected {expected_enum}, got {result}"

    def test_comprehensive_relation_mapping(self, transformer):
        """Test comprehensive relation type mapping"""
        # Test various relation types from the hierarchy
        test_cases = [
            ("hasPart", RelationType.HAS_PART),
            ("hasProperty", RelationType.HAS_PROPERTY),
            ("isA", RelationType.IS_A),
            ("contains", RelationType.CONTAINS),
            ("hasParticipant/hasPatient", RelationType.HAS_PATIENT),
            ("hasParticipant/hasAgent", RelationType.HAS_AGENT),
        ]

        for input_type, expected_enum in test_cases:
            result = transformer._map_relation_type(input_type)
            assert result == expected_enum, f"Failed for {input_type}: expected {expected_enum}, got {result}"