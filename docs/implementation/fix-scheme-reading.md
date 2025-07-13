# Code Implementation Guide: Enhancing MaintIE Scheme Integration

## **Phase 1: Hierarchy Traversal (Week 1)**
### **Simple Foundation - Zero Breaking Changes**

### **Step 1.1: Create Hierarchy Processor**
```python
# src/knowledge/schema_processor.py (NEW FILE)
"""
Schema processing utilities for MaintIE scheme.json
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SchemeNode:
    """Represents a node in the scheme hierarchy"""
    name: str
    fullname: str
    children: List['SchemeNode']
    metadata: Dict[str, Any]
    path: List[int]

class SchemeProcessor:
    """Process MaintIE scheme.json with hierarchy support"""

    def __init__(self, scheme_path: Path):
        self.scheme_path = scheme_path
        self.entity_hierarchy = {}
        self.relation_hierarchy = {}
        self.all_types = {"entity": set(), "relation": set()}

    def load_scheme(self) -> Dict[str, Any]:
        """Load and process complete scheme hierarchy"""
        if not self.scheme_path.exists():
            logger.warning(f"Scheme file not found: {self.scheme_path}")
            return {"entity": [], "relation": []}

        try:
            with open(self.scheme_path, 'r') as f:
                scheme = json.load(f)

            # Process hierarchies
            self.entity_hierarchy = self._build_hierarchy(scheme.get("entity", []))
            self.relation_hierarchy = self._build_hierarchy(scheme.get("relation", []))

            # Build flat type sets for quick lookup
            self._build_type_sets()

            logger.info(f"Loaded scheme: {len(self.all_types['entity'])} entity types, "
                       f"{len(self.all_types['relation'])} relation types")

            return scheme
        except Exception as e:
            logger.error(f"Error loading scheme: {e}")
            return {"entity": [], "relation": []}

    def _build_hierarchy(self, items: List[Dict]) -> Dict[str, SchemeNode]:
        """Build hierarchy tree from scheme items"""
        hierarchy = {}

        for item in items:
            node = self._create_scheme_node(item)
            hierarchy[node.fullname] = node

        return hierarchy

    def _create_scheme_node(self, item: Dict) -> SchemeNode:
        """Create scheme node with children"""
        children = []
        for child_item in item.get("children", []):
            children.append(self._create_scheme_node(child_item))

        return SchemeNode(
            name=item.get("name", ""),
            fullname=item.get("fullname", ""),
            children=children,
            metadata={
                "color": item.get("color", ""),
                "active": item.get("active", True),
                "description": item.get("description", ""),
                "example_terms": item.get("example_terms", []),
                "id": item.get("id", ""),
                "path": item.get("path", [])
            },
            path=item.get("path", [])
        )

    def _build_type_sets(self):
        """Build flat sets of all available types"""
        def collect_types(hierarchy, category):
            for fullname, node in hierarchy.items():
                self.all_types[category].add(fullname)
                self._collect_children_types(node, category)

        collect_types(self.entity_hierarchy, "entity")
        collect_types(self.relation_hierarchy, "relation")

    def _collect_children_types(self, node: SchemeNode, category: str):
        """Recursively collect all child type names"""
        for child in node.children:
            self.all_types[category].add(child.fullname)
            self._collect_children_types(child, category)

    def get_all_types(self, category: str) -> Set[str]:
        """Get all available types for category"""
        return self.all_types.get(category, set())

    def find_node(self, fullname: str, category: str) -> SchemeNode:
        """Find node by fullname"""
        hierarchy = self.entity_hierarchy if category == "entity" else self.relation_hierarchy
        return hierarchy.get(fullname)
```

### **Step 1.2: Enhance Data Transformer**
```python
# Update src/knowledge/data_transformer.py
# Add these imports at the top:
from src.knowledge.schema_processor import SchemeProcessor

class MaintIEDataTransformer:
    def __init__(self, gold_path: Optional[Path] = None, silver_path: Optional[Path] = None):
        # ... existing code ...

        # ENHANCEMENT: Replace simple scheme loading with hierarchy processor
        self.scheme_processor = SchemeProcessor(self.scheme_path)
        self.scheme_data = self.scheme_processor.load_scheme()
        self.type_mappings = self._build_enhanced_type_mappings()

    def _build_enhanced_type_mappings(self) -> Dict[str, Any]:
        """Build comprehensive type mappings using hierarchy"""
        mappings = {"entity": {}, "relation": {}}

        # Map all entity types (including children)
        for entity_type in self.scheme_processor.get_all_types("entity"):
            mappings["entity"][entity_type] = self._map_entity_type(entity_type)

        # Map all relation types (including children)
        for relation_type in self.scheme_processor.get_all_types("relation"):
            mappings["relation"][relation_type] = self._map_relation_type(relation_type)

        logger.info(f"Built enhanced mappings: {len(mappings['entity'])} entity types, "
                   f"{len(mappings['relation'])} relation types")

        return mappings
```

### **Step 1.3: Update Type Enums**
```python
# src/models/maintenance_models.py
# Add missing types based on your actual scheme.json

class EntityType(Enum):
    PHYSICAL_OBJECT = "PhysicalObject"
    ACTIVITY = "Activity"
    STATE = "State"
    PROBLEM = "Problem"
    PROCESS = "Process"
    PROPERTY = "Property"
    # Add hierarchy types from scheme.json
    SUBSTANCE = "PhysicalObject/Substance"
    GAS = "PhysicalObject/Substance/Gas"
    LIQUID = "PhysicalObject/Substance/Liquid"
    SOLID = "PhysicalObject/Substance/Solid"
    MIXTURE = "PhysicalObject/Substance/Mixture"
    ORGANISM = "PhysicalObject/Organism"
    PERSON = "PhysicalObject/Organism/Person"
    SENSING_OBJECT = "PhysicalObject/SensingObject"
    DESIRABLE_PROCESS = "Process/DesirableProcess"
    UNDESIRABLE_PROCESS = "Process/UndesirableProcess"
    DESIRABLE_PROPERTY = "Property/DesirableProperty"
    UNDESIRABLE_PROPERTY = "Property/UndesirableProperty"

class RelationType(Enum):
    HAS_PART = "hasPart"
    HAS_PROPERTY = "hasProperty"
    # Add types from scheme.json
    IS_A = "isA"
    CONTAINS = "contains"
    HAS_PARTICIPANT = "hasParticipant"
    HAS_PATIENT = "hasParticipant/hasPatient"
    HAS_AGENT = "hasParticipant/hasAgent"
```

### **Step 1.4: Enhanced Type Mapping Logic**
```python
# Update src/knowledge/data_transformer.py

def _map_entity_type(self, fullname: str) -> EntityType:
    """Enhanced entity type mapping with hierarchy support"""
    try:
        # Direct enum lookup first
        return EntityType(fullname)
    except ValueError:
        # Fallback to pattern matching for backwards compatibility
        name_lower = fullname.lower()

        if "physicalobject" in name_lower:
            if "substance" in name_lower:
                if "gas" in name_lower:
                    return EntityType.GAS
                elif "liquid" in name_lower:
                    return EntityType.LIQUID
                elif "solid" in name_lower:
                    return EntityType.SOLID
                elif "mixture" in name_lower:
                    return EntityType.MIXTURE
                else:
                    return EntityType.SUBSTANCE
            elif "organism" in name_lower:
                if "person" in name_lower:
                    return EntityType.PERSON
                else:
                    return EntityType.ORGANISM
            elif "sensing" in name_lower:
                return EntityType.SENSING_OBJECT
            else:
                return EntityType.PHYSICAL_OBJECT
        elif "process" in name_lower:
            if "undesirable" in name_lower:
                return EntityType.UNDESIRABLE_PROCESS
            elif "desirable" in name_lower:
                return EntityType.DESIRABLE_PROCESS
            else:
                return EntityType.PROCESS
        elif "property" in name_lower:
            if "undesirable" in name_lower:
                return EntityType.UNDESIRABLE_PROPERTY
            elif "desirable" in name_lower:
                return EntityType.DESIRABLE_PROPERTY
            else:
                return EntityType.PROPERTY
        elif "activity" in name_lower:
            return EntityType.ACTIVITY
        elif "state" in name_lower:
            return EntityType.STATE
        else:
            logger.warning(f"Unknown entity type: {fullname}, defaulting to PHYSICAL_OBJECT")
            return EntityType.PHYSICAL_OBJECT

def _map_relation_type(self, fullname: str) -> RelationType:
    """Enhanced relation type mapping with hierarchy support"""
    try:
        # Direct enum lookup first
        return RelationType(fullname)
    except ValueError:
        # Fallback to pattern matching
        name_lower = fullname.lower()

        if "haspart" in name_lower:
            return RelationType.HAS_PART
        elif "hasproperty" in name_lower:
            return RelationType.HAS_PROPERTY
        elif "isa" in name_lower:
            return RelationType.IS_A
        elif "contains" in name_lower:
            return RelationType.CONTAINS
        elif "hasparticipant" in name_lower:
            if "patient" in name_lower:
                return RelationType.HAS_PATIENT
            elif "agent" in name_lower:
                return RelationType.HAS_AGENT
            else:
                return RelationType.HAS_PARTICIPANT
        else:
            logger.warning(f"Unknown relation type: {fullname}, defaulting to HAS_PART")
            return RelationType.HAS_PART
```

## **Phase 2: Metadata Integration (Week 2)**
### **Professional Metadata Handling**

### **Step 2.1: Metadata Storage**
```python
# src/knowledge/metadata_manager.py (NEW FILE)
"""
Metadata management for enhanced MaintIE features
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass
from src.knowledge.schema_processor import SchemeProcessor

@dataclass
class TypeMetadata:
    """Metadata for entity/relation types"""
    color: str
    active: bool
    description: str
    example_terms: list
    path: list
    confidence: float = 1.0

class MetadataManager:
    """Manage type metadata for enhanced features"""

    def __init__(self, scheme_processor: SchemeProcessor):
        self.scheme_processor = scheme_processor
        self.entity_metadata = {}
        self.relation_metadata = {}
        self._build_metadata_cache()

    def _build_metadata_cache(self):
        """Build metadata cache from scheme processor"""
        # Cache entity metadata
        for fullname, node in self.scheme_processor.entity_hierarchy.items():
            self.entity_metadata[fullname] = TypeMetadata(
                color=node.metadata.get("color", "#cccccc"),
                active=node.metadata.get("active", True),
                description=node.metadata.get("description", ""),
                example_terms=node.metadata.get("example_terms", []),
                path=node.metadata.get("path", [])
            )

        # Cache relation metadata
        for fullname, node in self.scheme_processor.relation_hierarchy.items():
            self.relation_metadata[fullname] = TypeMetadata(
                color=node.metadata.get("color", "#cccccc"),
                active=node.metadata.get("active", True),
                description=node.metadata.get("description", ""),
                example_terms=node.metadata.get("example_terms", []),
                path=node.metadata.get("path", [])
            )

    def get_entity_metadata(self, entity_type: str) -> Optional[TypeMetadata]:
        """Get metadata for entity type"""
        return self.entity_metadata.get(entity_type)

    def get_relation_metadata(self, relation_type: str) -> Optional[TypeMetadata]:
        """Get metadata for relation type"""
        return self.relation_metadata.get(relation_type)

    def get_active_types(self, category: str) -> list:
        """Get only active types for category"""
        metadata_dict = self.entity_metadata if category == "entity" else self.relation_metadata
        return [type_name for type_name, metadata in metadata_dict.items() if metadata.active]
```

### **Step 2.2: Enhanced Entity Creation**
```python
# Update src/knowledge/data_transformer.py

class MaintIEDataTransformer:
    def __init__(self, gold_path: Optional[Path] = None, silver_path: Optional[Path] = None):
        # ... existing code ...

        # Add metadata manager
        from src.knowledge.metadata_manager import MetadataManager
        self.metadata_manager = MetadataManager(self.scheme_processor)

    def _create_entity(self, entity_data: Dict[str, Any], doc_id: str, doc_text: str,
                      doc_tokens: List[str], confidence_base: float) -> Optional[MaintenanceEntity]:
        """Enhanced entity creation with metadata"""
        try:
            # ... existing entity creation logic ...

            # Get metadata for the entity type
            entity_type_str = entity_data.get("type", "PhysicalObject")
            metadata = self.metadata_manager.get_entity_metadata(entity_type_str)

            # Check if type is active
            if metadata and not metadata.active:
                logger.debug(f"Skipping inactive entity type: {entity_type_str}")
                return None

            # ... rest of existing logic ...

            return MaintenanceEntity(
                entity_id=entity_id,
                text=text,
                entity_type=entity_type,
                confidence=min(confidence_base * entity_data.get("confidence", 1.0), 1.0),
                context=entity_data.get("context"),
                metadata={
                    "start": start,
                    "end": end,
                    "original_type": entity_type_str,
                    "doc_id": doc_id,
                    "color": metadata.color if metadata else "#cccccc",
                    "description": metadata.description if metadata else "",
                    "example_terms": metadata.example_terms if metadata else []
                }
            )
        except Exception as e:
            logger.warning(f"Error creating entity: {e}")
            return None
```

## **Phase 3: Testing & Validation (Week 3)**
### **Professional Testing Framework**

### **Step 3.1: Unit Tests**
```python
# tests/test_enhanced_scheme_integration.py
import pytest
from pathlib import Path
from src.knowledge.schema_processor import SchemeProcessor
from src.knowledge.data_transformer import MaintIEDataTransformer
from src.models.maintenance_models import EntityType, RelationType

class TestEnhancedSchemeIntegration:

    @pytest.fixture
    def scheme_processor(self):
        scheme_path = Path("data/raw/scheme.json")
        return SchemeProcessor(scheme_path)

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
        # Create sample entity data
        entity_data = {
            "id": "test_entity",
            "type": "PhysicalObject/Substance/Gas",
            "start": 0,
            "end": 3,
            "confidence": 0.9
        }

        entity = transformer._create_entity(
            entity_data, "test_doc", "gas leak", ["gas", "leak"], 1.0
        )

        assert entity is not None
        assert entity.entity_type == EntityType.GAS
        assert "color" in entity.metadata
        assert "description" in entity.metadata
```

### **Step 3.2: Integration Tests**
```python
# tests/test_end_to_end_enhancement.py
import pytest
from src.knowledge.data_transformer import MaintIEDataTransformer

class TestEndToEndEnhancement:

    def test_full_pipeline_with_hierarchy(self):
        """Test complete pipeline with enhanced scheme support"""
        transformer = MaintIEDataTransformer()

        # Run extraction
        stats = transformer.extract_maintenance_knowledge()

        # Verify enhanced features
        assert stats["entities_extracted"] > 0
        assert stats["relations_extracted"] > 0

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
        sample_entity = next(iter(transformer.entities.values()))
        assert "color" in sample_entity.metadata
        assert "description" in sample_entity.metadata
```

## **Deployment Lifecycle**

### **Week 1: Foundation**
```bash
# 1. Create new files
touch src/knowledge/schema_processor.py
touch src/knowledge/metadata_manager.py

# 2. Run tests
pytest tests/test_enhanced_scheme_integration.py -v

# 3. Validate no breaking changes
python -c "from src.knowledge.data_transformer import MaintIEDataTransformer; t=MaintIEDataTransformer(); print('✅ No breaking changes')"
```

### **Week 2: Integration**
```bash
# 1. Run comprehensive tests
pytest tests/ -v --cov=src/knowledge/

# 2. Performance benchmark
python benchmarks/compare_extraction_performance.py

# 3. Data quality validation
python scripts/validate_enhanced_extraction.py
```

### **Week 3: Production Ready**
```bash
# 1. Generate documentation
python scripts/generate_type_documentation.py

# 2. Create deployment package
python setup.py sdist bdist_wheel

# 3. Deploy to staging
az webapp deploy --resource-group maintie-rg --name maintie-rag-staging
```

This implementation follows your preferences: starts simple, maintains professional architecture, and provides a clear lifecycle workflow based on your actual codebase.