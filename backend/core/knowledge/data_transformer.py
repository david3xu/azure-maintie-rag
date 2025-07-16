"""
MaintIE data transformation module - Enhanced working version
Keeps your working logic + adds simple scheme.json type mapping
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter

from core.models.maintenance_models import (
    MaintenanceEntity, MaintenanceRelation, MaintenanceDocument,
    EntityType, RelationType
)
from core.knowledge.schema_processor import SchemeProcessor
from core.knowledge.metadata_manager import MetadataManager
from config.settings import settings


logger = logging.getLogger(__name__)


class MaintIEDataTransformer:
    """Transform MaintIE annotations into structured knowledge for RAG"""

    def __init__(self, gold_path: Optional[Path] = None, silver_path: Optional[Path] = None):
        """Initialize transformer with data paths"""

        # Use configurable filenames
        gold_filename = settings.gold_data_filename
        silver_filename = settings.silver_data_filename

        self.gold_path = gold_path or settings.raw_data_dir / gold_filename
        self.silver_path = silver_path or settings.raw_data_dir / silver_filename
        self.processed_dir = settings.processed_data_dir

        # ENHANCEMENT: Replace simple scheme loading with hierarchy processor
        self.scheme_path = settings.raw_data_dir / "scheme.json"
        self.scheme_processor = SchemeProcessor(self.scheme_path)
        self.scheme_data = self.scheme_processor.load_scheme()
        self.type_mappings = self._build_enhanced_type_mappings()

        # Add metadata manager
        self.metadata_manager = MetadataManager(self.scheme_processor)

        # Ensure directories exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Data containers
        self.entities: Dict[str, MaintenanceEntity] = {}
        self.relations: List[MaintenanceRelation] = []
        self.documents: Dict[str, MaintenanceDocument] = {}
        self.knowledge_graph: Optional[nx.Graph] = None

        logger.info(f"Initialized MaintIE transformer for {self.gold_path} and {self.silver_path}")

        # Ensure knowledge graph is built
        if self.knowledge_graph is None:
            logger.info("Knowledge graph not found, extracting/processing knowledge...")
            try:
                self.extract_maintenance_knowledge()
                logger.info("Knowledge graph built during initialization.")
            except Exception as e:
                logger.error(f"Failed to build knowledge graph during initialization: {e}")

    def check_knowledge_graph_status(self):
        """Diagnostic method to check knowledge graph status"""
        print(f"Knowledge graph exists: {self.knowledge_graph is not None}")
        if self.knowledge_graph:
            print(f"Graph nodes: {self.knowledge_graph.number_of_nodes()}")
            print(f"Graph edges: {self.knowledge_graph.number_of_edges()}")
        else:
            print("Knowledge graph is None - need to run extract_maintenance_knowledge()")

    def load_existing_processed_data(self) -> bool:
        """Load existing processed data if available"""
        try:
            required_files = [
                self.processed_dir / "maintenance_entities.json",
                self.processed_dir / "maintenance_documents.json",
                self.processed_dir / "knowledge_graph.json"
            ]
            for file_path in required_files:
                if not file_path.exists():
                    return False

            # Load entities (saved as a list)
            with open(self.processed_dir / "maintenance_entities.json", 'r') as f:
                entities_data = json.load(f)
            from core.models.maintenance_models import MaintenanceEntity
            self.entities = {}
            for entity_data in entities_data:
                entity = MaintenanceEntity.from_dict(entity_data) if hasattr(MaintenanceEntity, 'from_dict') else MaintenanceEntity(**entity_data)
                self.entities[entity.entity_id] = entity

            # Load documents (saved as a list)
            with open(self.processed_dir / "maintenance_documents.json", 'r') as f:
                documents_data = json.load(f)
            from core.models.maintenance_models import MaintenanceDocument
            self.documents = {}
            for doc_data in documents_data:
                doc = MaintenanceDocument.from_dict(doc_data) if hasattr(MaintenanceDocument, 'from_dict') else MaintenanceDocument(**doc_data)
                self.documents[doc.doc_id] = doc

            # Load knowledge graph
            with open(self.processed_dir / "knowledge_graph.json", 'r') as f:
                graph_data = json.load(f)
            self.knowledge_graph = nx.node_link_graph(graph_data, edges="links")

            logger.info("Loaded existing processed data from cache.")
            return True
        except Exception as e:
            logger.warning(f"Failed to load existing processed data: {e}")
            return False

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
                elif "storing" in name_lower:
                    return EntityType.STORING_OBJECT
                elif "emitting" in name_lower:
                    return EntityType.EMITTING_OBJECT
                elif "protecting" in name_lower:
                    return EntityType.PROTECTING_OBJECT
                elif "generating" in name_lower:
                    return EntityType.GENERATING_OBJECT
                elif "matterprocessing" in name_lower:
                    return EntityType.MATTER_PROCESSING_OBJECT
                elif "informationprocessing" in name_lower:
                    return EntityType.INFORMATION_PROCESSING_OBJECT
                elif "driving" in name_lower:
                    return EntityType.DRIVING_OBJECT
                elif "covering" in name_lower:
                    return EntityType.COVERING_OBJECT
                elif "presenting" in name_lower:
                    return EntityType.PRESENTING_OBJECT
                elif "controlling" in name_lower:
                    return EntityType.CONTROLLING_OBJECT
                elif "restricting" in name_lower:
                    return EntityType.RESTRICTING_OBJECT
                elif "humaninteraction" in name_lower:
                    return EntityType.HUMAN_INTERACTION_OBJECT
                elif "transforming" in name_lower:
                    return EntityType.TRANSFORMING_OBJECT
                elif "holding" in name_lower:
                    return EntityType.HOLDING_OBJECT
                elif "guiding" in name_lower:
                    return EntityType.GUIDING_OBJECT
                elif "interfacing" in name_lower:
                    return EntityType.INTERFACING_OBJECT
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
                if "maintenance" in name_lower:
                    if "adjust" in name_lower:
                        return EntityType.ADJUST
                    elif "calibrate" in name_lower:
                        return EntityType.CALIBRATE
                    elif "diagnose" in name_lower:
                        return EntityType.DIAGNOSE
                    elif "inspect" in name_lower:
                        return EntityType.INSPECT
                    elif "replace" in name_lower:
                        return EntityType.REPLACE
                    elif "repair" in name_lower:
                        return EntityType.REPAIR
                    elif "service" in name_lower:
                        return EntityType.SERVICE
                    else:
                        return EntityType.MAINTENANCE_ACTIVITY
                elif "supporting" in name_lower:
                    if "admin" in name_lower:
                        return EntityType.ADMIN
                    elif "assemble" in name_lower:
                        return EntityType.ASSEMBLE
                    elif "isolate" in name_lower:
                        return EntityType.ISOLATE
                    elif "measure" in name_lower:
                        return EntityType.MEASURE
                    elif "modify" in name_lower:
                        return EntityType.MODIFY
                    elif "move" in name_lower:
                        return EntityType.MOVE
                    elif "operate" in name_lower:
                        return EntityType.OPERATE
                    elif "perform" in name_lower:
                        return EntityType.PERFORM
                    elif "teamwork" in name_lower:
                        return EntityType.TEAMWORK
                    else:
                        return EntityType.SUPPORTING_ACTIVITY
                else:
                    return EntityType.ACTIVITY
            elif "state" in name_lower:
                if "undesirable" in name_lower:
                    if "degraded" in name_lower:
                        return EntityType.DEGRADED_STATE
                    elif "failed" in name_lower:
                        return EntityType.FAILED_STATE
                    else:
                        return EntityType.UNDESIRABLE_STATE
                elif "desirable" in name_lower:
                    if "normal" in name_lower:
                        return EntityType.NORMAL_STATE
                    else:
                        return EntityType.DESIRABLE_STATE
                else:
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

    def load_raw_data(self) -> Dict[str, Any]:
        """Load raw MaintIE datasets"""
        logger.info("Loading raw MaintIE datasets...")

        data = {
            "gold": self._load_json_file(self.gold_path),
            "silver": self._load_json_file(self.silver_path)
        }

        logger.info(f"Loaded {len(data['gold'])} gold annotations and {len(data['silver'])} silver annotations")
        return data

    def _load_json_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSON file with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}. Using empty dataset.")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
            return []

    def extract_maintenance_knowledge(self) -> Dict[str, Any]:
        """Extract entities, relations, and documents from raw data"""
        logger.info("Extracting maintenance knowledge from annotations...")

        # Load raw data
        raw_data = self.load_raw_data()

        # Process gold data (high confidence)
        gold_stats = self._process_dataset(raw_data["gold"], confidence_base=settings.gold_confidence_base)

        # Process silver data (lower confidence)
        silver_stats = self._process_dataset(raw_data["silver"], confidence_base=settings.silver_confidence_base)

        # Build knowledge graph
        self._build_knowledge_graph()

        # Save processed data
        self._save_processed_data()

        stats = {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "total_documents": len(self.documents),
            "gold_stats": gold_stats,
            "silver_stats": silver_stats,
            "graph_nodes": self.knowledge_graph.number_of_nodes() if self.knowledge_graph else 0,
            "graph_edges": self.knowledge_graph.number_of_edges() if self.knowledge_graph else 0
        }

        logger.info(f"Knowledge extraction complete: {stats}")
        return stats

    def _process_dataset(self, dataset: List[Dict[str, Any]], confidence_base: float) -> Dict[str, int]:
        """Process individual dataset (gold or silver)"""
        stats = {"documents": 0, "entities": 0, "relations": 0}

        for doc_data in dataset:
            try:
                # Create document
                doc_id = doc_data.get("id", f"doc_{stats['documents']}")
                text = doc_data.get("text", "")  # Get the document text here
                tokens = doc_data.get("tokens", [])  # Get the tokens here

                document = MaintenanceDocument(
                    doc_id=doc_id,
                    text=text,
                    title=doc_data.get("title", f"Document {doc_id}"),
                    metadata={
                        "source": doc_data.get("source", "maintie"),
                        "confidence_base": confidence_base
                    }
                )
                self.documents[doc_id] = document  # Move this line here

                # Extract entities
                entities_data = doc_data.get("entities", [])
                entities_in_doc: List[MaintenanceEntity] = []  # New list to store entities created for this document
                for entity_data in entities_data:
                    entity = self._create_entity(entity_data, doc_id, text, tokens, confidence_base)  # Pass doc_id, doc_text, and tokens
                    if entity:
                        self.entities[entity.entity_id] = entity
                        document.add_entity(entity)
                        entities_in_doc.append(entity)  # Add to list for relation processing
                        stats["entities"] += 1

                # Extract relations
                relations_data = doc_data.get("relations", [])
                for relation_data in relations_data:
                    relation = self._create_relation(relation_data, entities_in_doc, confidence_base)
                    if relation:
                        self.relations.append(relation)
                        document.add_relation(relation)
                        stats["relations"] += 1

                stats["documents"] += 1

            except Exception as e:
                logger.warning(f"Error processing document {doc_data.get('id', 'unknown')}: {e}")
                continue

        return stats

    def _create_entity(self, entity_data: Dict[str, Any], doc_id: str, doc_text: str, doc_tokens: List[str], confidence_base: float) -> Optional[MaintenanceEntity]:
        """Create MaintenanceEntity from annotation data with enhanced metadata"""
        try:
            # Generate a unique entity_id based on doc_id, start, and end
            start = entity_data.get("start")
            end = entity_data.get("end")
            entity_id = f"{doc_id}_entity_{start}_{end}"

            text = entity_data.get("text", "").strip()

            if not text:  # If text is not directly provided in entity_data
                # Use the provided doc_tokens to extract the entity text
                if doc_tokens and start is not None and end is not None and len(doc_tokens) >= end:
                    text = " ".join(doc_tokens[start:end]).strip()
                if not text:
                    logger.warning(f"Skipping entity creation: no text or invalid start/end for {doc_id}")
                    return None

            # ENHANCEMENT: Use scheme mapping if available
            entity_type_str = entity_data.get("type", "PhysicalObject")

            # Get metadata for the entity type
            metadata = self.metadata_manager.get_entity_metadata(entity_type_str)

            # Check if type is active
            if metadata and not metadata.active:
                logger.debug(f"Skipping inactive entity type: {entity_type_str}")
                return None

            # Try scheme mapping first
            if entity_type_str in self.type_mappings["entity"]:
                entity_type = self.type_mappings["entity"][entity_type_str]
            else:
                # Fallback to your original logic
                try:
                    entity_type = EntityType(entity_type_str)
                except ValueError:
                    entity_type = EntityType.PHYSICAL_OBJECT

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
                    "doc_id": doc_id,  # Add doc_id to metadata for completeness
                    "color": metadata.color if metadata else "#cccccc",
                    "description": metadata.description if metadata else "",
                    "example_terms": metadata.example_terms if metadata else []
                }
            )
        except Exception as e:
            logger.warning(f"Error creating entity: {e}")
            return None

    def _create_relation(self, relation_data: Dict[str, Any], entities_in_doc: List[MaintenanceEntity], confidence_base: float) -> Optional[MaintenanceRelation]:
        """Create MaintenanceRelation from annotation data with enhanced metadata"""
        try:
            relation_id = relation_data.get("id", f"relation_{len(self.relations)}")

            # Resolve source and target entities using head/tail indices
            head_index = relation_data.get("head")
            tail_index = relation_data.get("tail")

            if head_index is None or tail_index is None:
                logger.warning(f"Skipping relation creation: missing head or tail index for {relation_data}")
                return None

            if head_index >= len(entities_in_doc) or tail_index >= len(entities_in_doc):
                logger.warning(f"Skipping relation creation: head or tail index out of bounds for {relation_data}")
                return None

            source_entity_id = entities_in_doc[head_index].entity_id
            target_entity_id = entities_in_doc[tail_index].entity_id

            if not source_entity_id or not target_entity_id:
                logger.warning(f"Skipping relation creation: could not resolve source or target entity IDs for {relation_data}")
                return None

            # ENHANCEMENT: Use scheme mapping if available
            relation_type_str = relation_data.get("type", "hasPart")

            # Get metadata for the relation type
            metadata = self.metadata_manager.get_relation_metadata(relation_type_str)

            # Check if type is active
            if metadata and not metadata.active:
                logger.debug(f"Skipping inactive relation type: {relation_type_str}")
                return None

            # Try scheme mapping first
            if relation_type_str in self.type_mappings["relation"]:
                relation_type = self.type_mappings["relation"][relation_type_str]
            else:
                # Fallback to your original logic
                try:
                    relation_type = RelationType(relation_type_str)
                except ValueError:
                    relation_type = RelationType.HAS_PART

            return MaintenanceRelation(
                relation_id=relation_id,
                source_entity=source_entity_id,
                target_entity=target_entity_id,
                relation_type=relation_type,
                confidence=min(confidence_base * relation_data.get("confidence", 1.0), 1.0),
                context=relation_data.get("context"),
                metadata={
                    "original_type": relation_type_str,
                    "color": metadata.color if metadata else "#cccccc",
                    "description": metadata.description if metadata else "",
                    "example_terms": metadata.example_terms if metadata else []
                }
            )
        except Exception as e:
            logger.warning(f"Error creating relation: {e}")
            return None

    def _build_knowledge_graph(self) -> None:
        """Build NetworkX knowledge graph from entities and relations"""
        logger.info("Building knowledge graph...")

        self.knowledge_graph = nx.Graph()

        # Add entity nodes
        for entity_id, entity in self.entities.items():
            self.knowledge_graph.add_node(
                entity_id,
                text=entity.text,
                type=entity.entity_type.value,
                confidence=entity.confidence
            )

        # Add relation edges
        for relation in self.relations:
            if (relation.source_entity in self.entities and
                relation.target_entity in self.entities):
                self.knowledge_graph.add_edge(
                    relation.source_entity,
                    relation.target_entity,
                    type=relation.relation_type.value,
                    confidence=relation.confidence,
                    relation_id=relation.relation_id
                )

        logger.info(f"Knowledge graph built: {self.knowledge_graph.number_of_nodes()} nodes, "
                   f"{self.knowledge_graph.number_of_edges()} edges")

    def _save_processed_data(self) -> None:
        """Save processed data to JSON files"""
        logger.info("Saving processed data...")

        # Save entities
        entities_data = [entity.to_dict() for entity in self.entities.values()]
        self._save_json(entities_data, self.processed_dir / "maintenance_entities.json")

        # Save relations
        relations_data = [relation.to_dict() for relation in self.relations]
        self._save_json(relations_data, self.processed_dir / "maintenance_relations.json")

        # Save documents
        documents_data = [doc.to_dict() for doc in self.documents.values()]
        self._save_json(documents_data, self.processed_dir / "maintenance_documents.json")

        # Save knowledge graph
        if self.knowledge_graph:
            # Use JSON serialization instead of gpickle
            graph_data = nx.node_link_data(self.knowledge_graph, edges="links")
            self._save_json(graph_data, self.processed_dir / "knowledge_graph.json")
            logger.info(f"Knowledge graph saved to {self.processed_dir / 'knowledge_graph.json'}")

        # Save entity vocabulary
        entity_vocab = self._build_entity_vocabulary()
        self._save_json(entity_vocab, self.processed_dir / "entity_vocabulary.json")

        logger.info("Processed data saved successfully")

    def _build_entity_vocabulary(self) -> Dict[str, Any]:
        """Build entity vocabulary for quick lookup"""
        vocab = {
            "entity_to_type": {},
            "type_to_entities": defaultdict(list),
            "entity_frequency": Counter(),
            "entity_contexts": defaultdict(list)
        }

        for entity in self.entities.values():
            vocab["entity_to_type"][entity.text] = entity.entity_type.value
            vocab["type_to_entities"][entity.entity_type.value].append(entity.text)
            vocab["entity_frequency"][entity.text] += 1
            if entity.context:
                vocab["entity_contexts"][entity.text].append(entity.context)

        # Convert defaultdict to regular dict for JSON serialization
        return {
            "entity_to_type": vocab["entity_to_type"],
            "type_to_entities": dict(vocab["type_to_entities"]),
            "entity_frequency": dict(vocab["entity_frequency"]),
            "entity_contexts": dict(vocab["entity_contexts"])
        }

    def _save_json(self, data: Any, file_path: Path) -> None:
        """Save data to JSON file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving {file_path}: {e}")

    def get_entity_by_text(self, text: str) -> Optional[MaintenanceEntity]:
        """Get entity by text match"""
        for entity in self.entities.values():
            if entity.text.lower() == text.lower():
                return entity
        return None

    def get_related_entities(self, entity_id: str, max_distance: int = 2) -> List[str]:
        """Get entities related to given entity within max distance"""
        if not self.knowledge_graph or entity_id not in self.knowledge_graph:
            return []

        try:
            # Get all nodes within max_distance
            related = []
            for target in self.knowledge_graph.nodes():
                if target != entity_id:
                    try:
                        distance = nx.shortest_path_length(
                            self.knowledge_graph, entity_id, target
                        )
                        if distance <= max_distance:
                            related.append(target)
                    except nx.NetworkXNoPath:
                        continue

            return related[:20]  # Limit results
        except Exception as e:
            logger.warning(f"Error finding related entities for {entity_id}: {e}")
            return []


def load_transformer() -> MaintIEDataTransformer:
    """Factory function to create and initialize transformer"""
    transformer = MaintIEDataTransformer()
    return transformer


if __name__ == "__main__":
    # Example usage
    transformer = MaintIEDataTransformer()
    stats = transformer.extract_maintenance_knowledge()
    print(f"Extraction complete: {stats}")
