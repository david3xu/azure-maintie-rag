"""
MaintIE data transformation module
Loads and transforms MaintIE annotations into RAG-ready knowledge structures
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter

from src.models.maintenance_models import (
    MaintenanceEntity, MaintenanceRelation, MaintenanceDocument,
    EntityType, RelationType
)
from config.settings import settings


logger = logging.getLogger(__name__)


class MaintIEDataTransformer:
    """Transform MaintIE annotations into structured knowledge for RAG"""

    def __init__(self, gold_path: Optional[Path] = None, silver_path: Optional[Path] = None):
        """Initialize transformer with data paths"""
        from config.advanced_settings import advanced_settings

        # Use configurable filenames
        gold_filename = advanced_settings.gold_data_filename
        silver_filename = advanced_settings.silver_data_filename

        self.gold_path = gold_path or settings.raw_data_dir / gold_filename
        self.silver_path = silver_path or settings.raw_data_dir / silver_filename
        self.processed_dir = settings.processed_data_dir

        # Ensure directories exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Data containers
        self.entities: Dict[str, MaintenanceEntity] = {}
        self.relations: List[MaintenanceRelation] = []
        self.documents: Dict[str, MaintenanceDocument] = {}
        self.knowledge_graph: Optional[nx.Graph] = None

        logger.info(f"Initialized MaintIE transformer for {self.gold_path} and {self.silver_path}")

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
        gold_stats = self._process_dataset(raw_data["gold"], confidence_base=advanced_settings.gold_confidence_base)

        # Process silver data (lower confidence)
        silver_stats = self._process_dataset(raw_data["silver"], confidence_base=advanced_settings.silver_confidence_base)

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
                text = doc_data.get("text", "")

                document = MaintenanceDocument(
                    doc_id=doc_id,
                    text=text,
                    title=doc_data.get("title", f"Document {doc_id}"),
                    metadata={
                        "source": doc_data.get("source", "maintie"),
                        "confidence_base": confidence_base
                    }
                )

                # Extract entities
                entities_data = doc_data.get("entities", [])
                for entity_data in entities_data:
                    entity = self._create_entity(entity_data, confidence_base)
                    if entity:
                        self.entities[entity.entity_id] = entity
                        document.add_entity(entity)
                        stats["entities"] += 1

                # Extract relations
                relations_data = doc_data.get("relations", [])
                for relation_data in relations_data:
                    relation = self._create_relation(relation_data, confidence_base)
                    if relation:
                        self.relations.append(relation)
                        document.add_relation(relation)
                        stats["relations"] += 1

                self.documents[doc_id] = document
                stats["documents"] += 1

            except Exception as e:
                logger.warning(f"Error processing document {doc_data.get('id', 'unknown')}: {e}")
                continue

        return stats

    def _create_entity(self, entity_data: Dict[str, Any], confidence_base: float) -> Optional[MaintenanceEntity]:
        """Create MaintenanceEntity from annotation data"""
        try:
            entity_id = entity_data.get("id", f"entity_{len(self.entities)}")
            text = entity_data.get("text", "").strip()

            if not text:
                return None

            # Map entity type
            entity_type_str = entity_data.get("type", "PhysicalObject")
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
                    "start": entity_data.get("start"),
                    "end": entity_data.get("end"),
                    "original_type": entity_type_str
                }
            )
        except Exception as e:
            logger.warning(f"Error creating entity: {e}")
            return None

    def _create_relation(self, relation_data: Dict[str, Any], confidence_base: float) -> Optional[MaintenanceRelation]:
        """Create MaintenanceRelation from annotation data"""
        try:
            relation_id = relation_data.get("id", f"relation_{len(self.relations)}")
            source = relation_data.get("source", "")
            target = relation_data.get("target", "")

            if not source or not target:
                return None

            # Map relation type
            relation_type_str = relation_data.get("type", "hasPart")
            try:
                relation_type = RelationType(relation_type_str)
            except ValueError:
                relation_type = RelationType.HAS_PART

            return MaintenanceRelation(
                relation_id=relation_id,
                source_entity=source,
                target_entity=target,
                relation_type=relation_type,
                confidence=min(confidence_base * relation_data.get("confidence", 1.0), 1.0),
                context=relation_data.get("context"),
                metadata={
                    "original_type": relation_type_str
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
        """Save processed data to files"""
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
            nx.write_gpickle(self.knowledge_graph, self.processed_dir / "knowledge_graph.pkl")

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
