"""
Universal Text Processor
Replaces MaintIEDataTransformer with pure text-based processing for Universal RAG
Works with any text files - no domain assumptions, no gold/silver classification, no scheme.json
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import re
from datetime import datetime
import networkx as nx
from collections import defaultdict

from ..models.azure_rag_data_models import (
    UniversalEntity, UniversalRelation, UniversalDocument
)
from .extraction_client import OptimizedLLMExtractor
from ...config.settings import settings

logger = logging.getLogger(__name__)


class AzureOpenAITextProcessor:
    """
    Universal Text Processor for any domain

    Key Features:
    - Works with pure text files (.txt, .md, etc.)
    - No domain assumptions or hardcoded schemas
    - No gold/silver quality classification
    - LLM-powered knowledge extraction
    - Dynamic entity/relation type discovery
    - Domain-agnostic processing pipeline
    """

    def __init__(self, domain_name: str = "general"):
        """Initialize universal text processor"""
        self.domain_name = domain_name
        self.raw_data_dir = settings.raw_data_dir
        self.processed_dir = settings.processed_data_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Universal components
        self.llm_extractor = OptimizedLLMExtractor(domain_name)

        # Data containers (no domain assumptions)
        self.entities: Dict[str, UniversalEntity] = {}
        self.relations: List[UniversalRelation] = []
        self.documents: Dict[str, UniversalDocument] = {}
        self.knowledge_graph: Optional[nx.Graph] = None

        # Dynamic type discovery
        self.discovered_entity_types: Set[str] = set()
        self.discovered_relation_types: Set[str] = set()

        logger.info(f"AzureOpenAITextProcessor initialized for domain: {domain_name}")

    def load_text_files(self) -> List[Dict[str, Any]]:
        """Load all text files from raw data directory"""
        logger.info("Loading text files from raw data directory...")

        text_files = []

        # Find all text files
        for pattern in ["*.txt", "*.md"]:
            for file_path in self.raw_data_dir.glob(pattern):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()

                    if content:  # Only include non-empty files
                        text_files.append({
                            "id": file_path.stem,
                            "filename": file_path.name,
                            "path": str(file_path),
                            "text": content,
                            "title": file_path.stem.replace('_', ' ').title()
                        })

                except Exception as e:
                    logger.warning(f"Error reading {file_path}: {e}")
                    continue

        logger.info(f"Loaded {len(text_files)} text files")
        return text_files

    def extract_universal_knowledge(self) -> Dict[str, Any]:
        """Extract knowledge from text files using Universal RAG approach"""
        logger.info("Extracting universal knowledge from text files...")

        # Load text files
        text_data = self.load_text_files()

        if not text_data:
            logger.warning("No text files found - creating sample data for demonstration")
            return self._create_sample_knowledge()

        # Extract all text content
        all_texts = [item["text"] for item in text_data]

        # Step 1: LLM-powered knowledge extraction
        logger.info("🧠 Step 1: LLM knowledge extraction...")
        extraction_results = self.llm_extractor.extract_entities_and_relations(all_texts)

        # Step 2: Process documents
        logger.info("📄 Step 2: Processing documents...")
        doc_stats = self._process_documents(text_data)

        # Step 3: Extract entities with discovered types
        logger.info("🏷️ Step 3: Extracting entities...")
        entity_stats = self._extract_universal_entities(text_data, extraction_results)

        # Step 4: Extract relations with discovered types
        logger.info("🔗 Step 4: Extracting relations...")
        relation_stats = self._extract_universal_relations(text_data, extraction_results)

        # Step 5: Build knowledge graph
        logger.info("🕸️ Step 5: Building knowledge graph...")
        self._build_universal_knowledge_graph()

        # Step 6: Save processed data
        logger.info("💾 Step 6: Saving processed data...")
        self._save_universal_processed_data()

        # Compile statistics
        stats = {
            "domain_name": self.domain_name,
            "processing_approach": "universal_text_based",
            "total_text_files": len(text_data),
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "total_documents": len(self.documents),
            "discovered_entity_types": len(self.discovered_entity_types),
            "discovered_relation_types": len(self.discovered_relation_types),
            "graph_nodes": self.knowledge_graph.number_of_nodes() if self.knowledge_graph else 0,
            "graph_edges": self.knowledge_graph.number_of_edges() if self.knowledge_graph else 0,
            "document_stats": doc_stats,
            "entity_stats": entity_stats,
            "relation_stats": relation_stats,
            "extraction_results": extraction_results
        }

        logger.info(f"Universal knowledge extraction complete: {stats}")
        return stats

    def _process_documents(self, text_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Process text files into universal documents"""
        stats = {"processed": 0, "skipped": 0}

        for text_item in text_data:
            try:
                doc_id = text_item["id"]

                document = UniversalDocument(
                    doc_id=doc_id,
                    text=text_item["text"],
                    title=text_item["title"],
                    metadata={
                        "filename": text_item["filename"],
                        "source": "text_file",
                        "domain": self.domain_name,
                        "processing_method": "universal_text_processor"
                    }
                )

                self.documents[doc_id] = document
                stats["processed"] += 1

            except Exception as e:
                logger.warning(f"Error processing document {text_item.get('id', 'unknown')}: {e}")
                stats["skipped"] += 1
                continue

        return stats

    def _extract_universal_entities(self, text_data: List[Dict[str, Any]],
                                  extraction_results: Dict[str, Any]) -> Dict[str, int]:
        """Extract entities using LLM discoveries - no hardcoded types"""
        stats = {"extracted": 0, "skipped": 0}

        # Get discovered entity types from LLM
        discovered_entities = extraction_results.get("entities", [])
        self.discovered_entity_types.update(discovered_entities)

        # Simple entity extraction from text using discovered types
        entity_counter = 0

        for text_item in text_data:
            doc_id = text_item["id"]
            text = text_item["text"].lower()

            # Extract entities based on LLM discoveries
            for entity_type in discovered_entities:
                # Simple pattern matching for demonstration
                # In production, this would use more sophisticated NLP
                entity_pattern = entity_type.replace('_', ' ')

                if entity_pattern in text:
                    entity_id = f"{doc_id}_entity_{entity_counter}"

                    entity = UniversalEntity(
                        entity_id=entity_id,
                        text=entity_pattern,
                        entity_type=entity_type,
                        confidence=0.8,  # Default confidence
                        context=text_item["title"],
                        metadata={
                            "doc_id": doc_id,
                            "discovery_method": "llm_guided_extraction",
                            "domain": self.domain_name
                        }
                    )

                    self.entities[entity_id] = entity
                    self.documents[doc_id].add_entity(entity)
                    stats["extracted"] += 1
                    entity_counter += 1

        return stats

    def _extract_universal_relations(self, text_data: List[Dict[str, Any]],
                                   extraction_results: Dict[str, Any]) -> Dict[str, int]:
        """Extract relations using LLM discoveries - no hardcoded types"""
        stats = {"extracted": 0, "skipped": 0}

        # Get discovered relation types from LLM
        discovered_relations = extraction_results.get("relations", [])
        self.discovered_relation_types.update(discovered_relations)

        # For demonstration, create simple relations between entities in same document
        relation_counter = 0

        for doc_id, document in self.documents.items():
            entities_in_doc = list(document.entities.values())

            # Create relations between entities in same document
            for i in range(len(entities_in_doc)):
                for j in range(i + 1, min(i + 3, len(entities_in_doc))):  # Limit relations
                    if discovered_relations:
                        relation_type = discovered_relations[0]  # Use first discovered relation

                        relation = UniversalRelation(
                            relation_id=f"relation_{relation_counter}",
                            source_entity_id=entities_in_doc[i].entity_id,
                            target_entity_id=entities_in_doc[j].entity_id,
                            relation_type=relation_type,
                            confidence=0.7,
                            metadata={
                                "doc_id": doc_id,
                                "discovery_method": "proximity_based",
                                "domain": self.domain_name
                            }
                        )

                        self.relations.append(relation)
                        document.add_relation(relation)
                        stats["extracted"] += 1
                        relation_counter += 1

        return stats

    def _build_universal_knowledge_graph(self):
        """Build knowledge graph from universal entities and relations"""
        self.knowledge_graph = nx.Graph()

        # Add entity nodes
        for entity_id, entity in self.entities.items():
            self.knowledge_graph.add_node(
                entity_id,
                text=entity.text,
                entity_type=entity.entity_type,
                confidence=entity.confidence
            )

        # Add relation edges
        for relation in self.relations:
            if (relation.source_entity_id in self.knowledge_graph.nodes and
                relation.target_entity_id in self.knowledge_graph.nodes):

                self.knowledge_graph.add_edge(
                    relation.source_entity_id,
                    relation.target_entity_id,
                    relation_type=relation.relation_type,
                    confidence=relation.confidence
                )

    def _save_universal_processed_data(self):
        """Save processed universal data"""
        try:
            # Save entities
            entities_data = [entity.to_dict() for entity in self.entities.values()]
            with open(self.processed_dir / "universal_entities.json", 'w') as f:
                json.dump(entities_data, f, indent=2)

            # Save relations
            relations_data = [relation.to_dict() for relation in self.relations]
            with open(self.processed_dir / "universal_relations.json", 'w') as f:
                json.dump(relations_data, f, indent=2)

            # Save documents
            documents_data = [doc.to_dict() for doc in self.documents.values()]
            with open(self.processed_dir / "universal_documents.json", 'w') as f:
                json.dump(documents_data, f, indent=2)

            # Save knowledge graph
            if self.knowledge_graph:
                graph_data = nx.node_link_data(self.knowledge_graph)
                with open(self.processed_dir / "universal_knowledge_graph.json", 'w') as f:
                    json.dump(graph_data, f, indent=2)

            # Save discovered types
            types_data = {
                "entity_types": list(self.discovered_entity_types),
                "relation_types": list(self.discovered_relation_types),
                "domain": self.domain_name,
                "processing_date": datetime.now().isoformat()
            }
            with open(self.processed_dir / "universal_types.json", 'w') as f:
                json.dump(types_data, f, indent=2)

            logger.info("Universal processed data saved successfully")

        except Exception as e:
            logger.error(f"Error saving universal processed data: {e}")

    def _create_sample_knowledge(self) -> Dict[str, Any]:
        """Create sample knowledge for demonstration when no text files exist"""
        logger.info("Creating sample knowledge for demonstration...")

        # Create sample document
        sample_doc = UniversalDocument(
            doc_id="sample_doc",
            text="This is a sample document for Universal RAG demonstration.",
            title="Sample Document",
            metadata={"source": "sample_data", "domain": self.domain_name}
        )
        self.documents["sample_doc"] = sample_doc

        # Create sample entity
        sample_entity = UniversalEntity(
            entity_id="sample_entity",
            text="sample text",
            entity_type="sample_type",
            confidence=0.8,
            metadata={"source": "sample_data"}
        )
        self.entities["sample_entity"] = sample_entity
        self.discovered_entity_types.add("sample_type")

        return {
            "domain_name": self.domain_name,
            "processing_approach": "universal_sample",
            "total_entities": 1,
            "total_documents": 1,
            "message": "Sample data created - add .txt/.md files to data/raw/ for real processing"
        }

    def get_text_corpus(self) -> List[str]:
        """Get all text content as a corpus for further processing"""
        return [doc.text for doc in self.documents.values()]

    def get_domain_statistics(self) -> Dict[str, Any]:
        """Get statistics about the discovered domain"""
        return {
            "domain_name": self.domain_name,
            "documents": len(self.documents),
            "entities": len(self.entities),
            "relations": len(self.relations),
            "entity_types": list(self.discovered_entity_types),
            "relation_types": list(self.discovered_relation_types),
            "text_files_processed": len([d for d in self.documents.values()
                                       if d.metadata.get("source") == "text_file"])
        }


def create_universal_processor(domain_name: str = "general") -> AzureOpenAITextProcessor:
    """Factory function to create universal text processor"""
    return AzureOpenAITextProcessor(domain_name)


if __name__ == "__main__":
    # Example usage
    processor = AzureOpenAITextProcessor("maintenance")
    stats = processor.extract_universal_knowledge()
    print(f"Universal knowledge extraction complete: {stats}")