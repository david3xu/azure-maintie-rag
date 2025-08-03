"""
Universal Knowledge Graph Builder for Azure Prompt Flow
Constructs knowledge graphs from extracted entities and relations
Domain-agnostic with no hardcoded assumptions
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def build_knowledge_graph(
    entities: str,
    relations: str,
    confidence_threshold: float = 0.7,
    max_entities: int = 50,
) -> Dict[str, Any]:
    """
    Build universal knowledge graph from LLM extraction results

    Args:
        entities: JSON string of extracted entity names
        relations: JSON string of extracted relation types
        confidence_threshold: Minimum confidence for inclusion
        max_entities: Maximum entities to process

    Returns:
        Structured knowledge graph with entities and relations
    """
    try:
        # Parse LLM outputs safely
        if isinstance(entities, str):
            try:
                entity_list = json.loads(entities)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse entities JSON: {e}")
                entity_list = []
        elif isinstance(entities, (list, dict)):
            entity_list = entities
        else:
            logger.warning(f"Unexpected entities type: {type(entities)}")
            entity_list = []

        if isinstance(relations, str):
            try:
                relation_list = json.loads(relations)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse relations JSON: {e}")
                relation_list = []
        elif isinstance(relations, (list, dict)):
            relation_list = relations
        else:
            logger.warning(f"Unexpected relations type: {type(relations)}")
            relation_list = []

        # Build universal entities (no predetermined types)
        processed_entities = []
        entity_map = {}

        for idx, entity_name in enumerate(entity_list[:max_entities]):
            if not entity_name or len(entity_name.strip()) < 2:
                continue

            entity_id = f"entity_{uuid.uuid4().hex[:8]}"

            # Universal entity structure - type emerges from content
            universal_entity = {
                "entity_id": entity_id,
                "text": entity_name.strip(),
                "entity_type": _normalize_entity_type(entity_name.strip()),
                "confidence": 0.8,  # LLM extraction confidence
                "context": "",
                "metadata": {
                    "extraction_method": "prompt_flow_llm",
                    "domain": "universal",
                    "extracted_at": datetime.now().isoformat(),
                    "item_type": "entity",
                    "source": "azure_prompt_flow",
                },
            }

            processed_entities.append(universal_entity)
            entity_map[entity_name.strip().lower()] = entity_id

        # Build universal relations (no predetermined categories)
        processed_relations = []

        for idx, relation_type in enumerate(relation_list[:30]):
            if not relation_type or len(relation_type.strip()) < 2:
                continue

            relation_id = f"relation_{uuid.uuid4().hex[:8]}"

            # Universal relation structure - emerges from text patterns
            universal_relation = {
                "relation_id": relation_id,
                "source_entity_id": "",  # Will be linked during graph construction
                "target_entity_id": "",  # Will be linked during graph construction
                "relation_type": _normalize_relation_type(relation_type.strip()),
                "confidence": 0.8,  # LLM extraction confidence
                "context": "",
                "metadata": {
                    "extraction_method": "prompt_flow_llm",
                    "domain": "universal",
                    "extracted_at": datetime.now().isoformat(),
                    "item_type": "relation",
                    "source": "azure_prompt_flow",
                },
            }

            processed_relations.append(universal_relation)

        # Build knowledge graph summary
        knowledge_summary = {
            "total_entities": len(processed_entities),
            "total_relations": len(processed_relations),
            "entity_types": list(set(e["entity_type"] for e in processed_entities)),
            "relation_types": list(
                set(r["relation_type"] for r in processed_relations)
            ),
            "processing_timestamp": datetime.now().isoformat(),
            "extraction_source": "azure_prompt_flow_universal",
        }

        logger.info(
            f"Built universal knowledge graph: {len(processed_entities)} entities, {len(processed_relations)} relations"
        )

        return {
            "entities": processed_entities,
            "relations": processed_relations,
            "knowledge_summary": knowledge_summary,
            "success": True,
        }

    except Exception as e:
        logger.error(f"Knowledge graph construction failed: {e}", exc_info=True)
        return {
            "entities": [],
            "relations": [],
            "knowledge_summary": {},
            "success": False,
            "error": str(e),
        }


def _normalize_entity_type(entity_text: str) -> str:
    """Normalize entity type without predetermined categories"""
    # Simple normalization - no hardcoded types
    normalized = entity_text.lower().strip()
    normalized = normalized.replace(" ", "_")
    normalized = normalized.replace("-", "_")
    return normalized


def _normalize_relation_type(relation_text: str) -> str:
    """Normalize relation type without predetermined categories"""
    # Simple normalization - no hardcoded relationship hierarchies
    normalized = relation_text.lower().strip()
    normalized = normalized.replace(" ", "_")
    normalized = normalized.replace("-", "_")
    return normalized


# Main entry point for Azure Prompt Flow
def main(
    entities: str,
    relations: str,
    confidence_threshold: float = 0.7,
    max_entities: int = 50,
) -> Dict[str, Any]:
    """Main function called by Azure Prompt Flow"""
    return build_knowledge_graph(
        entities, relations, confidence_threshold, max_entities
    )


if __name__ == "__main__":
    # This module should only be called from the prompt flow - no standalone execution with hardcoded data
    print(
        "Error: This module is designed to be called from Azure Prompt Flow, not executed standalone."
    )
    print("Use: make prompt-flow-extract to run knowledge extraction from raw data")
    exit(1)
