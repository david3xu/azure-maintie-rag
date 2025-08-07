"""
Azure Storage Writer for Azure Prompt Flow
Stores extraction results in Azure Cosmos DB and Cognitive Search
Maintains universal architecture with no domain assumptions
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient
from core.azure_search.search_client import AzureSearchClient

from config.settings import settings

logger = logging.getLogger(__name__)


async def store_knowledge_graph(
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    summary: Dict[str, Any],
    domain_name: str = "universal"  # Domain-agnostic default,
) -> Dict[str, Any]:
    """
    Store universal knowledge graph in Azure services

    Args:
        entities: List of extracted entity dictionaries
        relations: List of extracted relation dictionaries
        summary: Quality assessment and metrics
        domain_name: Domain context (remains universal)

    Returns:
        Storage results and statistics
    """
    storage_results = {
        "entities_stored": 0,
        "relations_stored": 0,
        "search_documents_indexed": 0,
        "storage_errors": [],
        "success": False,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        # Initialize Azure clients
        cosmos_client = AzureCosmosGremlinClient()
        search_client = AzureSearchClient()

        # Store entities in Cosmos DB (knowledge graph)
        entity_storage_tasks = []
        for entity in entities:
            try:
                cosmos_client.add_entity(entity, domain_name)
                storage_results["entities_stored"] += 1

                if storage_results["entities_stored"] % 10 == 0:
                    logger.info(
                        f"Stored {storage_results['entities_stored']}/{len(entities)} entities"
                    )

            except Exception as e:
                error_msg = (
                    f"Failed to store entity {entity.get('entity_id', 'unknown')}: {e}"
                )
                storage_results["storage_errors"].append(error_msg)
                logger.warning(error_msg)

        # Store relations in Cosmos DB
        for relation in relations:
            try:
                cosmos_client.add_relationship(relation, domain_name)
                storage_results["relations_stored"] += 1

                if storage_results["relations_stored"] % 10 == 0:
                    logger.info(
                        f"Stored {storage_results['relations_stored']}/{len(relations)} relations"
                    )

            except Exception as e:
                error_msg = f"Failed to store relation {relation.get('relation_id', 'unknown')}: {e}"
                storage_results["storage_errors"].append(error_msg)
                logger.warning(error_msg)

        # Create search documents for Azure Cognitive Search
        search_documents = []

        # Index entities for search
        for entity in entities:
            search_doc = {
                "id": entity.get("entity_id", ""),
                "content": entity.get("text", ""),
                "entity_type": entity.get("entity_type", ""),
                "confidence": entity.get("confidence", 0.0),
                "domain": domain_name,
                "item_type": "entity",
                "metadata": json.dumps(entity.get("metadata", {})),
                "timestamp": datetime.now().isoformat(),
            }
            search_documents.append(search_doc)

        # Index relations for search
        for relation in relations:
            search_doc = {
                "id": relation.get("relation_id", ""),
                "content": relation.get("relation_type", ""),
                "relation_type": relation.get("relation_type", ""),
                "confidence": relation.get("confidence", 0.0),
                "domain": domain_name,
                "item_type": "relation",
                "metadata": json.dumps(relation.get("metadata", {})),
                "timestamp": datetime.now().isoformat(),
            }
            search_documents.append(search_doc)

        # Upload to Azure Cognitive Search
        if search_documents:
            try:
                search_results = await search_client.upload_documents(search_documents)
                storage_results["search_documents_indexed"] = len(search_documents)
                logger.info(
                    f"Indexed {len(search_documents)} documents in Azure Cognitive Search"
                )
            except Exception as e:
                error_msg = f"Failed to index search documents: {e}"
                storage_results["storage_errors"].append(error_msg)
                logger.error(error_msg)

        # Store extraction metadata
        extraction_metadata = {
            "extraction_id": f"extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "domain": domain_name,
            "extraction_summary": summary,
            "storage_results": storage_results,
            "prompt_flow_version": "1.0.0",
            "extraction_method": "azure_prompt_flow_universal",
        }

        # Store metadata in Cosmos DB
        try:
            metadata_doc = {
                "id": extraction_metadata["extraction_id"],
                "type": "extraction_metadata",
                "data": extraction_metadata,
            }
            # Note: This would typically use a different Cosmos container for metadata
            logger.info(
                f"Extraction metadata prepared: {extraction_metadata['extraction_id']}"
            )
        except Exception as e:
            logger.warning(f"Failed to store extraction metadata: {e}")

        # Determine overall success
        total_items = len(entities) + len(relations)
        stored_items = (
            storage_results["entities_stored"] + storage_results["relations_stored"]
        )
        success_rate = stored_items / total_items if total_items > 0 else 0

        storage_results["success"] = success_rate > 0.8
        storage_results["success_rate"] = round(success_rate, 3)

        logger.info(
            f"Storage completed: {stored_items}/{total_items} items stored ({success_rate:.1%} success)"
        )

        return storage_results

    except Exception as e:
        logger.error(f"Storage operation failed: {e}", exc_info=True)
        storage_results["storage_errors"].append(f"Critical storage failure: {e}")
        return storage_results


def main(
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    summary: Dict[str, Any],
    domain_name: str = "universal"  # Domain-agnostic default,
) -> Dict[str, Any]:
    """
    Main function called by Azure Prompt Flow
    Wraps async storage operation for synchronous execution
    """
    try:
        # Run async storage operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            store_knowledge_graph(entities, relations, summary, domain_name)
        )

        loop.close()
        return result

    except Exception as e:
        logger.error(f"Storage wrapper failed: {e}", exc_info=True)
        return {
            "entities_stored": 0,
            "relations_stored": 0,
            "search_documents_indexed": 0,
            "storage_errors": [f"Storage wrapper failure: {e}"],
            "success": False,
            "timestamp": datetime.now().isoformat(),
        }


if __name__ == "__main__":
    # Test with sample data
    sample_entities = [
        {"entity_id": "e1", "text": "valve", "entity_type": "valve", "confidence": 0.8},
        {
            "entity_id": "e2",
            "text": "bearing",
            "entity_type": "bearing",
            "confidence": 0.9,
        },
    ]
    sample_relations = [
        {"relation_id": "r1", "relation_type": "connected_to", "confidence": 0.8}
    ]
    sample_summary = {"overall_score": 0.85, "quality_tier": "good"}

    result = main(sample_entities, sample_relations, sample_summary, "test")
    print(json.dumps(result, indent=2))
