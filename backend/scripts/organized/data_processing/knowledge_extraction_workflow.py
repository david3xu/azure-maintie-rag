#!/usr/bin/env python3
"""
Knowledge Extraction Workflow Script
====================================

STEP 2 of Azure Universal RAG Pipeline
Processes stored chunks to extract entities and relations using Azure services.

Azure Services Used:
- Azure Blob Storage (read stored chunks)
- Azure OpenAI (entity and relation extraction)
- Azure Cosmos DB (store knowledge graph)
- Azure Text Analytics (text preprocessing)
"""

import sys
import asyncio
import time
import re
import hashlib
from pathlib import Path
from datetime import datetime
import json
import logging
from typing import List, Dict, Any, Set

# Configure logging for better error visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Import Azure services architecture components
from integrations.azure_services import AzureServicesManager
from integrations.azure_openai import AzureOpenAIClient
from config.settings import AzureSettings
from core.workflow.data_workflow_evidence import AzureDataWorkflowEvidenceCollector
from core.workflow.cost_tracker import AzureServiceCostTracker

azure_settings = AzureSettings()


def normalize_maintenance_text(text: str) -> str:
    """
    Normalize maintenance text for deduplication by removing IDs, numbers, dates
    to find semantic duplicates
    """
    # Remove common placeholders and IDs
    normalized = re.sub(r'<id>|<num>|<date>', '', text.lower())
    # Remove specific position numbers
    normalized = re.sub(r'position \d+', 'position', normalized)
    # Remove specific quantities
    normalized = re.sub(r'\d+ x', '', normalized)
    # Remove specific measurements
    normalized = re.sub(r'\d+ PSI|\d+ V|\d+ amp', '', normalized)
    # Clean up extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def deduplicate_maintenance_texts(texts: List[str]) -> List[str]:
    """
    Remove duplicate maintenance texts before LLM processing
    Returns unique texts while preserving original order
    """
    seen = set()
    unique_texts = []
    duplicates_removed = 0

    for text in texts:
        normalized = normalize_maintenance_text(text)
        if normalized not in seen:
            seen.add(normalized)
            unique_texts.append(text)
        else:
            duplicates_removed += 1

    logger.info(f"Deduplication: Removed {duplicates_removed} duplicate texts from {len(texts)} total")
    logger.info(f"Deduplication: Kept {len(unique_texts)} unique texts")

    return unique_texts


def deduplicate_relationships(relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate relationships based on source, target, and relation type
    """
    seen = set()
    unique_relationships = []
    duplicates_removed = 0

    for rel in relationships:
        # Create a unique key for the relationship
        source = rel.get('source_entity', '').lower().strip()
        target = rel.get('target_entity', '').lower().strip()
        rel_type = rel.get('relation_type', '').lower().strip()

        # Skip empty relationships
        if not source or not target or not rel_type:
            continue

        rel_key = f"{source}|{rel_type}|{target}"

        if rel_key not in seen:
            seen.add(rel_key)
            unique_relationships.append(rel)
        else:
            duplicates_removed += 1

    logger.info(f"Relationship deduplication: Removed {duplicates_removed} duplicate relationships")
    logger.info(f"Relationship deduplication: Kept {len(unique_relationships)} unique relationships")

    return unique_relationships


async def load_chunks_from_azure(rag_storage, domain: str) -> tuple:
    """Load processed chunks from Azure Blob Storage"""
    try:
        container_name = f"{domain}-chunks"
        chunks_data = await rag_storage.list_texts(container_name)

        if not chunks_data:
            logger.warning(f"No chunks found in container {container_name}")
            return [], {}

        # Create index for quick lookup
        chunks_index = {chunk['name']: chunk for chunk in chunks_data}

        logger.info(f"Loaded {len(chunks_data)} chunks from Azure Blob Storage")
        return chunks_data, chunks_index

    except Exception as e:
        logger.error(f"Failed to load chunks from Azure: {e}")
        return [], {}


async def main():
    """Execute knowledge extraction workflow with Azure services"""

    # Initialize workflow tracking
    start_time = time.time()
    domain = sys.argv[1] if len(sys.argv) > 1 else "general"

    print(f"🔄 STEP 2: Knowledge Extraction from Chunks")
    print(f"============================================================")
    print(f"📊 Purpose: Extract entities and relations from processed chunks")
    print(f"☁️  Azure Services: OpenAI, Cosmos DB, Text Analytics")
    print(f"⏱️  Workflow: Load chunks → Deduplicate → Extract knowledge → Store graph")
    print()

    try:
        # Initialize Azure services
        print(f"📝 Initializing Azure services...")
        azure_services = AzureServicesManager()

        # Validate services
        print(f"📝 Validating Azure services configuration...")
        validation = azure_services.validate_configuration()
        if not validation['all_configured']:
            raise RuntimeError(f"Azure services not properly configured: {validation}")

        openai_integration = AzureOpenAIClient()
        rag_storage = azure_services.get_rag_storage_client()
        cosmos_client = azure_services.get_service('cosmos')

        # Step 1: Load processed chunks from Azure Blob Storage
        print(f"\n☁️  Step 2.1: Loading processed chunks from Azure Blob Storage...")
        chunks_data, chunks_index = await load_chunks_from_azure(rag_storage, domain)

        if not chunks_data:
            print("❌ No processed chunks found. Please run 'make data-upload' first.")
            return 1

        # Step 2: Deduplicate chunks before processing
        print(f"\n🔍 Step 2.2: Deduplicating maintenance texts...")
        chunk_texts = [chunk['content'] for chunk in chunks_data]

        # Apply deduplication
        unique_chunk_texts = deduplicate_maintenance_texts(chunk_texts)

        print(f"   📊 Deduplication results:")
        print(f"      📝 Original chunks: {len(chunk_texts)}")
        print(f"      ✅ Unique chunks: {len(unique_chunk_texts)}")
        print(f"      🗑️  Duplicates removed: {len(chunk_texts) - len(unique_chunk_texts)}")

        # Step 3: Extract knowledge using Azure OpenAI Enterprise Service
        print(f"\n🤖 Step 2.3: Extracting knowledge with Azure OpenAI...")

        # Import enterprise knowledge extraction service
        from core.azure_openai.knowledge_extractor import AzureOpenAIKnowledgeExtractor

        # Initialize enterprise knowledge extractor
        knowledge_extractor = AzureOpenAIKnowledgeExtractor(domain)

        print(f"   🔍 Processing {len(unique_chunk_texts)} unique chunks for knowledge extraction...")
        extraction_results = await knowledge_extractor.extract_knowledge_from_texts(unique_chunk_texts)

        entities = extraction_results.get('entities', [])
        relations = extraction_results.get('relations', [])

        print(f"   ✅ Initial extraction completed:")
        print(f"      🔍 Entities found: {len(entities)}")
        print(f"      🔗 Relations found: {len(relations)}")

        # Step 4: Deduplicate relationships
        print(f"\n🔍 Step 2.4: Deduplicating relationships...")
        unique_relations = deduplicate_relationships(relations)

        print(f"   📊 Relationship deduplication results:")
        print(f"      🔗 Original relations: {len(relations)}")
        print(f"      ✅ Unique relations: {len(unique_relations)}")
        print(f"      🗑️  Duplicate relations removed: {len(relations) - len(unique_relations)}")

        # Step 5: Store extracted knowledge in Azure Cosmos DB
        print(f"\n💾 Step 2.5: Storing knowledge graph in Azure Cosmos DB...")

        entity_success_count = 0
        relation_success_count = 0

        # Store entities (entities are already dictionaries from extraction results)
        for entity in entities:
            try:
                cosmos_client.add_entity(entity, domain)
                entity_success_count += 1
                if entity_success_count % 10 == 0:
                    print(f"   📝 Stored entity {entity_success_count}/{len(entities)}")
            except Exception as e:
                logger.warning(f"Failed to store entity {entity.get('entity_id', 'unknown')} ({entity.get('text', 'unknown')}): {e}")

        # Store deduplicated relations
        for relation in unique_relations:
            try:
                cosmos_client.add_relationship(relation, domain)
                relation_success_count += 1
                if relation_success_count % 10 == 0:
                    print(f"   🔗 Stored relation {relation_success_count}/{len(unique_relations)}")
            except Exception as e:
                logger.warning(f"Failed to store relation {relation.get('relation_type', 'unknown')}: {e}")

        # Step 6: Store extraction metadata
        print(f"\n📊 Step 2.6: Storing extraction metadata...")

        metadata = {
            "extraction_timestamp": datetime.now().isoformat(),
            "domain": domain,
            "original_chunks": len(chunk_texts),
            "unique_chunks": len(unique_chunk_texts),
            "duplicates_removed": len(chunk_texts) - len(unique_chunk_texts),
            "entities_extracted": len(entities),
            "relations_extracted": len(relations),
            "unique_relations_stored": len(unique_relations),
            "duplicate_relations_removed": len(relations) - len(unique_relations),
            "entities_stored": entity_success_count,
            "relations_stored": relation_success_count,
            "deduplication_applied": True,
            "workflow_version": "2.0_with_deduplication"
        }

        # Save metadata locally
        metadata_file = Path(f"extraction_metadata_{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✅ Knowledge extraction workflow completed successfully!")
        print(f"📊 Final Statistics:")
        print(f"   📝 Original chunks: {len(chunk_texts)}")
        print(f"   ✅ Unique chunks processed: {len(unique_chunk_texts)}")
        print(f"   🗑️  Duplicates removed: {len(chunk_texts) - len(unique_chunk_texts)}")
        print(f"   🔍 Entities extracted: {len(entities)}")
        print(f"   🔗 Relations extracted: {len(relations)}")
        print(f"   ✅ Unique relations stored: {len(unique_relations)}")
        print(f"   🗑️  Duplicate relations removed: {len(relations) - len(unique_relations)}")
        print(f"   💾 Metadata saved: {metadata_file}")

        return 0

    except Exception as e:
        logger.error(f"Knowledge extraction workflow failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
