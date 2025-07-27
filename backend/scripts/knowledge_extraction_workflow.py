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
from pathlib import Path
from datetime import datetime
import json
import logging

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

async def load_chunks_from_azure(rag_storage, domain: str):
    """Load processed chunks from Azure Blob Storage"""

    chunks_container = f"processed-chunks-{domain}"
    index_blob_name = f"chunks_index_{domain}.json"

    try:
        # Load chunks index
        index_content = await rag_storage.download_text(chunks_container, index_blob_name)
        chunks_index = json.loads(index_content)

        print(f"📋 Found chunks index: {chunks_index['total_chunks']} chunks from {chunks_index['original_documents']} documents")
        print(f"   📅 Created: {chunks_index['created_at']}")

        # Load chunk contents
        chunks_data = []
        for i, chunk_meta in enumerate(chunks_index['chunks']):
            try:
                chunk_content = await rag_storage.download_text(chunks_container, chunk_meta['blob_name'])
                chunks_data.append({
                    'content': chunk_content,
                    'metadata': chunk_meta
                })

                if i % 50 == 0:
                    print(f"   📦 Loaded chunk {i+1}/{len(chunks_index['chunks'])}")

            except Exception as e:
                logger.warning(f"Failed to load chunk {chunk_meta['blob_name']}: {e}")
                continue

        print(f"✅ Successfully loaded {len(chunks_data)} chunks for processing")
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
    print(f"⏱️  Workflow: Load chunks → Extract knowledge → Store graph")
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

        # Step 2: Extract knowledge using Azure OpenAI Enterprise Service
        print(f"\n🤖 Step 2.2: Extracting knowledge with Azure OpenAI...")

        # Import enterprise knowledge extraction service
        from core.azure_openai.knowledge_extractor import AzureOpenAIKnowledgeExtractor

        # Initialize enterprise knowledge extractor
        knowledge_extractor = AzureOpenAIKnowledgeExtractor(domain)

        # Extract knowledge from chunks
        chunk_texts = [chunk['content'] for chunk in chunks_data]

        print(f"   🔍 Processing {len(chunk_texts)} chunks for knowledge extraction...")
        extraction_results = await knowledge_extractor.extract_knowledge_from_texts(chunk_texts)

        entities = extraction_results.get('entities', [])
        relations = extraction_results.get('relations', [])

        print(f"   ✅ Extraction completed:")
        print(f"      🔍 Entities found: {len(entities)}")
        print(f"      🔗 Relations found: {len(relations)}")

        # Step 3: Store extracted knowledge in Azure Cosmos DB
        print(f"\n💾 Step 2.3: Storing knowledge graph in Azure Cosmos DB...")

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

        # Store relations (relations are already dictionaries from extraction results)
        for relation in relations:
            try:
                cosmos_client.add_relationship(relation, domain)
                relation_success_count += 1
                if relation_success_count % 10 == 0:
                    print(f"   🔗 Stored relation {relation_success_count}/{len(relations)}")
            except Exception as e:
                logger.warning(f"Failed to store relation {relation.get('relation_type', 'unknown')}: {e}")

        # Step 4: Store extraction metadata
        print(f"\n📊 Step 2.4: Storing extraction metadata...")

        extraction_metadata = {
            "domain": domain,
            "extraction_timestamp": datetime.now().isoformat(),
            "source_chunks": len(chunk_texts),
            "entities_extracted": len(entities),
            "relations_extracted": len(relations),
            "entities_stored": entity_success_count,
            "relations_stored": relation_success_count,
            "source_documents": chunks_index.get('original_documents', 0),
            "extraction_duration_seconds": time.time() - start_time,
            "extraction_settings": {
                "discovery_sample_size": azure_settings.discovery_sample_size,
                "extraction_batch_size": azure_settings.extraction_batch_size,
                "extraction_confidence_threshold": azure_settings.extraction_confidence_threshold
            }
        }

        metadata_container = f"extraction-metadata-{domain}"
        await rag_storage.create_container(metadata_container)
        metadata_blob_name = f"extraction_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        await rag_storage.upload_text(metadata_container, metadata_blob_name, json.dumps(extraction_metadata, indent=2))

        # Final summary
        duration = time.time() - start_time
        print(f"\n📊 STEP 2 Completion Summary:")
        print(f"   📦 Chunks processed: {len(chunk_texts)}")
        print(f"   🔍 Entities extracted: {len(entities)}")
        print(f"   🔗 Relations extracted: {len(relations)}")
        print(f"   💾 Entities stored: {entity_success_count}/{len(entities)}")
        print(f"   💾 Relations stored: {relation_success_count}/{len(relations)}")
        print(f"   ⚡ Entity success rate: {(entity_success_count/max(len(entities),1)*100):.1f}%")
        print(f"   ⚡ Relation success rate: {(relation_success_count/max(len(relations),1)*100):.1f}%")
        print(f"   ⏱️  Total duration: {duration:.1f} seconds")
        print()
        print(f"✅ STEP 2 COMPLETED: Knowledge extraction and storage complete")
        print(f"🎉 Azure Universal RAG system ready for queries!")

        # Check for critical failures
        if entity_success_count == 0 and relation_success_count == 0:
            print(f"\n❌ Critical: No knowledge stored in Cosmos DB")
            return 1

        if entity_success_count < len(entities) * 0.8:  # Less than 80% success
            print(f"\n⚠️  Warning: Low entity storage success rate ({entity_success_count}/{len(entities)})")

        if relation_success_count < len(relations) * 0.8:  # Less than 80% success
            print(f"\n⚠️  Warning: Low relation storage success rate ({relation_success_count}/{len(relations)})")

        return 0

    except Exception as e:
        print(f"\n❌ Critical error in knowledge extraction workflow: {str(e)}")
        logger.error(f"Knowledge extraction workflow failed: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
