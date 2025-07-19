#!/usr/bin/env python3
"""
Data Preparation Workflow Script with Azure Services
==================================================

Demonstrates WORKFLOW 1: Raw Text Data Handling with Azure Services
Uses Azure services to convert raw text into searchable knowledge base.

Azure Services Used:
- Azure Blob Storage (store documents)
- Azure Cognitive Search (build indices)
- Azure OpenAI (process documents)
- Azure Cosmos DB (store metadata)
"""

import sys
import asyncio
import time
from pathlib import Path
from datetime import datetime

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Import Azure services architecture components
from azure.integrations.azure_services import AzureServicesManager
from azure.integrations.azure_openai import AzureOpenAIIntegration
from config.settings import AzureSettings


async def main():
    """Execute data preparation workflow with Azure services"""

    print("🔄 WORKFLOW 1: Raw Text Data Handling with Azure Services")
    print("=" * 60)
    print("📊 Purpose: Convert raw text files into searchable knowledge base using Azure")
    print("☁️  Azure Services: Blob Storage, Cognitive Search, OpenAI, Cosmos DB")
    print("⏱️  Frequency: Once per data update (initialization/startup)")

    domain = "general"
    start_time = time.time()

    try:
        # Initialize Azure services
        print(f"\n📝 Initializing Azure services...")
        azure_services = AzureServicesManager()
        await azure_services.initialize()

        openai_integration = AzureOpenAIIntegration()
        azure_settings = AzureSettings()

        # Sample text data for processing
        sample_texts = [
            "Regular system monitoring helps prevent issues and ensures optimal performance.",
            "Documentation and record keeping are essential for tracking operational history.",
            "Proper training and procedures ensure consistent and safe operations.",
            "Quality control measures verify that standards and requirements are met.",
            "Preventive measures and regular checks help identify potential problems early."
        ]

        # Step 1: Store documents in Azure Blob Storage
        print(f"\n☁️  Step 1: Storing documents in Azure Blob Storage...")
        container_name = f"rag-data-{domain}"
        await azure_services.storage_client.create_container(container_name)

        for i, text in enumerate(sample_texts):
            blob_name = f"document_{i}.txt"
            await azure_services.storage_client.upload_text(container_name, blob_name, text)

        # Step 2: Process documents with Azure OpenAI
        print(f"\n🤖 Step 2: Processing documents with Azure OpenAI...")
        processed_docs = await openai_integration.process_documents(sample_texts, domain)

        # Step 3: Build search index with Azure Cognitive Search
        print(f"\n🔍 Step 3: Building search index with Azure Cognitive Search...")
        index_name = f"rag-index-{domain}"
        await azure_services.search_client.create_index(index_name)

        for i, text in enumerate(sample_texts):
            document = {
                "id": f"doc_{i}",
                "content": text,
                "domain": domain,
                "metadata": {"source": "workflow", "index": i}
            }
            await azure_services.search_client.index_document(index_name, document)

        # Step 4: Store metadata in Azure Cosmos DB
        print(f"\n💾 Step 4: Storing metadata in Azure Cosmos DB...")
        database_name = f"rag-metadata-{domain}"
        container_name = "documents"

        # Gremlin automatically creates graph structure
        logger.info(f"Azure Cosmos DB Gremlin graph ready for domain: {domain}")

        metadata_doc = {
            "id": f"metadata-{domain}",
            "domain": domain,
            "total_documents": len(sample_texts),
            "processed_documents": len(processed_docs),
            "index_name": index_name,
            "storage_container": container_name,
            "timestamp": datetime.now().isoformat()
        }

        await azure_services.cosmos_client.add_entity(metadata_doc, domain)

        processing_time = time.time() - start_time

        print(f"\n✅ Data preparation completed successfully!")
        print(f"⏱️  Processing time: {processing_time:.2f}s")
        print(f"📊 Documents processed: {len(sample_texts)}")
        print(f"🤖 Documents processed with Azure OpenAI: {len(processed_docs)}")
        print(f"🔍 Search index created: {index_name}")
        print(f"💾 Metadata stored in Cosmos DB: {database_name}")

        print(f"\n📋 Azure Services Usage Summary:")
        print(f"   ✅ Azure Blob Storage - Stored {len(sample_texts)} documents")
        print(f"   ✅ Azure OpenAI - Processed documents for knowledge extraction")
        print(f"   ✅ Azure Cognitive Search - Built search index")
        print(f"   ✅ Azure Cosmos DB - Stored metadata and tracking")

        print(f"\n🚀 System Status: Ready for user queries!")

    except Exception as e:
        print(f"❌ Data preparation workflow failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    """Execute data preparation workflow"""
    exit_code = asyncio.run(main())
    sys.exit(exit_code)