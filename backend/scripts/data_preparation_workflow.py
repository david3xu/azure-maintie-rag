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
import json
import glob

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Import Azure services architecture components
from integrations.azure_services import AzureServicesManager
from integrations.azure_openai import AzureOpenAIClient
from config.settings import AzureSettings


def load_raw_data_from_directory(data_dir: str = "data/raw") -> list:
    """Load all markdown files from the raw data directory"""
    raw_data_path = Path(data_dir)
    if not raw_data_path.exists():
        print(f"❌ Raw data directory not found: {raw_data_path}")
        return []

    # Find all markdown files
    markdown_files = list(raw_data_path.glob("*.md"))

    if not markdown_files:
        print(f"❌ No markdown files found in {raw_data_path}")
        return []

    documents = []
    for file_path in markdown_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    'filename': file_path.name,
                    'content': content,
                    'size': len(content),
                    'path': str(file_path)
                })
            print(f"📄 Loaded: {file_path.name} ({len(content)} characters)")
        except Exception as e:
            print(f"❌ Error loading {file_path.name}: {e}")

    return documents


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
        # Step 0: Azure Data State Validation (NEW)
        print(f"\n🔍 Step 0: Azure Data State Analysis...")
        # Initialize Azure services (existing pattern)
        azure_services = AzureServicesManager()
        validation = azure_services.validate_configuration()
        if not validation['all_configured']:
            raise RuntimeError(f"Azure services not properly configured: {validation}")
        # Validate Azure data state
        data_state = await azure_services.validate_domain_data_state(domain)
        # Display Azure service state
        print(f"📊 Azure Services Data State:")
        print(f"   🗄️  Azure Blob Storage: {'✅ Has Data' if data_state['azure_blob_storage']['has_data'] else '❌ Empty'} ({data_state['azure_blob_storage']['document_count']} docs)")
        print(f"   🔍 Azure Cognitive Search: {'✅ Has Index' if data_state['azure_cognitive_search']['has_data'] else '❌ No Index'} ({data_state['azure_cognitive_search']['document_count']} docs)")
        print(f"   💾 Azure Cosmos DB: {'✅ Has Metadata' if data_state['azure_cosmos_db']['has_data'] else '❌ No Metadata'} ({data_state['azure_cosmos_db']['vertex_count']} entities)")
        print(f"   📁 Raw Data: {'✅ Available' if data_state['raw_data_directory']['has_files'] else '❌ Missing'} ({data_state['raw_data_directory']['file_count']} files)")
        # Processing decision based on data state
        processing_requirement = data_state['requires_processing']
        if processing_requirement == "no_raw_data":
            print(f"❌ No raw data files found. Please add markdown files to data/raw/")
            return 1
        elif processing_requirement == "data_exists_check_policy":
            # Check environment policy for handling existing data
            from config.settings import azure_settings
            skip_if_exists = getattr(azure_settings, 'skip_processing_if_data_exists', False)
            force_reprocess = getattr(azure_settings, 'force_data_reprocessing', False)
            if skip_if_exists and not force_reprocess:
                print(f"⏭️  Skipping data preparation - Azure services already contain data for domain '{domain}'")
                print(f"💡 To force reprocessing, set FORCE_DATA_REPROCESSING=true in environment")
                print(f"⏱️  Processing time: {time.time() - start_time:.2f}s (skipped)")
                return 0
            elif force_reprocess:
                print(f"🔄 Force reprocessing enabled - proceeding with data preparation...")
            else:
                print(f"⚠️  Existing data detected. Configure processing policy in environment:")
                print(f"    SKIP_PROCESSING_IF_DATA_EXISTS=true  # Skip if data exists")
                print(f"    FORCE_DATA_REPROCESSING=true        # Always reprocess")
                return 1
        print(f"✅ Proceeding with Azure data preparation workflow...")

        # Load raw data from data/raw directory
        print(f"\n📂 Loading raw data from data/raw directory...")
        raw_documents = load_raw_data_from_directory()

        if not raw_documents:
            print("❌ No raw documents found. Please add markdown files to data/raw/")
            return 1

        print(f"✅ Loaded {len(raw_documents)} documents from data/raw/")

        # Initialize Azure services
        print(f"\n📝 Initializing Azure services...")
        azure_services = AzureServicesManager()
        # Remove the await azure_services.initialize() call

        # Validate services instead
        print(f"\n📝 Validating Azure services configuration...")
        validation = azure_services.validate_configuration()
        if not validation['all_configured']:
            raise RuntimeError(f"Azure services not properly configured: {validation}")

        openai_integration = AzureOpenAIClient()
        azure_settings = AzureSettings()

        # Use get_rag_storage_client for storage
        rag_storage = azure_services.get_rag_storage_client()
        search_client = azure_services.get_service('search')
        cosmos_client = azure_services.get_service('cosmos')

        # Step 1: Store documents in Azure Blob Storage using RAG storage
        print(f"\n☁️  Step 1: Storing documents in Azure Blob Storage (RAG)...")
        container_name = f"rag-data-{domain}"
        await rag_storage.create_container(container_name)

        for doc in raw_documents:
            blob_name = f"{doc['filename']}"
            await rag_storage.upload_text(container_name, blob_name, doc['content'])

        # Step 2: Process documents with Azure OpenAI
        print(f"\n🤖 Step 2: Processing documents with Azure OpenAI...")
        # Extract content from documents for processing
        document_contents = [doc['content'] for doc in raw_documents]
        processed_docs = await openai_integration.process_documents(document_contents, domain)

        # Step 3: Build search index with Azure Cognitive Search
        print(f"\n🔍 Step 3: Building search index with Azure Cognitive Search...")
        index_name = f"rag-index-{domain}"
        await search_client.create_index(index_name)

        for i, doc in enumerate(raw_documents):
            # Use consistent document ID pattern
            document_id = f"doc_{i}_{doc['filename'].replace('.md', '').replace('.txt', '')}"

            document = {
                "id": document_id,  # Must match search client expectations
                "content": doc['content'],
                "title": doc['filename'],
                "domain": domain,
                "source": f"data/raw/{doc['filename']}",  # Full source path
                "metadata": json.dumps({
                    "original_filename": doc['filename'],
                    "file_size": doc['size'],
                    "processing_timestamp": datetime.now().isoformat(),
                    "index_position": i,
                    "content_type": "markdown" if doc['filename'].endswith('.md') else "text"
                })
            }

            # Validate document structure before indexing
            validation = await search_client.validate_document_structure(document)
            if validation['valid']:
                index_result = await search_client.index_document(index_name, document)
                if not index_result['success']:
                    logger.error(f"Failed to index document {document_id}: {index_result['error']}")
            else:
                logger.error(f"Document structure validation failed: {validation['errors']}")

        # Step 3.5: Validate search index population
        print(f"\n🔍 Step 3.5: Validating search index population...")

        # Wait for indexing to complete
        import asyncio
        await asyncio.sleep(2)  # Allow Azure Search indexing propagation

        # Validate indexed documents are searchable
        validation_results = []
        for i, doc in enumerate(raw_documents):
            test_query = doc['filename'].replace('.md', '').replace('.txt', '')
            search_results = await search_client.search_documents(index_name, test_query, top_k=1)

            validation_results.append({
                "document": doc['filename'],
                "query": test_query,
                "found": len(search_results) > 0,
                "results_count": len(search_results)
            })

        # Report validation results
        found_count = sum(1 for v in validation_results if v['found'])
        print(f"📊 Index Validation: {found_count}/{len(validation_results)} documents searchable")

        if found_count == 0:
            print(f"⚠️  Warning: No documents found in search index - investigating...")
            # Get index statistics for diagnostics
            index_stats = await search_client._get_index_statistics(index_name)
            print(f"📈 Index Statistics: {index_stats}")

        for validation in validation_results:
            status = "✅" if validation['found'] else "❌"
            print(f"   {status} {validation['document']}: {validation['results_count']} results")

        # Step 4: Store metadata in Azure Cosmos DB
        print(f"\n💾 Step 4: Storing metadata in Azure Cosmos DB...")
        database_name = f"rag-metadata-{domain}"
        container_name = "documents"

        # Gremlin automatically creates graph structure
        # logger.info(f"Azure Cosmos DB Gremlin graph ready for domain: {domain}") # This line was commented out in the original file

        metadata_doc = {
            "id": f"metadata-{domain}",
            "domain": domain,
            "total_documents": len(raw_documents),
            "processed_documents": len(processed_docs),
            "index_name": index_name,
            "storage_container": container_name,
            "timestamp": datetime.now().isoformat(),
            "source_directory": "data/raw",
            "file_types": list(set([doc['filename'].split('.')[-1] for doc in raw_documents]))
        }

        cosmos_client.add_entity(metadata_doc, domain)

        processing_time = time.time() - start_time

        print(f"\n✅ Data preparation completed successfully!")
        print(f"⏱️  Processing time: {processing_time:.2f}s")
        print(f"📊 Documents processed: {len(raw_documents)}")
        print(f"🤖 Documents processed with Azure OpenAI: {len(processed_docs)}")
        print(f"🔍 Search index created: {index_name}")
        print(f"💾 Metadata stored in Cosmos DB: {database_name}")

        print(f"\n📋 Azure Services Usage Summary:")
        print(f"   ✅ Azure Blob Storage - Stored {len(raw_documents)} documents")
        print(f"   ✅ Azure OpenAI - Processed documents for knowledge extraction")
        print(f"   ✅ Azure Cognitive Search - Built search index")
        print(f"   ✅ Azure Cosmos DB - Stored metadata and tracking")

        print(f"\n📁 Raw Data Summary:")
        for doc in raw_documents:
            print(f"   📄 {doc['filename']} ({doc['size']} characters)")

        print(f"\n🚀 System Status: Ready for user queries!")

    except Exception as e:
        print(f"❌ Data preparation workflow failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    """Execute data preparation workflow"""
    exit_code = asyncio.run(main())
    sys.exit(exit_code)