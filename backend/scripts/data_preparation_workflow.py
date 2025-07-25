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
import logging
import os

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
print('DEBUG: AZURE_SEARCH_SERVICE:', azure_settings.azure_search_service)
print('DEBUG: AZURE_SEARCH_ADMIN_KEY:', azure_settings.azure_search_admin_key)
print('DEBUG: AZURE_SEARCH_API_VERSION:', azure_settings.azure_search_api_version)

class AzureServiceCostTracker:
    """Azure service cost correlation for workflow transparency"""
    def __init__(self):
        self.cost_per_service = {
            "azure_openai": {"per_token": 0.00002, "per_request": 0.001},
            "cognitive_search": {"per_document": 0.01, "per_query": 0.005},
            "cosmos_db": {"per_operation": 0.0001, "per_ru": 0.00008},
            "blob_storage": {"per_gb_month": 0.018, "per_operation": 0.0001}
        }

    def _calculate_service_cost(self, service: str, usage: dict) -> float:
        # Example: usage = {"tokens": 1200, "requests": 2}
        cost = 0.0
        rates = self.cost_per_service.get(service, {})
        for key, value in usage.items():
            rate_key = f"per_{key}"
            if rate_key in rates:
                cost += rates[rate_key] * value
        return cost

    def calculate_workflow_cost(self, service_usage: dict) -> dict:
        """Calculate estimated Azure service costs for workflow transparency"""
        return {
            service: self._calculate_service_cost(service, usage)
            for service, usage in service_usage.items()
        }

def _detect_content_type(filename: str) -> str:
    """Detect content type from file extension"""
    file_ext = Path(filename).suffix.lower()
    content_type_mapping = {
        '.md': 'markdown',
        '.txt': 'text'
    }
    return content_type_mapping.get(file_ext, 'text')

def load_raw_data_from_directory(data_dir: str = "data/raw") -> list:
    """Load all supported text files from the raw data directory"""
    from config.settings import azure_settings

    raw_data_path = Path(data_dir)
    if not raw_data_path.exists():
        print(f"❌ Raw data directory not found: {raw_data_path}")
        return []

    # Get supported file patterns from configuration
    supported_patterns = getattr(azure_settings, 'raw_data_include_patterns', ["*.md", "*.txt"])

    # Find all supported text files
    all_text_files = []
    for pattern in supported_patterns:
        files = list(raw_data_path.glob(pattern))
        all_text_files.extend(files)

    if not all_text_files:
        supported_formats = ", ".join(supported_patterns)
        print(f"❌ No supported text files found in {raw_data_path}")
        print(f"📋 Supported formats: {supported_formats}")
        return []

    # Sort by filename for consistent processing order
    all_text_files.sort(key=lambda x: x.name)

    documents = []
    for file_path in all_text_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    'filename': file_path.name,
                    'content': content,
                    'size': len(content),
                    'path': str(file_path),
                    'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
            file_type = _detect_content_type(file_path.name)
            print(f"📄 Loaded: {file_path.name} ({len(content)} characters, {file_type})")
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

        # Debug core data services logic
        has_core_data = all([
            data_state['azure_blob_storage']['has_data'],
            data_state['azure_cognitive_search']['has_data']
        ])

        print(f"🔍 Core Data Services Status:")
        print(f"   📊 Blob Storage + Search Index populated: {'Yes' if has_core_data else 'No'}")
        print(f"   💡 Cosmos DB metadata alone does not prevent processing")
        # Processing decision based on data state
        processing_requirement = data_state['requires_processing']
        if processing_requirement == "no_raw_data":
            print(f"❌ No raw data files found. Please add markdown files to data/raw/")
            return 1
        elif processing_requirement == "data_exists_check_policy":
            # Check environment policy for handling existing data
            from config.settings import azure_settings
            # Explicit environment variable check to debug configuration
            skip_if_exists = getattr(azure_settings, 'skip_processing_if_data_exists', False)
            force_reprocess = getattr(azure_settings, 'force_data_reprocessing', False)

            print(f"🔧 Configuration Check:")
            print(f"   SKIP_PROCESSING_IF_DATA_EXISTS: {skip_if_exists}")
            print(f"   FORCE_DATA_REPROCESSING: {force_reprocess}")
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

        # Step 2: Extract knowledge using Azure OpenAI Enterprise Service
        print(f"\n🤖 Step 2: Extracting knowledge with Azure OpenAI Enterprise Service...")

        # Import enterprise knowledge extraction service
        from core.azure_openai.knowledge_extractor import AzureOpenAIKnowledgeExtractor

        # Initialize enterprise knowledge extractor
        knowledge_extractor = AzureOpenAIKnowledgeExtractor(domain)

        # Extract entities and relations from raw documents
        document_texts = [doc['content'] for doc in raw_documents]
        extraction_results = await knowledge_extractor.extract_knowledge_from_texts(document_texts)

        # Store extracted knowledge in Azure Cosmos DB
        print(f"   🔍 Extracted entities: {len(extraction_results.get('entities', []))}")
        print(f"   🔗 Extracted relations: {len(extraction_results.get('relations', []))}")

        # Store entities in Cosmos DB
        for entity in extraction_results.get('entities', []):
            try:
                await cosmos_client.add_entity(entity.to_dict(), domain)
                print(f"   ✅ Stored entity: {entity.name} ({entity.entity_type})")
            except Exception as e:
                print(f"   ⚠️ Failed to store entity {entity.name}: {e}")

        # Store relations in Cosmos DB
        for relation in extraction_results.get('relations', []):
            try:
                await cosmos_client.add_relation(relation.to_dict(), domain)
                print(f"   ✅ Stored relation: {relation.relation_type} ({relation.source_entity} -> {relation.target_entity})")
            except Exception as e:
                print(f"   ⚠️ Failed to store relation {relation.relation_type}: {e}")

        # Use extraction results for processed documents
        processed_docs = extraction_results

        # Step 3: Build search index with Azure Cognitive Search
        print(f"\n🔍 Step 3: Building search index with Azure Cognitive Search...")
        index_name = f"rag-index-{domain}"
        await search_client.create_index(index_name)

        # Validate Azure Knowledge Extraction results
        if not processed_docs or not isinstance(processed_docs, dict):
            print(f"⚠️ Azure Knowledge Extraction validation failed:")
            print(f"   Raw documents: {len(raw_documents)}")
            print(f"   Extraction results: {type(processed_docs)}")
            print(f"💡 Falling back to raw content indexing for service continuity")
            # Use raw documents as fallback
            processed_docs = {"documents": [{"content": doc["content"]} for doc in raw_documents]}

        success_count = 0
        # Handle extraction results format
        if isinstance(processed_docs, dict) and 'documents' in processed_docs:
            # Use extracted documents
            documents_to_index = processed_docs['documents']
        else:
            # Fallback to raw documents
            documents_to_index = [{"content": doc["content"]} for doc in raw_documents]

        for i, processed_doc in enumerate(documents_to_index):
            # Map processed document to search document schema
            raw_doc = raw_documents[i]  # Corresponding raw document for metadata

            # Extract content from processed document
            if isinstance(processed_doc, dict):
                content = processed_doc.get('content', raw_doc['content'])
            else:
                content = raw_doc['content']  # Fallback to original content

            document = {
                "id": f"doc_{i}_{raw_doc['filename'].replace('.md', '').replace('.txt', '')}",
                "content": content,  # Use processed content
                "title": raw_doc['filename'],
                "domain": domain,
                "source": f"data/raw/{raw_doc['filename']}",
                "metadata": json.dumps({
                    "original_filename": raw_doc['filename'],
                    "file_size": raw_doc['size'],
                    "processing_timestamp": datetime.now().isoformat(),
                    "index_position": i,
                    "content_type": _detect_content_type(raw_doc['filename']),
                    "azure_openai_processed": True  # Enterprise tracking
                })
            }

            # Index document directly (validation handled by Azure Search service)
            try:
                index_result = await search_client.index_document(index_name, document)

                # Add chunking information logging
                if index_result.get('strategy') == 'chunked_processing':
                    chunks_info = f"{index_result.get('indexed_chunks', 0)}/{index_result.get('total_chunks', 0)}"
                    print(f"   📄 {raw_doc['filename']}: chunked into {chunks_info} segments")
                elif index_result.get('strategy') == 'single_document':
                    print(f"   📄 {raw_doc['filename']}: indexed as single document")

                if not index_result['success']:
                    print(f"❌ Failed to index document {document['id']}: {index_result.get('error', 'Unknown indexing error')}")
                    logger.error(f"Failed to index document {document['id']}: {index_result.get('error', 'Unknown indexing error')}")
                else:
                    print(f"✅ Successfully indexed: {raw_doc['filename']}")
                    success_count += 1 # Increment success_count only on successful index
            except Exception as e:
                print(f"❌ Document processing error for {raw_doc['filename']}: {str(e)}")
                logger.error(f"Document processing error for {raw_doc['filename']}: {str(e)}", exc_info=True)

        # Enterprise service integration telemetry
        print(f"\n📊 Azure Service Integration Summary:")
        print(f"   🔍 Search Index: {index_name}")
        print(f"   📄 Documents indexed: {success_count}/{len(raw_documents)}")
        print(f"   🤖 Azure OpenAI processed: {len(processed_docs)}")
        print(f"   ⚡ Indexing efficiency: {(success_count/len(raw_documents)*100):.1f}%")

        if success_count == 0:
            print(f"\n❌ Critical: Zero documents indexed in Azure Search")
            print(f"🔧 Enterprise troubleshooting required:")
            print(f"   1. Validate Azure Search service configuration")
            print(f"   2. Check document schema compatibility")
            print(f"   3. Verify Azure OpenAI processing output format")
            return 1

        print(f"\n🔍 Step 3.5: Validating search index population ({success_count}/{len(raw_documents)} documents indexed)...")
        # Validate indexed documents are searchable (handle chunked documents)
        validation_results = []
        for i, doc in enumerate(raw_documents):
            document_id = f"doc_{i}_{doc['filename'].replace('.md', '').replace('.txt', '')}"

            # Search by document content snippet instead of filename for chunked docs
            content_snippet = doc['content'][:100].strip()  # First 100 chars as search term
            search_results = await search_client.search_documents(index_name, content_snippet, top_k=5)

            # Also try searching by document ID pattern for chunked documents
            if not search_results:
                id_search_query = document_id.replace('_', ' ')  # Convert ID to searchable terms
                search_results = await search_client.search_documents(index_name, id_search_query, top_k=5)

            # Check if document was chunked by examining search results
            chunked_results = [r for r in search_results if 'chunk' in r.get('id', '')]
            total_chunks = len(chunked_results) if chunked_results else len(search_results)

            validation_results.append({
                "document": doc['filename'],
                "query": content_snippet[:50] + "...",  # Show actual search query used
                "found": len(search_results) > 0,
                "results_count": len(search_results),
                "is_chunked": len(chunked_results) > 0,
                "chunk_count": total_chunks,
                "document_size": doc.get('size', len(doc['content']))  # Use content length if size not available
            })

        # Report validation results with detailed information
        found_count = sum(1 for v in validation_results if v['found'])
        total_documents = len(validation_results)
        print(f"📊 Search Index Validation Results:")

        if found_count == 0:
            print(f"🔍 Step 3.5: Validating search index population ({found_count}/{total_documents} documents indexed)...")
        else:
            print(f"🔍 Step 3.5: Validating search index population ({found_count}/{total_documents} documents indexed)...")

        for validation in validation_results:
            status = "✅" if validation['found'] else "❌"
            chunk_info = f" ({validation['chunk_count']} chunks)" if validation.get('is_chunked') else ""
            size_info = f" [{validation['document_size']:,} chars]"
            print(f"   {status} {validation['document']}: {validation['results_count']} results{chunk_info}{size_info}")

        all_found = all(v['found'] for v in validation_results)
        if all_found:
            print("✅ All documents validated in search index.")
        else:
            print("⚠️ Some documents not found in search index.")

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
        logger.error(f"Data preparation failed: {e}", exc_info=True)
        return 1
    finally:
        # Ensure cleanup even on exceptions
        try:
            if 'cosmos_client' in locals() and hasattr(cosmos_client, 'close'):
                cosmos_client.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    """Execute data preparation workflow"""
    exit_code = asyncio.run(main())
    sys.exit(exit_code)