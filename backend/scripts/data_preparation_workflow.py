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
from core.workflow.progress_tracker import create_progress_tracker, track_async_operation, track_sync_operation
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

    # Initialize real-time progress tracker
    progress_tracker = create_progress_tracker("Azure Data Preparation")
    progress_tracker.start_workflow()

    domain = "general"
    start_time = time.time()

    try:
        # Step 0: Azure Data State Validation (NEW)
        progress_tracker.start_step("Initialization", {"domain": domain})
        # Initialize Azure services (existing pattern)
        azure_services = AzureServicesManager()
        validation = azure_services.validate_configuration()
        if not validation['all_configured']:
            progress_tracker.complete_step("Initialization", success=False, error_message=f"Azure services not properly configured: {validation}")
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
        progress_tracker.start_step("Data Loading", {"directory": "data/raw"})
        raw_documents = load_raw_data_from_directory()

        if not raw_documents:
            progress_tracker.complete_step("Data Loading", success=False, error_message="No raw documents found")
            print("❌ No raw documents found. Please add markdown files to data/raw/")
            return 1

        progress_tracker.update_step_progress("Data Loading", {"documents_loaded": len(raw_documents)})
        progress_tracker.complete_step("Data Loading", success=True)
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
        progress_tracker.start_step("Blob Storage", {"container": f"rag-data-{domain}", "documents": len(raw_documents)})
        container_name = f"rag-data-{domain}"
        await rag_storage.create_container(container_name)

        uploaded_count = 0
        for doc in raw_documents:
            blob_name = f"{doc['filename']}"
            await rag_storage.upload_text(container_name, blob_name, doc['content'])
            uploaded_count += 1
            progress_tracker.update_step_progress("Blob Storage", {"uploaded": uploaded_count, "total": len(raw_documents)})

        progress_tracker.complete_step("Blob Storage", success=True)

        # Step 2: Extract knowledge using Azure OpenAI Enterprise Service
        progress_tracker.start_step("Knowledge Extraction", {"documents": len(raw_documents)})

        # Import enterprise knowledge extraction service
        from core.azure_openai.knowledge_extractor import AzureOpenAIKnowledgeExtractor

        # Initialize enterprise knowledge extractor
        knowledge_extractor = AzureOpenAIKnowledgeExtractor(domain)

        # Use intelligent document processor to chunk documents
        from core.utilities.intelligent_document_processor import UniversalDocumentProcessor

        doc_processor = UniversalDocumentProcessor(
            azure_openai_client=openai_integration,
            max_chunk_size=2000,  # Optimal size for knowledge extraction
            overlap_size=200
        )

        # Process all documents into intelligent chunks
        progress_tracker.update_step_progress("Knowledge Extraction", {"status": "Processing documents with intelligent chunking"})
        all_chunks = []
        for doc in raw_documents:
            chunks = await doc_processor.process_document(doc)
            all_chunks.extend(chunks)
            progress_tracker.update_step_progress("Knowledge Extraction", {
                "documents_processed": len(all_chunks),
                "total_documents": len(raw_documents),
                "chunks_created": len(all_chunks)
            })

        progress_tracker.update_step_progress("Knowledge Extraction", {
            "total_chunks": len(all_chunks),
            "status": "Extracting knowledge from chunks"
        })

        # Extract knowledge from intelligent chunks
        chunk_texts = [chunk.content for chunk in all_chunks]
        extraction_results = await knowledge_extractor.extract_knowledge_from_texts(chunk_texts)

        # Store extracted knowledge in Azure Cosmos DB
        entities_count = len(extraction_results.get('entities', []))
        relations_count = len(extraction_results.get('relations', []))

        progress_tracker.update_step_progress("Knowledge Extraction", {
            "entities_extracted": entities_count,
            "relations_extracted": relations_count,
            "status": "Storing entities and relations"
        })

        # Store entities in Cosmos DB
        stored_entities = 0
        for entity in extraction_results.get('entities', []):
            try:
                # Handle both entity objects and dictionaries
                entity_data = entity.to_dict() if hasattr(entity, 'to_dict') else entity
                await cosmos_client.add_entity(entity_data, domain)
                stored_entities += 1
                progress_tracker.update_step_progress("Knowledge Extraction", {
                    "entities_stored": stored_entities,
                    "entities_extracted": entities_count,
                    "relations_extracted": relations_count
                })
            except Exception as e:
                entity_name = getattr(entity, 'name', str(entity))
                print(f"   ⚠️ Failed to store entity {entity_name}: {e}")

        # Store relations in Cosmos DB
        stored_relations = 0
        for relation in extraction_results.get('relations', []):
            try:
                # Handle both relation objects and dictionaries
                relation_data = relation.to_dict() if hasattr(relation, 'to_dict') else relation
                await cosmos_client.add_relation(relation_data, domain)
                stored_relations += 1
                progress_tracker.update_step_progress("Knowledge Extraction", {
                    "entities_stored": stored_entities,
                    "relations_stored": stored_relations,
                    "entities_extracted": entities_count,
                    "relations_extracted": relations_count
                })
            except Exception as e:
                relation_type = getattr(relation, 'relation_type', str(relation))
                print(f"   ⚠️ Failed to store relation {relation_type}: {e}")

        progress_tracker.complete_step("Knowledge Extraction", success=True)
        print(f"   🔍 Extracted entities: {entities_count}")
        print(f"   🔗 Extracted relations: {relations_count}")

        # Use extraction results for processed documents
        processed_docs = extraction_results

        # Step 3: Build search index with Azure Cognitive Search
        progress_tracker.start_step("Search Indexing", {"chunks": len(all_chunks), "index": f"rag-index-{domain}"})
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

        # Index all intelligent chunks for better search
        progress_tracker.update_step_progress("Search Indexing", {"status": f"Indexing {len(all_chunks)} intelligent chunks"})

        for i, chunk in enumerate(all_chunks):
            # Create search document from chunk
            document = {
                "id": f"chunk_{i}_{chunk.source_info['filename'].replace('.md', '').replace('.txt', '')}_{chunk.chunk_index}",
                "content": chunk.content,
                "title": f"{chunk.source_info['filename']} - Part {chunk.chunk_index + 1}",
                "domain": domain,
                "source": f"data/raw/{chunk.source_info['filename']}",
                "chunk_type": chunk.chunk_type,
                "chunk_index": chunk.chunk_index,
                "metadata": json.dumps({
                    "original_filename": chunk.source_info['filename'],
                    "chunk_index": chunk.chunk_index,
                    "chunk_type": chunk.chunk_type,
                    "total_chunks": chunk.metadata['total_chunks'],
                    "processing_method": chunk.metadata['processing_method'],
                    "file_size": chunk.source_info['size'],
                    "processing_timestamp": datetime.now().isoformat(),
                    "content_type": _detect_content_type(chunk.source_info['filename']),
                    "azure_openai_processed": True,
                    "intelligent_chunking": True
                })
            }

            # Index document directly (validation handled by Azure Search service)
            try:
                index_result = await search_client.index_document(index_name, document)

                if not index_result['success']:
                    print(f"❌ Failed to index chunk {document['id']}: {index_result.get('error', 'Unknown indexing error')}")
                    logger.error(f"Failed to index chunk {document['id']}: {index_result.get('error', 'Unknown indexing error')}")
                else:
                    success_count += 1
                    progress_tracker.update_step_progress("Search Indexing", {
                        "indexed": success_count,
                        "total": len(all_chunks),
                        "progress": f"{success_count}/{len(all_chunks)}"
                    })
            except Exception as e:
                print(f"❌ Chunk processing error for {document['id']}: {str(e)}")
                logger.error(f"Chunk processing error for {document['id']}: {str(e)}", exc_info=True)

        progress_tracker.complete_step("Search Indexing", success=True)

        # Enterprise service integration telemetry
        progress_tracker.start_step("Validation", {
            "documents": len(raw_documents),
            "chunks": len(all_chunks),
            "indexed": success_count,
            "entities": len(extraction_results.get('entities', [])),
            "relations": len(extraction_results.get('relations', []))
        })

        print(f"\n📊 Azure Service Integration Summary:")
        print(f"   🔍 Search Index: {index_name}")
        print(f"   📄 Original documents: {len(raw_documents)}")
        print(f"   🧠 Intelligent chunks created: {len(all_chunks)}")
        print(f"   ✅ Chunks indexed: {success_count}/{len(all_chunks)}")
        print(f"   🔍 Extracted entities: {len(extraction_results.get('entities', []))}")
        print(f"   🔗 Extracted relations: {len(extraction_results.get('relations', []))}")
        print(f"   ⚡ Indexing efficiency: {(success_count/len(all_chunks)*100):.1f}%")

        if success_count == 0:
            progress_tracker.complete_step("Validation", success=False, error_message="Zero documents indexed")
            print(f"\n❌ Critical: Zero documents indexed in Azure Search")
            print(f"🔧 Enterprise troubleshooting required:")
            print(f"   1. Validate Azure Search service configuration")
            print(f"   2. Check document schema compatibility")
            print(f"   3. Verify Azure OpenAI processing output format")
            return 1

        progress_tracker.complete_step("Validation", success=True)

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

        progress_tracker.finish_workflow(success=True)

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
        if 'progress_tracker' in locals():
            progress_tracker.finish_workflow(success=False)
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