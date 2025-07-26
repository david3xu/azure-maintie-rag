#!/usr/bin/env python3
"""
Data Upload and Chunking Workflow Script
========================================

STEP 1 of Azure Universal RAG Pipeline
Handles document upload and intelligent chunking using Azure services.

Azure Services Used:
- Azure Blob Storage (store documents and chunks)
- Azure OpenAI (intelligent document analysis and chunking)
- Azure Cognitive Search (create search indices for chunks)
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
from core.utilities.intelligent_document_processor import UniversalDocumentProcessor

azure_settings = AzureSettings()

def _detect_content_type(filename: str) -> str:
    """Detect content type from filename extension"""
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
    """Execute data upload and chunking workflow with Azure services"""
    
    # Initialize workflow tracking
    start_time = time.time()
    domain = sys.argv[1] if len(sys.argv) > 1 else "general"
    
    print(f"🔄 STEP 1: Data Upload & Intelligent Chunking")
    print(f"============================================================")
    print(f"📊 Purpose: Upload documents and create intelligent chunks using Azure")
    print(f"☁️  Azure Services: Blob Storage, OpenAI, Cognitive Search")
    print(f"⏱️  Workflow: Document upload → Intelligent chunking → Search indexing")
    print()

    try:
        # Load raw data from data/raw directory
        print(f"📂 Loading raw documents from data/raw directory...")
        raw_documents = load_raw_data_from_directory()

        if not raw_documents:
            print("❌ No raw documents found. Please add markdown files to data/raw/")
            return 1

        print(f"✅ Loaded {len(raw_documents)} documents from data/raw/")

        # Initialize Azure services
        print(f"\n📝 Initializing Azure services...")
        azure_services = AzureServicesManager()

        # Validate services
        print(f"\n📝 Validating Azure services configuration...")
        validation = azure_services.validate_configuration()
        if not validation['all_configured']:
            raise RuntimeError(f"Azure services not properly configured: {validation}")

        openai_integration = AzureOpenAIClient()

        # Use get_rag_storage_client for storage
        rag_storage = azure_services.get_rag_storage_client()
        search_client = azure_services.get_service('search')

        # Step 1: Store original documents in Azure Blob Storage
        print(f"\n☁️  Step 1.1: Storing original documents in Azure Blob Storage...")
        original_container = f"raw-documents-{domain}"
        await rag_storage.create_container(original_container)

        for doc in raw_documents:
            blob_name = f"original/{doc['filename']}"
            await rag_storage.upload_text(original_container, blob_name, doc['content'])
            print(f"   📄 Uploaded: {doc['filename']}")

        # Step 2: Intelligent document chunking
        print(f"\n🧠 Step 1.2: Intelligent document chunking with Azure OpenAI...")
        
        doc_processor = UniversalDocumentProcessor(
            azure_openai_client=openai_integration,
            max_chunk_size=2000,  # Optimal size for knowledge extraction
            overlap_size=200
        )
        
        # Process all documents into intelligent chunks
        all_chunks = []
        total_chunks_created = 0
        
        for doc in raw_documents:
            print(f"   🔍 Processing: {doc['filename']}")
            chunks = await doc_processor.process_document(doc)
            all_chunks.extend(chunks)
            total_chunks_created += len(chunks)
            print(f"   ✅ Created {len(chunks)} intelligent chunks from {doc['filename']}")
        
        print(f"\n📊 Chunking Summary:")
        print(f"   📄 Original documents: {len(raw_documents)}")
        print(f"   🧩 Total chunks created: {total_chunks_created}")
        print(f"   📈 Average chunks per document: {total_chunks_created/len(raw_documents):.1f}")

        # Step 3: Store chunks in Azure Blob Storage
        print(f"\n☁️  Step 1.3: Storing chunks in Azure Blob Storage...")
        chunks_container = f"processed-chunks-{domain}"
        await rag_storage.create_container(chunks_container)

        chunks_metadata = []
        for i, chunk in enumerate(all_chunks):
            # Store chunk content
            chunk_blob_name = f"chunk_{i:04d}_{chunk.source_info['filename']}_{chunk.chunk_index}.txt"
            await rag_storage.upload_text(chunks_container, chunk_blob_name, chunk.content)
            
            # Store chunk metadata
            chunk_metadata = {
                "chunk_id": f"chunk_{i:04d}",
                "blob_name": chunk_blob_name,
                "source_file": chunk.source_info['filename'],
                "chunk_index": chunk.chunk_index,
                "chunk_type": chunk.chunk_type,
                "content_length": len(chunk.content),
                "metadata": chunk.metadata,
                "processing_timestamp": datetime.now().isoformat()
            }
            chunks_metadata.append(chunk_metadata)
            
            if i % 20 == 0:
                print(f"   📦 Stored chunk {i+1}/{len(all_chunks)}")

        # Store chunks index file
        chunks_index = {
            "domain": domain,
            "total_chunks": len(all_chunks),
            "original_documents": len(raw_documents),
            "created_at": datetime.now().isoformat(),
            "chunks": chunks_metadata
        }
        
        index_blob_name = f"chunks_index_{domain}.json"
        await rag_storage.upload_text(chunks_container, index_blob_name, json.dumps(chunks_index, indent=2))
        print(f"   📋 Stored chunks index: {index_blob_name}")

        # Step 4: Create search index for chunks (without chunk_type field to avoid schema issues)
        print(f"\n🔍 Step 1.4: Creating search index for chunks...")
        index_name = f"chunks-index-{domain}"
        await search_client.create_index(index_name)

        success_count = 0
        for i, chunk in enumerate(all_chunks):
            # Create search document from chunk (excluding problematic fields)
            document = {
                "id": f"chunk_{i:04d}_{chunk.source_info['filename'].replace('.md', '').replace('.txt', '')}_{chunk.chunk_index}",
                "content": chunk.content,
                "title": f"{chunk.source_info['filename']} - Part {chunk.chunk_index + 1}",
                "domain": domain,
                "source": f"data/raw/{chunk.source_info['filename']}",
                "metadata": json.dumps({
                    "original_filename": chunk.source_info['filename'],
                    "chunk_index": chunk.chunk_index,
                    "chunk_type": chunk.chunk_type,
                    "total_chunks": chunk.metadata['total_chunks'],
                    "processing_method": chunk.metadata['processing_method'],
                    "file_size": chunk.source_info['size'],
                    "processing_timestamp": datetime.now().isoformat(),
                    "content_type": _detect_content_type(chunk.source_info['filename']),
                    "intelligent_chunking": True,
                    "workflow_step": "upload_and_chunking"
                })
            }

            try:
                index_result = await search_client.index_document(index_name, document)
                if index_result['success']:
                    success_count += 1
                    if i % 50 == 0:
                        print(f"   ✅ Indexed chunk {i+1}/{len(all_chunks)}")
                else:
                    logger.error(f"Failed to index chunk {document['id']}: {index_result.get('error', 'Unknown error')}")
            except Exception as e:
                logger.error(f"Error indexing chunk {document['id']}: {str(e)}")

        # Final summary
        duration = time.time() - start_time
        print(f"\n📊 STEP 1 Completion Summary:")
        print(f"   📄 Documents processed: {len(raw_documents)}")
        print(f"   🧩 Chunks created: {len(all_chunks)}")
        print(f"   ☁️  Chunks stored in Azure: {len(all_chunks)}")
        print(f"   🔍 Chunks indexed: {success_count}/{len(all_chunks)}")
        print(f"   ⚡ Success rate: {(success_count/len(all_chunks)*100):.1f}%")
        print(f"   ⏱️  Total duration: {duration:.1f} seconds")
        print()
        print(f"✅ STEP 1 COMPLETED: Data upload and chunking ready for knowledge extraction")
        print(f"🔄 Next step: Run 'make knowledge-extract' to extract entities and relations")

        if success_count == 0:
            print(f"\n❌ Critical: Zero chunks indexed in Azure Search")
            return 1

        return 0

    except Exception as e:
        print(f"\n❌ Critical error in data upload workflow: {str(e)}")
        logger.error(f"Data upload workflow failed: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)