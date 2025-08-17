#!/usr/bin/env python3
"""
Create Missing Search Index for Vector Search
============================================

This script creates the missing 'maintie-prod-index' search index
to fix the tri-modal search system.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticSearch,
    SemanticField
)
from config.settings import azure_settings
from infrastructure.azure_auth.session_manager import AzureSessionManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def create_search_index() -> Dict[str, Any]:
    """Create the missing search index with proper schema"""
    logger.info("🔍 Creating missing search index for vector search...")
    
    try:
        # Get credentials from session manager
        session_manager = AzureSessionManager()
        credential = session_manager.get_credential()
        
        # Initialize search index client
        index_client = SearchIndexClient(
            endpoint=azure_settings.azure_search_endpoint,
            credential=credential
        )
        
        # Define the search index schema
        index_name = azure_settings.azure_search_index
        logger.info(f"📝 Creating index: {index_name}")
        
        # Define fields for the index
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SimpleField(name="file_path", type=SearchFieldDataType.String, filterable=True),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,  # text-embedding-ada-002 dimension
                vector_search_profile_name="hnsw-profile"
            )
        ]
        
        # Configure vector search
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="hnsw-algorithm",
                    parameters={
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine"
                    }
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="hnsw-profile",
                    algorithm_configuration_name="hnsw-algorithm"
                )
            ]
        )
        
        # Configure semantic search for better relevance
        semantic_config = SemanticConfiguration(
            name="semantic-config",
            prioritized_fields={
                "title_field": SemanticField(field_name="title"),
                "content_fields": [SemanticField(field_name="content")],
                "keywords_fields": [SemanticField(field_name="file_path")]
            }
        )
        
        semantic_search = SemanticSearch(configurations=[semantic_config])
        
        # Create the index
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search
        )
        
        # Create or update the index
        logger.info(f"🏗️ Creating search index with vector search capabilities...")
        result = index_client.create_or_update_index(index)
        
        logger.info(f"✅ Search index created successfully!")
        logger.info(f"📊 Index name: {result.name}")
        logger.info(f"📊 Fields count: {len(result.fields)}")
        logger.info(f"📊 Vector search enabled: {result.vector_search is not None}")
        logger.info(f"📊 Semantic search enabled: {result.semantic_search is not None}")
        
        return {
            "status": "success",
            "index_name": result.name,
            "fields_count": len(result.fields),
            "vector_search_enabled": result.vector_search is not None,
            "semantic_search_enabled": result.semantic_search is not None,
            "message": "Search index created successfully for tri-modal search"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to create search index: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "message": "Could not create search index"
        }

async def verify_index_creation() -> Dict[str, Any]:
    """Verify the index was created successfully"""
    logger.info("🔍 Verifying search index creation...")
    
    try:
        # Get credentials from session manager
        session_manager = AzureSessionManager()
        credential = session_manager.get_credential()
        
        # Initialize search index client
        index_client = SearchIndexClient(
            endpoint=azure_settings.azure_search_endpoint,
            credential=credential
        )
        
        index_name = azure_settings.azure_search_index
        
        # Get index statistics
        stats = index_client.get_index_statistics(index_name)
        
        logger.info(f"✅ Index verification successful!")
        logger.info(f"📊 Document count: {stats.document_count}")
        logger.info(f"📊 Storage size: {stats.storage_size}")
        
        return {
            "status": "verified",
            "index_name": index_name,
            "document_count": stats.document_count,
            "storage_size": stats.storage_size,
            "message": "Index is ready for vector search operations"
        }
        
    except Exception as e:
        logger.error(f"❌ Index verification failed: {e}")
        return {
            "status": "verification_failed",
            "error": str(e)
        }

async def main():
    """Main execution"""
    logger.info("🎯 CREATING MISSING SEARCH INDEX FOR TRI-MODAL SEARCH")
    logger.info("=" * 60)
    
    # Step 1: Create the search index
    creation_result = await create_search_index()
    
    if creation_result["status"] != "success":
        logger.error("❌ Index creation failed")
        return 1
    
    # Step 2: Verify the index
    verification_result = await verify_index_creation()
    
    if verification_result["status"] != "verified":
        logger.error("❌ Index verification failed")
        return 1
    
    logger.info("=" * 60)
    logger.info("🎉 SUCCESS: Search index created and verified!")
    logger.info("💡 Vector search modality is now operational")
    logger.info("📝 Next: Fix Cosmos DB authentication for Graph search")
    logger.info("🎯 Goal: Complete tri-modal search (Vector + Graph + GNN)")
    logger.info("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)