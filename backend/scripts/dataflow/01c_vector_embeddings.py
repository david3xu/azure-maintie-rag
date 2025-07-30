#!/usr/bin/env python3
"""
Vector Embeddings - Step 01c  
Add vector embeddings to Azure Search index for semantic search

This script adds 1536-dimensional embeddings to the search index:
- Generates embeddings using Azure OpenAI
- Updates index schema with vector fields  
- Re-indexes all documents with embeddings
"""

import sys
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any, List
import logging

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.infrastructure_service import InfrastructureService
from config.domain_patterns import DomainPatternManager
from config.settings import azure_settings

logger = logging.getLogger(__name__)

class VectorEmbeddingStage:
    """Step 01c: Add Vector Embeddings to Search Index"""
    
    def __init__(self):
        self.infrastructure = InfrastructureService()
        
    async def execute(self, domain: str = "maintenance") -> Dict[str, Any]:
        """
        Execute vector embedding generation and indexing
        
        Args:
            domain: Domain for processing
            
        Returns:
            Dict with embedding results
        """
        print("🎯 Step 01c: Vector Embeddings Generation")
        print("=" * 45)
        
        start_time = asyncio.get_event_loop().time()
        
        stage_results = {
            "stage": "01c_vector_embeddings", 
            "domain": domain,
            "documents_processed": 0,
            "embeddings_generated": 0,
            "success": False
        }
        
        try:
            search_service = self.infrastructure.search_service
            vector_service = self.infrastructure.vector_service
            
            if not search_service:
                raise RuntimeError("❌ Azure Search service not initialized")
            if not vector_service:
                raise RuntimeError("❌ Vector service not initialized")
            
            index_name = DomainPatternManager.get_index_name(domain, azure_settings.azure_search_index)
            stage_results["index_name"] = index_name
            
            print(f"🔍 Processing index: {index_name}")
            
            # Step 1: Get existing documents from specific index
            print("📄 Retrieving existing documents...")
            from azure.search.documents import SearchClient
            from azure.identity import DefaultAzureCredential
            
            credential = DefaultAzureCredential()
            search_client = SearchClient(
                endpoint=azure_settings.azure_search_endpoint,
                index_name=index_name,
                credential=credential
            )
            
            # Get all documents
            documents = []
            results = search_client.search("*", top=1000)
            for result in results:
                documents.append(dict(result))
            
            print(f"📄 Found {len(documents)} documents to process")
            
            # Step 2: Create new index with vector fields
            print("🏗️  Creating index with vector fields...")
            await self._create_vector_index(search_service, index_name, domain)
            
            # Step 3: Generate embeddings and re-index
            print("🎯 Generating embeddings...")
            processed_docs = []
            
            for i, doc in enumerate(documents):
                try:
                    content = doc.get('content', '')
                    if not content:
                        continue
                    
                    print(f"🎯 Processing document {i+1}/{len(documents)}: {doc.get('id', 'unknown')[:20]}...")
                    
                    # Generate embedding using Vector Service
                    embedding_result = await vector_service.embedding_client.generate_embedding(content)
                    
                    if embedding_result.get('success'):
                        embedding = embedding_result['data']['embedding']
                    else:
                        embedding = None
                    
                    if embedding:
                        # Add embedding to document
                        doc['content_vector'] = embedding
                        processed_docs.append(doc)
                        stage_results["embeddings_generated"] += 1
                        
                        # Process in batches
                        if len(processed_docs) >= 10:
                            await self._index_batch_with_vectors(search_service, processed_docs, index_name)
                            stage_results["documents_processed"] += len(processed_docs)
                            print(f"✅ Batch indexed: {len(processed_docs)} documents")
                            processed_docs = []
                    
                except Exception as e:
                    logger.error(f"Failed to process document {i}: {e}")
                    continue
            
            # Index remaining documents
            if processed_docs:
                await self._index_batch_with_vectors(search_service, processed_docs, index_name)
                stage_results["documents_processed"] += len(processed_docs)
                print(f"✅ Final batch indexed: {len(processed_docs)} documents")
            
            # Results
            duration = asyncio.get_event_loop().time() - start_time
            stage_results.update({
                "duration_seconds": round(duration, 2),
                "success": stage_results["embeddings_generated"] > 0
            })
            
            print(f"\n✅ Step 01c Complete:")
            print(f"   🎯 Embeddings generated: {stage_results['embeddings_generated']}")
            print(f"   📄 Documents processed: {stage_results['documents_processed']}")
            print(f"   ⏱️  Duration: {stage_results['duration_seconds']}s")
            
            if stage_results['success']:
                print(f"🎉 Vector embeddings added successfully!")
                print(f"🔍 Index now supports semantic search with 1536D vectors")
            else:
                print(f"⚠️  No embeddings generated")
            
            return stage_results
            
        except Exception as e:
            stage_results["error"] = str(e)
            stage_results["duration_seconds"] = round(asyncio.get_event_loop().time() - start_time, 2)
            print(f"❌ Step 01c Failed: {e}")
            logger.error(f"Vector embedding failed: {e}", exc_info=True)
            return stage_results
    
    async def _create_vector_index(self, search_service, index_name: str, domain: str):
        """Create or update index with vector fields"""
        try:
            from azure.search.documents.indexes import SearchIndexClient
            from azure.search.documents.indexes.models import (
                SearchIndex, SimpleField, SearchableField, VectorSearch,
                SearchField, SearchFieldDataType, VectorSearchProfile,
                HnswAlgorithmConfiguration, VectorSearchAlgorithmKind
            )
            from azure.identity import DefaultAzureCredential
            
            credential = DefaultAzureCredential()
            index_client = SearchIndexClient(
                endpoint=azure_settings.azure_search_endpoint,
                credential=credential
            )
            
            # Define fields including vector field
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="content", type=SearchFieldDataType.String, searchable=True),
                SearchableField(name="title", type=SearchFieldDataType.String, searchable=True),
                SimpleField(name="category", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="domain", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="metadata", type=SearchFieldDataType.String),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=1536,
                    vector_search_profile_name="vector-profile"
                )
            ]
            
            # Configure vector search
            vector_search = VectorSearch(
                profiles=[VectorSearchProfile(
                    name="vector-profile",
                    algorithm_configuration_name="hnsw-config"
                )],
                algorithms=[HnswAlgorithmConfiguration(
                    name="hnsw-config",
                    kind=VectorSearchAlgorithmKind.HNSW
                )]
            )
            
            # Create index
            index = SearchIndex(
                name=index_name,
                fields=fields,
                vector_search=vector_search
            )
            
            # Delete and recreate index to add vector fields
            try:
                index_client.delete_index(index_name)
                print(f"🗑️  Deleted existing index: {index_name}")
            except:
                pass
            
            index_client.create_index(index)
            print(f"✅ Created vector-enabled index: {index_name}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to create vector index: {e}")
    
    async def _index_batch_with_vectors(self, search_service, documents: List[Dict], index_name: str):
        """Index batch of documents with vector embeddings"""
        try:
            result = await search_service.index_documents(documents, index_name)
            if not result.get('success'):
                raise RuntimeError(f"Batch indexing failed: {result.get('error')}")
        except Exception as e:
            logger.error(f"Failed to index batch: {e}")
            raise e


async def main():
    """Main entry point for vector embedding generation"""
    parser = argparse.ArgumentParser(
        description="Step 01c: Vector Embeddings Generation"
    )
    parser.add_argument(
        "--domain", 
        default="maintenance",
        help="Domain for processing"
    )
    
    args = parser.parse_args()
    
    # Execute stage
    stage = VectorEmbeddingStage()
    results = await stage.execute(domain=args.domain)
    
    # Return appropriate exit code
    return 0 if results.get("success") else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))