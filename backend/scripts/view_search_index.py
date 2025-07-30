#!/usr/bin/env python3
"""
View Azure Cognitive Search Index Data
Quick script to visualize indexed documents from Step 01b
"""

import sys
import asyncio
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.infrastructure_service import InfrastructureService
from config.settings import azure_settings

async def view_search_index(domain="maintenance", limit=10):
    """View documents in the search index"""
    
    infrastructure = InfrastructureService()
    search_service = infrastructure.search_service
    
    if not search_service:
        print("❌ Search service not initialized")
        return
    
    # Use the correct index name format
    from config.domain_patterns import DomainPatternManager
    index_name = DomainPatternManager.get_index_name(domain, azure_settings.azure_search_index)
    
    print(f"🔍 Viewing Search Index: {index_name}")
    print("=" * 50)
    
    try:
        # Search for all documents
        result = await search_service.search_documents("*", top=limit)
        
        if result.get('success'):
            documents = result.get('data', {}).get('documents', [])
            count = result.get('data', {}).get('total_count', 0)
            
            print(f"📊 Total documents in index: {count}")
            print(f"📄 Showing first {len(documents)} documents:")
            print("-" * 50)
            
            for i, doc in enumerate(documents, 1):
                print(f"\n{i}. Document ID: {doc.get('id', 'N/A')}")
                print(f"   Title: {doc.get('title', 'N/A')}")
                print(f"   Domain: {doc.get('domain', 'N/A')}")
                print(f"   Type: {doc.get('document_type', 'N/A')}")
                print(f"   Content: {doc.get('content', '')[:100]}...")
                print(f"   Score: {doc.get('@search.score', 'N/A')}")
        else:
            print(f"❌ Search failed: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Error viewing index: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="View Azure Search Index Data")
    parser.add_argument("--domain", default="maintenance", help="Domain to search")
    parser.add_argument("--limit", type=int, default=10, help="Number of documents to show")
    parser.add_argument("--query", default="*", help="Search query")
    
    args = parser.parse_args()
    
    asyncio.run(view_search_index(args.domain, args.limit))