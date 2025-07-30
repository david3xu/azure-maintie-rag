#!/usr/bin/env python3
"""
Check Azure Search Index Schema
Verify if vector embeddings are included in the index
"""

import sys
import asyncio
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.infrastructure_service import InfrastructureService
from config.domain_patterns import DomainPatternManager
from config.settings import azure_settings

async def check_index_schema(domain="maintenance"):
    """Check the current search index schema"""
    
    infrastructure = InfrastructureService()
    search_service = infrastructure.search_service
    
    if not search_service:
        print("❌ Search service not initialized")
        return
    
    index_name = DomainPatternManager.get_index_name(domain, azure_settings.azure_search_index)
    
    print(f"🔍 Checking Index Schema: {index_name}")
    print("=" * 50)
    
    try:
        # Get index definition
        from azure.search.documents.indexes import SearchIndexClient
        from azure.identity import DefaultAzureCredential
        
        credential = DefaultAzureCredential()
        index_client = SearchIndexClient(
            endpoint=azure_settings.azure_search_endpoint,
            credential=credential
        )
        
        # Get the index
        index = index_client.get_index(index_name)
        
        print(f"📊 Index Name: {index.name}")
        print(f"📄 Total Fields: {len(index.fields)}")
        print("\n🏗️  Index Fields:")
        print("-" * 30)
        
        vector_fields = []
        text_fields = []
        
        for field in index.fields:
            field_type = str(field.type).split('.')[-1] if hasattr(field, 'type') else 'Unknown'
            
            print(f"• {field.name}")
            print(f"  Type: {field_type}")
            print(f"  Searchable: {getattr(field, 'searchable', False)}")
            print(f"  Retrievable: {getattr(field, 'retrievable', False)}")
            
            # Check for vector fields
            if 'vector' in field.name.lower() or field_type == 'Collection(Edm.Single)':
                vector_fields.append(field.name)
                print(f"  🎯 VECTOR FIELD!")
            else:
                text_fields.append(field.name)
            
            print()
        
        print("📈 Summary:")
        print(f"   Text fields: {len(text_fields)}")
        print(f"   Vector fields: {len(vector_fields)}")
        
        if vector_fields:
            print(f"✅ Vector embeddings found: {vector_fields}")
        else:
            print("❌ NO VECTOR EMBEDDINGS FOUND!")
            print("   Index only contains text fields - no semantic search capability")
            
    except Exception as e:
        print(f"❌ Error checking schema: {e}")

if __name__ == "__main__":
    asyncio.run(check_index_schema())