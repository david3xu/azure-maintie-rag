#!/usr/bin/env python3
"""
Azure Data Viewer Script
Displays current state of all Azure services with detailed extraction results
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

from integrations.azure_services import AzureServicesManager
from config.settings import settings, azure_settings


async def view_azure_data_state():
    """Display comprehensive Azure data state"""
    print("🔍 Azure Data Extraction Results Viewer")
    print("=" * 50)

    # Initialize Azure services
    azure_services = AzureServicesManager()

    # 1. Azure Blob Storage
    print("\n📦 Azure Blob Storage:")
    try:
        rag_storage = azure_services.get_rag_storage_client()
        container_name = f"rag-data-general"
        blobs = rag_storage.list_blobs(container_name)
        print(f"   Container: {container_name}")
        print(f"   Documents: {len(blobs)}")
        for blob in blobs[:5]:  # Show first 5
            print(f"   📄 {blob}")
        if len(blobs) > 5:
            print(f"   ... and {len(blobs) - 5} more")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # 2. Azure Cognitive Search
    print("\n🔍 Azure Cognitive Search:")
    try:
        search_client = azure_services.get_service('search')
        index_name = "rag-index-general"

        # Get document count using search
        search_results = await search_client.search_documents(index_name, "*", top_k=1)
        total_docs = len(search_results)
        print(f"   Index: {index_name}")
        print(f"   Total Documents: {total_docs}")

        # Show sample documents
        sample_results = await search_client.search_documents(index_name, "*", top_k=3)
        for i, doc in enumerate(sample_results):
            print(f"   📄 Document {i+1}: {doc.get('title', 'Unknown')}")
            print(f"      ID: {doc.get('id', 'Unknown')}")
            print(f"      Type: {doc.get('chunk_type', 'Unknown')}")
            content_preview = doc.get('content', '')[:100]
            print(f"      Content: {content_preview}...")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # 3. Azure Cosmos DB - Entities
    print("\n🏗️ Azure Cosmos DB - Extracted Entities:")
    try:
        cosmos_client = azure_services.get_service('cosmos')
        # Use Gremlin query to get entities
        entities_query = "g.V().hasLabel('entity').has('domain', 'general')"
        entities = cosmos_client._execute_gremlin_query_safe(entities_query)
        print(f"   Total Entities: {len(entities)}")

        # Show entity details
        for i, entity in enumerate(entities[:5]):
            print(f"   🔍 Entity {i+1}: {entity.get('name', 'Unknown')}")
            print(f"      Type: {entity.get('entity_type', 'Unknown')}")
            print(f"      Properties: {len(entity.get('properties', {}))}")
        if len(entities) > 5:
            print(f"   ... and {len(entities) - 5} more entities")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # 4. Azure Cosmos DB - Relations
    print("\n🔗 Azure Cosmos DB - Extracted Relations:")
    try:
        cosmos_client = azure_services.get_service('cosmos')
        # Use Gremlin query to get relations
        relations_query = "g.E().hasLabel('relation').has('domain', 'general')"
        relations = cosmos_client._execute_gremlin_query_safe(relations_query)
        print(f"   Total Relations: {len(relations)}")

        # Show relation details
        for i, relation in enumerate(relations[:5]):
            print(f"   🔗 Relation {i+1}: {relation.get('relation_type', 'Unknown')}")
            print(f"      Source: {relation.get('source_entity', 'Unknown')}")
            print(f"      Target: {relation.get('target_entity', 'Unknown')}")
        if len(relations) > 5:
            print(f"   ... and {len(relations) - 5} more relations")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # 5. Knowledge Graph Summary
    print("\n🧠 Knowledge Extraction Summary:")
    try:
        cosmos_client = azure_services.get_service('cosmos')
        entities_query = "g.V().hasLabel('entity').has('domain', 'general')"
        relations_query = "g.E().hasLabel('relation').has('domain', 'general')"
        entities = cosmos_client._execute_gremlin_query_safe(entities_query)
        relations = cosmos_client._execute_gremlin_query_safe(relations_query)

        # Entity type distribution
        entity_types = {}
        for entity in entities:
            entity_type = entity.get('entity_type', 'Unknown')
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

        print(f"   📊 Entity Types: {len(entity_types)}")
        for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
            print(f"      {entity_type}: {count}")

        # Relation type distribution
        relation_types = {}
        for relation in relations:
            relation_type = relation.get('relation_type', 'Unknown')
            relation_types[relation_type] = relation_types.get(relation_type, 0) + 1

        print(f"   📊 Relation Types: {len(relation_types)}")
        for relation_type, count in sorted(relation_types.items(), key=lambda x: x[1], reverse=True):
            print(f"      {relation_type}: {count}")

    except Exception as e:
        print(f"   ❌ Error: {e}")

    # 6. Azure Portal Links
    print("\n🌐 Azure Portal Direct Links:")
    print(f"   🔍 Search Service: {azure_settings.azure_search_service}.search.windows.net")
    print(f"   🏗️ Cosmos DB Account: {azure_settings.azure_cosmos_account}")
    print(f"   📦 Storage Account: {azure_settings.azure_storage_account}")
    print(f"   📊 Resource Group: {azure_settings.azure_resource_group}")

    print("\n" + "=" * 50)
    print("✅ Azure Data Viewer completed!")


if __name__ == "__main__":
    asyncio.run(view_azure_data_state())