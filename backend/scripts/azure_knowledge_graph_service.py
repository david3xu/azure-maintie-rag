#!/usr/bin/env python3
"""
Azure Knowledge Graph Service Orchestrator
Based on real azure_services.migrate_data_to_azure implementation
"""

import asyncio
import sys
from integrations.azure_services import AzureServicesManager

async def orchestrate_knowledge_graph_population():
    """Execute Azure knowledge graph population service"""
    domain = sys.argv[1] if len(sys.argv) > 1 else "general"
    source_data_path = "data/raw"
    print(f"🚀 Azure Knowledge Graph Service: {domain}")
    print(f"📁 Source: {source_data_path}")
    # Initialize Azure services manager (existing implementation)
    azure_services = AzureServicesManager()
    # Service context based on real codebase pattern
    migration_context = {
        "migration_id": f"azure_kg_extraction_{domain}",
        "source": "universal_rag_migration",
        "timestamp": "auto-generated"
    }
    # Execute using existing azure_services.migrate_data_to_azure
    result = await azure_services.migrate_data_to_azure(
        source_data_path=source_data_path,
        domain=domain
    )
    # Azure service result validation
    if result.get("success"):
        print(f"✅ Azure Cosmos DB population completed")
        # FIX: Access correct result structure
        cosmos_result = result.get("migration_results", {}).get("cosmos_migration", {})
        cosmos_details = cosmos_result.get("details", {})
        entities_count = cosmos_details.get("entities_migrated", 0)
        relations_count = cosmos_details.get("relations_migrated", 0)
        print(f"📊 Entities: {entities_count}")
        print(f"📊 Relations: {relations_count}")
        # Additional extraction summary if available
        extraction_summary = cosmos_details.get("extraction_summary", {})
        if extraction_summary:
            print(f"📈 Entity Types: {len(extraction_summary.get('unique_entity_types', []))}")
            print(f"📈 Relation Types: {len(extraction_summary.get('unique_relation_types', []))}")
        return 0
    else:
        print(f"❌ Azure service error: {result.get('error')}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(orchestrate_knowledge_graph_population())
    sys.exit(exit_code)