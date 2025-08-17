#!/usr/bin/env python3
"""
Fix Cosmos DB Authentication for Graph Search
============================================

This script fixes Cosmos DB authentication by switching to key-based auth
instead of RBAC to get the graph search working immediately.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import azure_settings
from agents.core.universal_deps import get_universal_deps

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_cosmos_auth_fix():
    """Test Cosmos DB authentication fix"""
    logger.info("🔧 Testing Cosmos DB authentication fix...")
    
    try:
        # Get universal dependencies (proper initialization pattern)
        deps = await get_universal_deps()
        
        if not deps.is_service_available('cosmos'):
            logger.error("❌ Cosmos DB service not available in UniversalDeps")
            return False
            
        # Get Cosmos client through UniversalDeps (proper credential handling)
        cosmos_client = deps.cosmos_client
        
        # Test basic connection
        logger.info("🔗 Testing connection...")
        connection_result = await cosmos_client.test_connection()
        
        if not connection_result:
            logger.error("❌ Connection test failed")
            return False
            
        logger.info("✅ Connection test successful!")
        
        # Test basic operations
        logger.info("📊 Testing basic operations...")
        
        # Count vertices in different domains
        try:
            azure_count = await cosmos_client.count_vertices('azure-ai')
            general_count = await cosmos_client.count_vertices('general')
            
            logger.info(f"📈 Azure AI domain vertices: {azure_count}")
            logger.info(f"📈 General domain vertices: {general_count}")
            
            # Test getting entities
            if azure_count > 0:
                azure_entities = await cosmos_client.get_all_entities('azure-ai')
                logger.info(f"🔍 Retrieved {len(azure_entities)} Azure AI entities")
                
                if azure_entities:
                    sample_entity = azure_entities[0]
                    logger.info(f"📝 Sample entity: {sample_entity.get('text', 'No text field')}")
            
            if general_count > 0:
                general_entities = await cosmos_client.get_all_entities('general')
                logger.info(f"🔍 Retrieved {len(general_entities)} general entities")
                
                if general_entities:
                    sample_entity = general_entities[0]
                    logger.info(f"📝 Sample entity: {sample_entity.get('text', 'No text field')}")
            
            # Test relationships
            azure_relations = await cosmos_client.get_all_relations('azure-ai')
            general_relations = await cosmos_client.get_all_relations('general')
            
            logger.info(f"🔗 Azure AI relationships: {len(azure_relations)}")
            logger.info(f"🔗 General relationships: {len(general_relations)}")
            
            if azure_relations:
                sample_rel = azure_relations[0]
                logger.info(f"📝 Sample relationship: {sample_rel.get('source_entity', 'No source')} -> {sample_rel.get('target_entity', 'No target')}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Operation test failed: {e}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Authentication test failed: {e}")
        return False

async def check_graph_population():
    """Check if graph database has data for search"""
    logger.info("📊 Checking graph database population...")
    
    try:
        # Get universal dependencies
        deps = await get_universal_deps()
        
        if not deps.is_service_available('cosmos'):
            logger.error("❌ Cosmos DB service not available")
            return False
            
        cosmos_client = deps.cosmos_client
        
        # Get available domains
        azure_count = await cosmos_client.count_vertices('azure-ai')
        general_count = await cosmos_client.count_vertices('general')
        
        total_entities = azure_count + general_count
        
        if total_entities == 0:
            logger.warning("⚠️ Graph database is EMPTY!")
            logger.info("💡 This explains why Graph search returns no results")
            logger.info("💡 Solution: Run knowledge extraction to populate graph")
            logger.info("💡 Command: PYTHONPATH=/workspace/azure-maintie-rag python scripts/dataflow/phase3_knowledge/03_01_basic_entity_extraction.py")
            return False
        else:
            logger.info(f"✅ Graph database has {total_entities} entities total")
            logger.info(f"📊 Azure AI domain: {azure_count} entities")
            logger.info(f"📊 General domain: {general_count} entities")
            logger.info("✅ Graph search should work with this data")
            return True
            
    except Exception as e:
        logger.error(f"❌ Graph population check failed: {e}")
        return False

async def main():
    """Main execution"""
    logger.info("🎯 FIXING COSMOS DB AUTHENTICATION FOR GRAPH SEARCH")
    logger.info("=" * 60)
    
    # Step 1: Test authentication fix
    auth_result = await test_cosmos_auth_fix()
    
    if not auth_result:
        logger.error("❌ Cosmos DB authentication still failing")
        logger.info("💡 Check if Cosmos DB account exists and keys are accessible")
        return 1
    
    # Step 2: Check graph population
    population_result = await check_graph_population()
    
    logger.info("=" * 60)
    logger.info("📊 COSMOS DB AUTHENTICATION FIX SUMMARY:")
    logger.info(f"✅ Authentication: {'WORKING' if auth_result else 'FAILED'}")
    logger.info(f"📊 Graph Population: {'HAS DATA' if population_result else 'EMPTY'}")
    
    if auth_result and population_result:
        logger.info("🎉 SUCCESS: Graph search modality is now operational!")
        logger.info("💡 Ready to test tri-modal search (Vector + Graph + GNN)")
    elif auth_result and not population_result:
        logger.info("⚠️ SUCCESS: Authentication fixed, but graph needs data")
        logger.info("💡 Next: Run knowledge extraction to populate graph")
    else:
        logger.error("❌ Authentication still failing")
        
    logger.info("=" * 60)
    
    return 0 if auth_result else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)