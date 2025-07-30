#!/usr/bin/env python3
"""
Quick fix for the knowledge gap - use existing working Cosmos client
"""
import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient

def quick_fix_knowledge_gap():
    """Quick fix using existing proven Cosmos client"""
    print("🚀 QUICK KNOWLEDGE GAP FIX")
    print("=" * 50)
    print("Using existing proven AzureCosmosGremlinClient")
    
    # Load extraction data
    extraction_file = "/workspace/azure-maintie-rag/backend/data/outputs/complete_dataflow_20250730_044432/step02_knowledge_extraction.json"
    
    with open(extraction_file, 'r') as f:
        data = json.load(f)
    
    knowledge_data = data.get('knowledge_data', {})
    entities = knowledge_data.get('entities', [])
    relationships = knowledge_data.get('relationships', [])
    
    print(f"📊 Found {len(entities)} entities and {len(relationships)} relationships")
    
    # Initialize proven Cosmos client
    cosmos_config = {
        'endpoint': 'https://cosmos-maintie-rag-staging-oeeopj3ksgnlo.documents.azure.com:443/',
        'database': 'maintie-rag-staging',
        'container': 'knowledge-graph-staging'
    }
    
    try:
        cosmos_client = AzureCosmosGremlinClient(cosmos_config)
        
        # Test connection
        health = cosmos_client.health_check()
        if health.get('status') != 'healthy':
            print(f"❌ Cosmos DB unhealthy: {health}")
            return False
        
        print("✅ Cosmos DB client connected and healthy")
        
        # Get initial stats
        initial_stats = cosmos_client.get_graph_statistics('maintenance')
        print(f"📊 Current graph: {initial_stats.get('vertex_count', 0)} vertices, {initial_stats.get('edge_count', 0)} edges")
        
        # Store entities using existing proven method (limit to first 50 to test)
        print(f"\n📦 Storing entities using proven add_entity method...")
        entities_stored = 0
        
        for i, entity in enumerate(entities[:50]):  # Test with first 50
            try:
                entity_data = {
                    'text': entity.get('text', ''),
                    'entity_type': entity.get('type', 'unknown'),
                    'context': entity.get('context', ''),
                    'confidence': 1.0,
                    'source': 'knowledge_extraction',
                    'created_at': datetime.now().isoformat()
                }
                
                result = cosmos_client.add_entity(entity_data, 'maintenance')
                
                if result.get('success', False):
                    entities_stored += 1
                    if entities_stored % 10 == 0:
                        print(f"   📦 Stored {entities_stored} entities...")
                else:
                    print(f"   ⚠️ Failed to store entity: {entity.get('text', 'N/A')}")
                    
            except Exception as e:
                print(f"   ❌ Error storing entity {i}: {e}")
        
        print(f"✅ Stored {entities_stored}/50 entities successfully")
        
        # Store relationships using existing proven method (limit to first 20)
        print(f"\n🔗 Storing relationships using proven add_relationship method...")
        relationships_stored = 0
        
        for i, relationship in enumerate(relationships[:20]):  # Test with first 20
            try:
                relation_data = {
                    'source_entity': relationship.get('source', ''),
                    'target_entity': relationship.get('target', ''),
                    'relation_type': relationship.get('relation', 'related'),
                    'context': relationship.get('context', ''),
                    'confidence': 1.0,
                    'source': 'knowledge_extraction',
                    'created_at': datetime.now().isoformat()
                }
                
                result = cosmos_client.add_relationship(relation_data, 'maintenance')
                
                if result.get('success', False):
                    relationships_stored += 1
                    if relationships_stored % 5 == 0:
                        print(f"   🔗 Stored {relationships_stored} relationships...")
                else:
                    print(f"   ⚠️ Failed to store relationship: {relationship.get('source', 'N/A')} -> {relationship.get('target', 'N/A')}")
                    
            except Exception as e:
                print(f"   ❌ Error storing relationship {i}: {e}")
        
        print(f"✅ Stored {relationships_stored}/20 relationships successfully")
        
        # Get final stats
        final_stats = cosmos_client.get_graph_statistics('maintenance')
        print(f"\n📊 Final graph: {final_stats.get('vertex_count', 0)} vertices, {final_stats.get('edge_count', 0)} edges")
        
        edges_added = final_stats.get('edge_count', 0) - initial_stats.get('edge_count', 0)
        vertices_added = final_stats.get('vertex_count', 0) - initial_stats.get('vertex_count', 0)
        
        print(f"📈 Added: {vertices_added} vertices, {edges_added} edges")
        
        if edges_added > 0:
            print(f"\n🎉 SUCCESS: KNOWLEDGE GAP PARTIALLY FIXED!")
            print(f"✅ Cosmos DB now has {edges_added} relationships!")
            print(f"🔧 This proves the extraction->storage pipeline works")
            return True
        else:
            print(f"\n⚠️ No relationships were added - need to investigate further")
            return False
            
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        try:
            if 'cosmos_client' in locals():
                cosmos_client.close()
        except:
            pass

if __name__ == "__main__":
    success = quick_fix_knowledge_gap()
    
    if success:
        print(f"\n🏆 KNOWLEDGE GAP QUICK FIX: SUCCESS!")
        print(f"💡 Now we have proven that extracted knowledge CAN be stored in Cosmos DB")
        print(f"🔧 Next step: Scale this up to store all 540 entities and 597 relationships")
    else:
        print(f"\n💥 QUICK FIX FAILED")
        print(f"🔍 Need to investigate why the proven Cosmos client isn't working")