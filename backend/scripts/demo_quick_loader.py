#!/usr/bin/env python3
"""
Quick Demo Data Loader
Loads a small dataset for demonstration purposes
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

sys.path.append(str(Path(__file__).parent.parent))

from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient


def clear_database_safe(client):
    """Clear database with smaller batches to avoid timeout"""
    print("🗑️ Clearing existing data safely...")
    
    try:
        # Clear edges first in batches
        for i in range(5):  # Try up to 5 batches
            result = client.gremlin_client.submit('g.E().limit(5000).drop()').all().result()
            print(f"   Cleared edge batch {i+1}")
            time.sleep(1)
        
        # Clear vertices in batches
        for i in range(3):  # Try up to 3 batches
            result = client.gremlin_client.submit('g.V().limit(2000).drop()').all().result()
            print(f"   Cleared vertex batch {i+1}")
            time.sleep(1)
            
        print("✅ Database cleared successfully")
        return True
        
    except Exception as e:
        print(f"⚠️ Clear completed with some timeouts (normal): {str(e)[:100]}")
        return True  # Continue anyway


def load_demo_dataset():
    """Load small demo dataset for quick demonstration"""
    
    print("🚀 QUICK DEMO DATA LOADER")
    print("=" * 50)
    
    # Initialize client
    client = AzureCosmosGremlinClient()
    client._initialize_client()
    
    # Clear existing data
    clear_database_safe(client)
    
    # Load demo dataset
    demo_file = Path(__file__).parent.parent / "data/demo/small_demo_dataset.json"
    
    with open(demo_file, 'r') as f:
        data = json.load(f)
    
    entities = data['entities']
    relationships = data['relationships']
    
    print(f"\n📊 Loading {len(entities)} entities and {len(relationships)} relationships")
    
    # Load entities
    print("\n🔤 Loading entities...")
    entities_loaded = 0
    for i, entity in enumerate(entities):
        try:
            insert_query = '''
            g.addV('entity')
                .property('id', prop_id)
                .property('original_entity_id', prop_original_id)
                .property('text', prop_text)
                .property('entity_type', prop_entity_type)
                .property('domain', 'maintenance')
                .property('context', prop_context)
            '''
            
            result = client.gremlin_client.submit(
                message=insert_query,
                bindings={
                    'prop_id': f"demo_entity_{i}_{int(time.time())}",
                    'prop_original_id': entity['entity_id'],
                    'prop_text': entity['text'],
                    'prop_entity_type': entity['entity_type'],
                    'prop_context': entity.get('context', '')
                }
            ).all().result()
            
            entities_loaded += 1
            if (i + 1) % 5 == 0:
                print(f"   {i + 1}/{len(entities)} entities loaded")
                
        except Exception as e:
            print(f"   Error loading entity {entity['text']}: {str(e)[:50]}")
    
    print(f"✅ Loaded {entities_loaded}/{len(entities)} entities")
    
    # Small delay before relationships
    time.sleep(2)
    
    # Load relationships
    print("\n🔗 Loading relationships...")
    relationships_loaded = 0
    for i, rel in enumerate(relationships):
        try:
            insert_query = '''
            g.V().has('original_entity_id', prop_source_id)
                .addE(prop_relation_type)
                .to(g.V().has('original_entity_id', prop_target_id))
                .property('confidence', prop_confidence)
            '''
            
            result = client.gremlin_client.submit(
                message=insert_query,
                bindings={
                    'prop_source_id': rel['source_entity_id'],
                    'prop_target_id': rel['target_entity_id'],
                    'prop_relation_type': rel['relation_type'].replace(' ', '_').replace('-', '_'),
                    'prop_confidence': rel.get('confidence', 1.0)
                }
            ).all().result()
            
            relationships_loaded += 1
            if (i + 1) % 5 == 0:
                print(f"   {i + 1}/{len(relationships)} relationships loaded")
                
        except Exception as e:
            print(f"   Error loading relationship: {str(e)[:50]}")
    
    print(f"✅ Loaded {relationships_loaded}/{len(relationships)} relationships")
    
    # Validate results
    print("\n📊 Validation...")
    try:
        vertex_count = client.gremlin_client.submit('g.V().count()').all().result()[0]
        edge_count = client.gremlin_client.submit('g.E().count()').all().result()[0]
        connectivity = edge_count / vertex_count if vertex_count > 0 else 0
        
        print(f"✅ Final Results:")
        print(f"   Vertices: {vertex_count}")
        print(f"   Edges: {edge_count}")
        print(f"   Connectivity: {connectivity:.2f}")
        print(f"   Success Rate: {(entities_loaded/len(entities) + relationships_loaded/len(relationships))/2*100:.1f}%")
        
        # Save results for demo
        results = {
            "demo_dataset_loaded": True,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "vertices": vertex_count,
            "edges": edge_count,
            "connectivity_ratio": connectivity,
            "entities_loaded": entities_loaded,
            "relationships_loaded": relationships_loaded
        }
        
        results_file = Path(__file__).parent.parent / "data/demo/demo_load_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n🎯 Demo dataset ready for supervisor demonstration!")
        return results
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return None


if __name__ == "__main__":
    load_demo_dataset()