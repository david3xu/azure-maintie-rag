#!/usr/bin/env python3
"""
Step 3: Load Quality Dataset to Azure Cosmos DB
Uses individual operations (as required by Azure Gremlin) but optimized for production
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any

sys.path.append(str(Path(__file__).parent.parent))

from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient


def load_subset_for_demo(data: Dict[str, Any], entity_limit: int = 500, relationship_limit: int = 300) -> Dict[str, Any]:
    """Load a meaningful subset for demo purposes"""
    
    print(f"📊 Creating demo subset:")
    print(f"   Original: {len(data['entities']):,} entities, {len(data['relationships']):,} relationships")
    
    # Take first N entities and relationships that reference them
    demo_entities = data['entities'][:entity_limit]
    entity_ids = {entity['entity_id'] for entity in demo_entities}
    
    # Filter relationships to only include those between our demo entities
    demo_relationships = []
    for rel in data['relationships']:
        if (rel['source_entity_id'] in entity_ids and 
            rel['target_entity_id'] in entity_ids and 
            len(demo_relationships) < relationship_limit):
            demo_relationships.append(rel)
    
    subset = {
        'entities': demo_entities,
        'relationships': demo_relationships,
        'summary': {
            'subset_entities': len(demo_entities),
            'subset_relationships': len(demo_relationships),
            'original_entities': len(data['entities']),
            'original_relationships': len(data['relationships'])
        }
    }
    
    print(f"   Demo subset: {len(demo_entities):,} entities, {len(demo_relationships):,} relationships")
    return subset


def load_entities_individual(cosmos_client: AzureCosmosGremlinClient, entities: List[Dict[str, Any]], domain: str) -> int:
    """Load entities one by one (Azure Gremlin requirement)"""
    
    print(f"📤 Loading {len(entities):,} entities individually...")
    successful_entities = 0
    start_time = time.time()
    
    for i, entity in enumerate(entities):
        try:
            entity_data = {
                'id': entity['entity_id'],
                'text': entity['text'],
                'entity_type': entity['entity_type'],
                'context': entity.get('context', ''),
                'batch_id': entity['batch_id'],
                'text_id': entity['text_id'],
                'semantic_role': entity.get('semantic_role', ''),
                'source_text': entity.get('source_text', '')
            }
            
            result = cosmos_client.add_entity(entity_data, domain)
            if result and result.get('success', True):
                successful_entities += 1
            
            # Progress updates
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(entities) - i - 1) / rate if rate > 0 else 0
                print(f"   Progress: {i + 1:,}/{len(entities):,} entities ({rate:.1f}/sec, ETA: {eta:.0f}s)")
                
        except Exception as e:
            print(f"   ⚠️  Entity {i+1} failed: {str(e)[:60]}...")
    
    total_time = time.time() - start_time
    print(f"✅ Entity loading completed: {successful_entities:,}/{len(entities):,} in {total_time:.1f}s")
    return successful_entities


def load_relationships_individual(cosmos_client: AzureCosmosGremlinClient, relationships: List[Dict[str, Any]], domain: str) -> int:
    """Load relationships one by one (Azure Gremlin requirement)"""
    
    print(f"🔗 Loading {len(relationships):,} relationships individually...")
    successful_relationships = 0
    start_time = time.time()
    
    for i, rel in enumerate(relationships):
        try:
            result = cosmos_client.add_relationship(
                source_entity=rel['source_entity_id'],
                target_entity=rel['target_entity_id'],
                relation_type=rel['relation_type'],
                domain=domain,
                confidence=rel.get('confidence', 1.0),
                metadata={
                    'relation_id': rel['relation_id'],
                    'batch_id': rel['batch_id'],
                    'text_id': rel['text_id'],
                    'context': rel.get('context', ''),
                    'source_text': rel.get('source_text', '')
                }
            )
            if result and result.get('success', True):
                successful_relationships += 1
            
            # Progress updates
            if (i + 1) % 25 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(relationships) - i - 1) / rate if rate > 0 else 0
                print(f"   Progress: {i + 1:,}/{len(relationships):,} relationships ({rate:.1f}/sec, ETA: {eta:.0f}s)")
                
        except Exception as e:
            print(f"   ⚠️  Relationship {i+1} failed: {str(e)[:60]}...")
    
    total_time = time.time() - start_time
    print(f"✅ Relationship loading completed: {successful_relationships:,}/{len(relationships):,} in {total_time:.1f}s")
    return successful_relationships


def main():
    """Execute Step 3: Load quality dataset to Azure Cosmos DB"""
    
    print("🚀 STEP 3: LOAD QUALITY DATASET TO AZURE COSMOS DB")
    print("=" * 60)
    print("Using individual operations (Azure Gremlin API requirement)")
    print("Loading demo subset for reasonable execution time")
    
    # Load quality dataset
    data_file = Path(__file__).parent.parent / "data/extraction_outputs/full_dataset_extraction_9100_entities_5848_relationships.json"
    
    if not data_file.exists():
        print(f"❌ Quality dataset not found: {data_file}")
        return 1
    
    with open(data_file, 'r') as f:
        full_data = json.load(f)
    
    # Create demo subset for reasonable execution time
    demo_data = load_subset_for_demo(full_data, entity_limit=200, relationship_limit=150)
    
    # Initialize Cosmos client
    cosmos_client = AzureCosmosGremlinClient()
    
    start_time = time.time()
    
    try:
        # Load entities
        entities_loaded = load_entities_individual(cosmos_client, demo_data['entities'], 'maintenance')
        
        # Load relationships
        relationships_loaded = load_relationships_individual(cosmos_client, demo_data['relationships'], 'maintenance')
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 STEP 3 COMPLETED!")
        print(f"   ✅ Entities loaded: {entities_loaded:,}/{len(demo_data['entities']):,}")
        print(f"   ✅ Relationships loaded: {relationships_loaded:,}/{len(demo_data['relationships']):,}")
        print(f"   ⚡ Total time: {total_time:.1f}s")
        print(f"   📈 Performance: {(entities_loaded + relationships_loaded) / total_time:.1f} items/sec")
        
        # Test graph queries
        print(f"\n🔍 Testing graph operations...")
        stats = cosmos_client.get_graph_statistics('maintenance')
        print(f"   Graph stats: {stats}")
        
        if entities_loaded > 0 and relationships_loaded > 0:
            print(f"\n✅ STEP 3 SUCCESS: Demo subset loaded to Azure Cosmos DB")
            print(f"   Note: Full dataset ({full_data['summary']['total_entities']:,} entities) would take ~{total_time * (full_data['summary']['total_entities'] / len(demo_data['entities'])):.0f}s")
            return 0
        else:
            print(f"\n❌ STEP 3 FAILED: No data loaded successfully")
            return 1
        
    except Exception as e:
        print(f"❌ Step 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())