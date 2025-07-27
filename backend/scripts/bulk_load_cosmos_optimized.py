#!/usr/bin/env python3
"""
Optimized Bulk Loading to Azure Cosmos DB
Uses batch Gremlin queries instead of individual operations
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient


class OptimizedCosmosLoader:
    """Optimized bulk loader using batch Gremlin queries"""
    
    def __init__(self):
        self.cosmos_client = AzureCosmosGremlinClient()
        self.cosmos_client._initialize_client()
        
    def build_bulk_entity_query(self, entities: List[Dict[str, Any]], domain: str) -> str:
        """Build a single Gremlin query for multiple entities"""
        
        query_parts = []
        for i, entity in enumerate(entities):
            entity_id = entity['entity_id'].replace("'", "\\'")
            entity_text = str(entity['text'])[:500].replace("'", "\\'")
            entity_type = entity['entity_type'].replace("'", "\\'")
            
            query_parts.append(f"""
                g.addV('Entity{i}')
                    .property('id', '{entity_id}')
                    .property('partitionKey', '{domain}')
                    .property('text', '{entity_text}')
                    .property('domain', '{domain}')
                    .property('entity_type', '{entity_type}')
                    .property('created_at', '{datetime.now().isoformat()}')
            """)
        
        # Join all parts with semicolons for batch execution
        return "; ".join(query_parts)
    
    def build_bulk_relationship_query(self, relationships: List[Dict[str, Any]], domain: str) -> str:
        """Build a single Gremlin query for multiple relationships"""
        
        query_parts = []
        for i, rel in enumerate(relationships):
            source_id = rel['source_entity_id'].replace("'", "\\'")
            target_id = rel['target_entity_id'].replace("'", "\\'")
            relation_type = rel['relation_type'].replace("'", "\\'")
            confidence = rel.get('confidence', 1.0)
            
            query_parts.append(f"""
                g.V().has('id', '{source_id}').has('domain', '{domain}')
                    .addE('{relation_type}')
                    .property('confidence', {confidence})
                    .property('domain', '{domain}')
                    .property('created_at', '{datetime.now().isoformat()}')
                    .to(g.V().has('id', '{target_id}').has('domain', '{domain}'))
            """)
        
        return "; ".join(query_parts)
    
    def execute_bulk_query(self, query: str, operation: str, batch_size: int) -> bool:
        """Execute bulk Gremlin query with error handling"""
        
        try:
            start_time = time.time()
            result = self.cosmos_client.gremlin_client.submit(query)
            result.all().result()  # Wait for completion
            
            execution_time = time.time() - start_time
            print(f"   ✅ {operation} batch ({batch_size} items): {execution_time:.2f}s")
            return True
            
        except Exception as e:
            print(f"   ❌ {operation} batch failed: {str(e)[:100]}...")
            return False
    
    def load_entities_optimized(self, entities: List[Dict[str, Any]], domain: str, batch_size: int = 25) -> int:
        """Load entities using optimized bulk queries"""
        
        print(f"📤 Loading {len(entities):,} entities with batch size {batch_size}...")
        successful_batches = 0
        total_entities_loaded = 0
        
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(entities) + batch_size - 1) // batch_size
            
            print(f"   Processing entity batch {batch_num}/{total_batches}...")
            
            # Build bulk query for this batch
            bulk_query = self.build_bulk_entity_query(batch, domain)
            
            # Execute bulk query
            if self.execute_bulk_query(bulk_query, "Entity", len(batch)):
                successful_batches += 1
                total_entities_loaded += len(batch)
            
            # Progress update every 10 batches
            if batch_num % 10 == 0:
                print(f"   📊 Progress: {total_entities_loaded:,}/{len(entities):,} entities loaded")
        
        print(f"✅ Entity loading completed: {total_entities_loaded:,}/{len(entities):,} loaded successfully")
        return total_entities_loaded
    
    def load_relationships_optimized(self, relationships: List[Dict[str, Any]], domain: str, batch_size: int = 25) -> int:
        """Load relationships using optimized bulk queries"""
        
        print(f"🔗 Loading {len(relationships):,} relationships with batch size {batch_size}...")
        successful_batches = 0
        total_relationships_loaded = 0
        
        for i in range(0, len(relationships), batch_size):
            batch = relationships[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(relationships) + batch_size - 1) // batch_size
            
            print(f"   Processing relationship batch {batch_num}/{total_batches}...")
            
            # Build bulk query for this batch
            bulk_query = self.build_bulk_relationship_query(batch, domain)
            
            # Execute bulk query
            if self.execute_bulk_query(bulk_query, "Relationship", len(batch)):
                successful_batches += 1
                total_relationships_loaded += len(batch)
            
            # Progress update every 10 batches
            if batch_num % 10 == 0:
                print(f"   📊 Progress: {total_relationships_loaded:,}/{len(relationships):,} relationships loaded")
        
        print(f"✅ Relationship loading completed: {total_relationships_loaded:,}/{len(relationships):,} loaded successfully")
        return total_relationships_loaded


def main():
    """Main function for optimized bulk loading"""
    
    print("🚀 OPTIMIZED BULK LOADING TO AZURE COSMOS DB")
    print("=" * 60)
    print("Using batch Gremlin queries instead of individual operations")
    print("Cosmos DB: 4000 RU/s autoscale enabled")
    
    # Load quality dataset
    data_file = Path(__file__).parent.parent / "data/extraction_outputs/full_dataset_extraction_9100_entities_5848_relationships.json"
    
    if not data_file.exists():
        print(f"❌ Quality dataset not found: {data_file}")
        return 1
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    entities = data['entities']
    relationships = data['relationships']
    
    print(f"📊 Dataset loaded:")
    print(f"   Entities: {len(entities):,}")
    print(f"   Relationships: {len(relationships):,}")
    
    # Initialize optimized loader
    loader = OptimizedCosmosLoader()
    
    start_time = time.time()
    
    try:
        # Load entities with optimized bulk operations
        entities_loaded = loader.load_entities_optimized(entities, 'maintenance', batch_size=25)
        
        # Load relationships with optimized bulk operations  
        relationships_loaded = loader.load_relationships_optimized(relationships, 'maintenance', batch_size=25)
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 OPTIMIZED BULK LOADING COMPLETED!")
        print(f"   ✅ Entities loaded: {entities_loaded:,}/{len(entities):,}")
        print(f"   ✅ Relationships loaded: {relationships_loaded:,}/{len(relationships):,}")
        print(f"   ⚡ Total time: {total_time:.1f}s")
        print(f"   📈 Performance: {(entities_loaded + relationships_loaded) / total_time:.1f} items/sec")
        
        if entities_loaded > 0 and relationships_loaded > 0:
            print(f"\n✅ STEP 3 COMPLETED: Quality dataset successfully loaded to Azure Cosmos DB")
            return 0
        else:
            print(f"\n❌ STEP 3 FAILED: Bulk loading encountered errors")
            return 1
        
    except Exception as e:
        print(f"❌ Optimized bulk loading failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())