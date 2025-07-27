#!/usr/bin/env python3
"""
Azure Knowledge Graph Bulk Loader
Reusable script for loading entities and relationships to Azure Cosmos DB
with real-time progress monitoring and error handling
"""

import json
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Add backend to path - for running from backend directory
sys.path.insert(0, '.')

from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient


class AzureKGBulkLoader:
    """Azure Knowledge Graph bulk loader with progress monitoring"""
    
    def __init__(self, batch_size: int = 100, max_entities: int = None):
        self.batch_size = batch_size
        self.max_entities = max_entities
        self.client = AzureCosmosGremlinClient()
        self.client._initialize_client()
        
        # Statistics
        self.stats = {
            'entities_loaded': 0,
            'relationships_loaded': 0,
            'entities_failed': 0,
            'relationships_failed': 0,
            'start_time': None,
            'batches_completed': 0
        }
        
    def clear_database(self) -> bool:
        """Clear all existing data from Azure Cosmos DB"""
        print("🗑️ Clearing existing data from Azure Cosmos DB...")
        try:
            self.client.gremlin_client.submit('g.V().drop()').all().result()
            print("   ✅ Database cleared successfully")
            time.sleep(1)  # Brief pause for consistency
            return True
        except Exception as e:
            print(f"   ❌ Clear failed: {str(e)[:100]}")
            return False
    
    def load_entities_batch(self, entities: List[Dict], batch_num: int) -> Tuple[int, int]:
        """Load a batch of entities with real-time feedback"""
        
        print(f"📊 Batch {batch_num}: Loading {len(entities)} entities...")
        
        # Azure Gremlin vertex query
        insert_vertex_query = '''
        g.addV('entity')
            .property('id', prop_id)
            .property('original_entity_id', prop_original_id)
            .property('text', prop_text)
            .property('entity_type', prop_entity_type)
            .property('domain', prop_domain)
            .property('context', prop_context)
        '''
        
        success_count = 0
        error_count = 0
        
        for i, entity in enumerate(entities):
            try:
                # Generate unique ID to avoid conflicts
                unique_id = f"entity_{batch_num}_{i}_{int(time.time() * 1000000) % 1000000}"
                
                result = self.client.gremlin_client.submit(
                    message=insert_vertex_query,
                    bindings={
                        'prop_id': unique_id,
                        'prop_original_id': entity['entity_id'],
                        'prop_text': entity['text'][:200],  # Limit text length
                        'prop_entity_type': entity['entity_type'],
                        'prop_domain': 'maintenance',
                        'prop_context': entity.get('context', '')[:200]
                    }
                ).all().result()
                
                success_count += 1
                
                # Real-time progress within batch
                if (i + 1) % 20 == 0:
                    progress = (i + 1) / len(entities) * 100
                    print(f"      {i + 1}/{len(entities)} ({progress:.1f}%) - Success: {success_count}, Errors: {error_count}")
                    
            except Exception as e:
                error_count += 1
                if error_count <= 3:  # Show first few errors
                    print(f"      Error {error_count}: {str(e)[:80]}...")
        
        print(f"   ✅ Batch {batch_num} completed: {success_count}/{len(entities)} entities loaded, {error_count} errors")
        return success_count, error_count
    
    def load_relationships_batch(self, relationships: List[Dict], entity_id_map: Dict, batch_num: int) -> Tuple[int, int]:
        """Load a batch of relationships with real-time feedback"""
        
        print(f"🔗 Batch {batch_num}: Loading {len(relationships)} relationships...")
        
        # Azure Gremlin edge query
        insert_edge_query = '''
        g.V().has('original_entity_id', prop_source_id)
            .addE(prop_relation_type)
            .to(g.V().has('original_entity_id', prop_target_id))
            .property('confidence', prop_confidence)
        '''
        
        success_count = 0
        error_count = 0
        
        for i, rel in enumerate(relationships):
            try:
                result = self.client.gremlin_client.submit(
                    message=insert_edge_query,
                    bindings={
                        'prop_source_id': rel['source_entity_id'],
                        'prop_target_id': rel['target_entity_id'],
                        'prop_relation_type': rel['relation_type'].replace(' ', '_').replace('-', '_'),
                        'prop_confidence': rel.get('confidence', 1.0)
                    }
                ).all().result()
                
                success_count += 1
                
                # Real-time progress within batch
                if (i + 1) % 10 == 0:
                    progress = (i + 1) / len(relationships) * 100
                    print(f"      {i + 1}/{len(relationships)} ({progress:.1f}%) - Success: {success_count}, Errors: {error_count}")
                    
            except Exception as e:
                error_count += 1
                if error_count <= 3:  # Show first few errors
                    print(f"      Error {error_count}: {str(e)[:80]}...")
        
        print(f"   ✅ Batch {batch_num} completed: {success_count}/{len(relationships)} relationships loaded, {error_count} errors")
        return success_count, error_count
    
    def load_entities(self, entities: List[Dict]) -> bool:
        """Load all entities in batches with progress monitoring"""
        
        total_entities = min(len(entities), self.max_entities) if self.max_entities else len(entities)
        entities_to_load = entities[:total_entities]
        
        print(f"\\n📊 LOADING {total_entities:,} ENTITIES")
        print("=" * 60)
        
        self.stats['start_time'] = time.time()
        
        # Process in batches
        for batch_start in range(0, len(entities_to_load), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(entities_to_load))
            batch_entities = entities_to_load[batch_start:batch_end]
            batch_num = (batch_start // self.batch_size) + 1
            
            success, errors = self.load_entities_batch(batch_entities, batch_num)
            
            self.stats['entities_loaded'] += success
            self.stats['entities_failed'] += errors
            self.stats['batches_completed'] += 1
            
            # Overall progress
            elapsed = time.time() - self.stats['start_time']
            rate = self.stats['entities_loaded'] / elapsed if elapsed > 0 else 0
            remaining = total_entities - self.stats['entities_loaded']
            eta = remaining / rate if rate > 0 else 0
            
            print(f"\\n📈 Overall Progress: {self.stats['entities_loaded']:,}/{total_entities:,} entities")
            print(f"   Rate: {rate:.1f} entities/sec")
            print(f"   ETA: {eta/60:.1f} minutes")
            print(f"   Success Rate: {self.stats['entities_loaded']/(self.stats['entities_loaded']+self.stats['entities_failed'])*100:.1f}%")
            print()
            
            # Brief pause between batches
            time.sleep(0.5)
        
        return self.stats['entities_loaded'] > 0
    
    def load_relationships(self, relationships: List[Dict]) -> bool:
        """Load relationships in batches with progress monitoring"""
        
        print(f"\\n🔗 LOADING RELATIONSHIPS")
        print("=" * 60)
        
        # Filter relationships to only include those between loaded entities
        print("   Filtering relationships for loaded entities...")
        
        # Get loaded entity IDs
        loaded_entities_query = "g.V().values('original_entity_id')"
        try:
            loaded_entity_ids = set(self.client.gremlin_client.submit(loaded_entities_query).all().result())
            print(f"   Found {len(loaded_entity_ids)} loaded entities")
        except Exception as e:
            print(f"   Error getting loaded entities: {e}")
            return False
        
        # Filter valid relationships and deduplicate
        valid_relationships = []
        seen_relationships = set()
        for rel in relationships:
            if (rel['source_entity_id'] in loaded_entity_ids and 
                rel['target_entity_id'] in loaded_entity_ids):
                # Create unique key for deduplication
                rel_key = (rel['source_entity_id'], rel['target_entity_id'], rel['relation_type'])
                if rel_key not in seen_relationships:
                    valid_relationships.append(rel)
                    seen_relationships.add(rel_key)
        
        print(f"   Found {len(valid_relationships)} valid relationships to load")
        
        if not valid_relationships:
            print("   ⚠️ No valid relationships found")
            return False
        
        # Process in batches
        for batch_start in range(0, len(valid_relationships), self.batch_size // 2):  # Smaller batches for relationships
            batch_end = min(batch_start + self.batch_size // 2, len(valid_relationships))
            batch_relationships = valid_relationships[batch_start:batch_end]
            batch_num = (batch_start // (self.batch_size // 2)) + 1
            
            success, errors = self.load_relationships_batch(batch_relationships, {}, batch_num)
            
            self.stats['relationships_loaded'] += success
            self.stats['relationships_failed'] += errors
            
            # Overall progress
            progress = (batch_end / len(valid_relationships)) * 100
            print(f"\\n📈 Relationship Progress: {batch_end}/{len(valid_relationships)} ({progress:.1f}%)")
            print(f"   Loaded: {self.stats['relationships_loaded']}")
            print(f"   Errors: {self.stats['relationships_failed']}")
            print()
            
            # Brief pause between batches
            time.sleep(0.5)
        
        return self.stats['relationships_loaded'] > 0
    
    def validate_results(self) -> Dict[str, Any]:
        """Validate the loaded knowledge graph"""
        
        print("\\n📊 VALIDATING LOADED KNOWLEDGE GRAPH")
        print("=" * 60)
        
        try:
            # Get counts
            vertex_count = self.client.gremlin_client.submit('g.V().count()').all().result()[0]
            edge_count = self.client.gremlin_client.submit('g.E().count()').all().result()[0]
            
            # Get entity type distribution
            type_query = "g.V().groupCount().by('entity_type')"
            type_result = self.client.gremlin_client.submit(type_query).all().result()
            entity_types = type_result[0] if type_result else {}
            
            # Calculate metrics
            connectivity = edge_count / vertex_count if vertex_count > 0 else 0
            elapsed = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
            
            validation_results = {
                'vertices': vertex_count,
                'edges': edge_count,
                'connectivity_ratio': connectivity,
                'entity_types': len(entity_types),
                'total_duration': elapsed,
                'entities_loaded': self.stats['entities_loaded'],
                'relationships_loaded': self.stats['relationships_loaded'],
                'success_rate_entities': self.stats['entities_loaded'] / (self.stats['entities_loaded'] + self.stats['entities_failed']) * 100 if (self.stats['entities_loaded'] + self.stats['entities_failed']) > 0 else 0,
                'success_rate_relationships': self.stats['relationships_loaded'] / (self.stats['relationships_loaded'] + self.stats['relationships_failed']) * 100 if (self.stats['relationships_loaded'] + self.stats['relationships_failed']) > 0 else 0
            }
            
            print(f"✅ Validation Results:")
            print(f"   Vertices in Azure: {vertex_count:,}")
            print(f"   Edges in Azure: {edge_count:,}")
            print(f"   Connectivity Ratio: {connectivity:.3f}")
            print(f"   Entity Types: {len(entity_types)}")
            print(f"   Total Duration: {elapsed:.1f}s")
            print(f"   Entity Success Rate: {validation_results['success_rate_entities']:.1f}%")
            print(f"   Relationship Success Rate: {validation_results['success_rate_relationships']:.1f}%")
            
            return validation_results
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return {}
    
    def save_results(self, validation_results: Dict[str, Any]) -> str:
        """Save loading results to file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(__file__).parent.parent / f"data/loading_results/azure_kg_load_{timestamp}.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'timestamp': timestamp,
            'loading_stats': self.stats,
            'validation_results': validation_results,
            'configuration': {
                'batch_size': self.batch_size,
                'max_entities': self.max_entities
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\\n📁 Results saved: {results_file}")
        return str(results_file)


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description='Azure Knowledge Graph Bulk Loader')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for loading (default: 50)')
    parser.add_argument('--max-entities', type=int, default=1000, help='Maximum entities to load (default: 1000)')
    parser.add_argument('--skip-clear', action='store_true', help='Skip clearing existing data')
    parser.add_argument('--entities-only', action='store_true', help='Load only entities, skip relationships')
    
    args = parser.parse_args()
    
    print("🚀 AZURE KNOWLEDGE GRAPH BULK LOADER")
    print("=" * 80)
    print(f"Configuration: batch_size={args.batch_size}, max_entities={args.max_entities}")
    print()
    
    # Load quality dataset
    data_file = Path("data/extraction_outputs/full_dataset_extraction_9100_entities_5848_relationships.json")
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        return 1
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    entities = data['entities']
    relationships = data['relationships']
    
    print(f"📊 Dataset: {len(entities):,} entities, {len(relationships):,} relationships")
    print()
    
    # Initialize loader
    loader = AzureKGBulkLoader(batch_size=args.batch_size, max_entities=args.max_entities)
    
    try:
        # Clear database if requested
        if not args.skip_clear:
            if not loader.clear_database():
                print("❌ Failed to clear database")
                return 1
        
        # Load entities
        if not loader.load_entities(entities):
            print("❌ Failed to load entities")
            return 1
        
        # Load relationships if requested
        if not args.entities_only:
            if not loader.load_relationships(relationships):
                print("⚠️ Failed to load relationships (continuing)")
        
        # Validate results
        validation_results = loader.validate_results()
        
        # Save results
        results_file = loader.save_results(validation_results)
        
        print(f"\\n🎉 LOADING COMPLETED SUCCESSFULLY!")
        print(f"   Entities: {validation_results.get('vertices', 0):,}")
        print(f"   Relationships: {validation_results.get('edges', 0):,}")
        print(f"   Results: {results_file}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\\n⚠️ Loading interrupted by user")
        return 1
    except Exception as e:
        print(f"\\n❌ Loading failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())