#!/usr/bin/env python3
"""
Load Quality Dataset to Azure Cosmos DB
Uses the existing high-quality dataset efficiently
"""

import sys
import json
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient
from core.workflow.progress_tracker import create_progress_tracker


def load_quality_dataset() -> Dict[str, Any]:
    """Load the existing quality dataset"""
    
    data_file = Path(__file__).parent.parent / "data/extraction_outputs/full_dataset_extraction_9100_entities_5848_relationships.json"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Quality dataset not found: {data_file}")
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    print(f"✅ Loaded quality dataset:")
    print(f"   Entities: {len(data['entities']):,}")
    print(f"   Relationships: {len(data['relationships']):,}")
    print(f"   File size: {data_file.stat().st_size / 1024 / 1024:.1f}MB")
    
    return data


def load_to_cosmos_db(data: Dict[str, Any], batch_size: int = 50) -> Dict[str, Any]:
    """Load data efficiently to Azure Cosmos DB using the working pattern"""
    
    # Initialize progress tracker (using working pattern)
    progress_tracker = create_progress_tracker("Quality Dataset to Cosmos DB")
    progress_tracker.start_workflow()
    
    # Initialize Cosmos client
    cosmos_client = AzureCosmosGremlinClient()
    
    entities = data['entities']
    relationships = data['relationships']
    
    print(f"\n🚀 Starting efficient load to Azure Cosmos DB...")
    print(f"   Entities: {len(entities):,}")
    print(f"   Relationships: {len(relationships):,}")
    print(f"   Batch size: {batch_size}")
    
    # Load entities efficiently
    progress_tracker.start_step("Entity Loading", {
        "total_entities": len(entities),
        "batch_size": batch_size
    })
    
    successful_entities = 0
    entity_errors = 0
    
    print(f"\n📤 Loading entities...")
    for i in range(0, len(entities), batch_size):
        batch = entities[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(entities) + batch_size - 1) // batch_size
        
        batch_start_time = time.time()
        batch_successes = 0
        
        for entity in batch:
            try:
                # Use working format from load_real_demo_data.py
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
                
                result = cosmos_client.add_entity(entity_data, 'maintenance')
                if result and result.get('success', True):
                    successful_entities += 1
                    batch_successes += 1
                else:
                    entity_errors += 1
                    
            except Exception as e:
                entity_errors += 1
                if entity_errors <= 5:  # Only show first few errors
                    print(f"   ⚠️  Entity error: {str(e)[:60]}...")
        
        batch_time = time.time() - batch_start_time
        print(f"   ✅ Batch {batch_num}/{total_batches}: {batch_successes}/{len(batch)} entities loaded ({batch_time:.1f}s)")
        
        progress_tracker.update_step_progress("Entity Loading", {
            "current_batch": batch_num,
            "total_batches": total_batches,
            "successful_entities": successful_entities,
            "entity_errors": entity_errors
        })
    
    progress_tracker.complete_step("Entity Loading", success=True)
    
    # Load relationships efficiently
    progress_tracker.start_step("Relationship Loading", {
        "total_relationships": len(relationships),
        "batch_size": batch_size
    })
    
    successful_relationships = 0
    relationship_errors = 0
    
    print(f"\n🔗 Loading relationships...")
    for i in range(0, len(relationships), batch_size):
        batch = relationships[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(relationships) + batch_size - 1) // batch_size
        
        batch_start_time = time.time()
        batch_successes = 0
        
        for rel in batch:
            try:
                result = cosmos_client.add_relationship(
                    source_entity=rel['source_entity_id'],
                    target_entity=rel['target_entity_id'],
                    relation_type=rel['relation_type'],
                    domain='maintenance',
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
                    batch_successes += 1
                else:
                    relationship_errors += 1
                    
            except Exception as e:
                relationship_errors += 1
                if relationship_errors <= 5:  # Only show first few errors
                    print(f"   ⚠️  Relationship error: {str(e)[:60]}...")
        
        batch_time = time.time() - batch_start_time
        print(f"   ✅ Batch {batch_num}/{total_batches}: {batch_successes}/{len(batch)} relationships loaded ({batch_time:.1f}s)")
        
        progress_tracker.update_step_progress("Relationship Loading", {
            "current_batch": batch_num,
            "total_batches": total_batches,
            "successful_relationships": successful_relationships,
            "relationship_errors": relationship_errors
        })
    
    progress_tracker.complete_step("Relationship Loading", success=True)
    progress_tracker.finish_workflow(success=True)
    
    results = {
        "successful_entities": successful_entities,
        "entity_errors": entity_errors,
        "successful_relationships": successful_relationships,
        "relationship_errors": relationship_errors,
        "total_entities": len(entities),
        "total_relationships": len(relationships)
    }
    
    print(f"\n🎉 STEP 3 COMPLETED: Quality dataset loaded to Azure Cosmos DB")
    print(f"   ✅ Entities: {successful_entities:,}/{len(entities):,} loaded ({entity_errors} errors)")
    print(f"   ✅ Relationships: {successful_relationships:,}/{len(relationships):,} loaded ({relationship_errors} errors)")
    print(f"   ✅ Success rate: {(successful_entities + successful_relationships) / (len(entities) + len(relationships)) * 100:.1f}%")
    
    return results


def main():
    """Main function for loading quality dataset to Cosmos DB"""
    
    print("🌐 QUALITY DATASET TO AZURE COSMOS DB")
    print("=" * 60)
    print("Loading existing high-quality dataset efficiently")
    
    try:
        # Load quality dataset
        data = load_quality_dataset()
        
        # Load to Cosmos DB
        results = load_to_cosmos_db(data)
        
        print(f"\n{'='*80}")
        print("🎯 STEP 3 COMPLETE: QUALITY DATA IN AZURE COSMOS DB")
        print(f"{'='*80}")
        print(f"\n📊 Final Results:")
        print(f"   Entities loaded: {results['successful_entities']:,}")
        print(f"   Relationships loaded: {results['successful_relationships']:,}")
        print(f"   Total elements: {results['successful_entities'] + results['successful_relationships']:,}")
        print(f"\n🚀 Ready for Step 4: GNN Training Pipeline!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Quality dataset loading failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())