#!/usr/bin/env python3
"""
Test script for optimized bulk loader with small dataset
"""

import asyncio
import json
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from scripts.optimized_cosmos_bulk_loader import OptimizedCosmosBulkLoader, BulkLoadConfig


async def test_small_dataset():
    """Test with a very small dataset to validate the optimization approach"""
    
    print("🧪 TESTING OPTIMIZED BULK LOADER - SMALL DATASET")
    print("=" * 60)
    
    # Create small test dataset
    test_entities = [
        {"entity_id": "test_001", "text": "test air conditioner", "entity_type": "equipment"},
        {"entity_id": "test_002", "text": "test thermostat", "entity_type": "component"},
        {"entity_id": "test_003", "text": "test failure", "entity_type": "issue"}
    ]
    
    test_relationships = [
        {
            "source_entity_id": "test_001",
            "target_entity_id": "test_002", 
            "relation_type": "has_part",
            "confidence": 0.9
        },
        {
            "source_entity_id": "test_002",
            "target_entity_id": "test_003",
            "relation_type": "has_issue", 
            "confidence": 0.8
        }
    ]
    
    # Conservative configuration for testing
    config = BulkLoadConfig(
        max_concurrent_requests=3,
        batch_size=5,
        rate_limit_delay=0.5,  # Slower for stability
        retry_attempts=2
    )
    
    loader = OptimizedCosmosBulkLoader(config)
    
    try:
        print(f"\n🔄 Testing entity loading...")
        start_time = time.time()
        
        entity_results = await loader.bulk_load_entities(test_entities)
        entity_duration = time.time() - start_time
        
        print(f"✅ Entity results: {entity_results}")
        print(f"⏱️  Entity duration: {entity_duration:.2f}s")
        print(f"📈 Entity throughput: {len(test_entities)/entity_duration:.2f} items/sec")
        
        print(f"\n🔄 Testing relationship loading...")
        start_time = time.time()
        
        relationship_results = await loader.bulk_load_relationships(test_relationships)
        relationship_duration = time.time() - start_time
        
        print(f"✅ Relationship results: {relationship_results}")
        print(f"⏱️  Relationship duration: {relationship_duration:.2f}s") 
        print(f"📈 Relationship throughput: {len(test_relationships)/relationship_duration:.2f} items/sec")
        
        # Calculate overall performance
        total_items = len(test_entities) + len(test_relationships)
        total_duration = entity_duration + relationship_duration
        overall_throughput = total_items / total_duration
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   Total items: {total_items}")
        print(f"   Total duration: {total_duration:.2f}s")
        print(f"   Overall throughput: {overall_throughput:.2f} items/sec")
        
        # Compare to baseline (1.3 items/sec from individual operations)
        baseline_throughput = 1.3
        improvement = overall_throughput / baseline_throughput
        print(f"   Improvement over baseline: {improvement:.1f}x")
        
        # Save test results
        results = {
            "test_type": "small_dataset_validation",
            "timestamp": time.time(),
            "entities": {
                "count": len(test_entities),
                "results": entity_results,
                "duration": entity_duration,
                "throughput": len(test_entities)/entity_duration
            },
            "relationships": {
                "count": len(test_relationships), 
                "results": relationship_results,
                "duration": relationship_duration,
                "throughput": len(test_relationships)/relationship_duration
            },
            "overall": {
                "total_items": total_items,
                "total_duration": total_duration,
                "overall_throughput": overall_throughput,
                "baseline_comparison": improvement
            },
            "configuration": {
                "max_concurrent_requests": config.max_concurrent_requests,
                "batch_size": config.batch_size,
                "rate_limit_delay": config.rate_limit_delay
            }
        }
        
        output_file = Path(__file__).parent.parent / "data/demo_outputs/bulk_loader_test_results.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Test results saved: {output_file}")
        
        if overall_throughput > baseline_throughput:
            print(f"🎉 SUCCESS: Optimized loader is {improvement:.1f}x faster!")
            return 0
        else:
            print(f"⚠️  WARNING: Performance not improved significantly")
            return 1
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(asyncio.run(test_small_dataset()))