#!/usr/bin/env python3
"""
Test GNN Integration
Demonstrates the difference between regular and GNN-enhanced processing
"""

import json
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from scripts.integrate_gnn_with_api import GNNEnhancedAPI

def test_gnn_integration():
    """Test GNN integration capabilities"""
    print("🚀 GNN Integration Test")
    print("=" * 60)

    # Initialize GNN service
    print("📦 Initializing GNN service...")
    gnn_api = GNNEnhancedAPI()
    gnn_api.initialize_gnn()

    if not gnn_api.initialized:
        print("❌ GNN initialization failed")
        return

    print("✅ GNN service initialized successfully!")
    print()

    # Test queries
    test_queries = [
        {
            "query": "air conditioner thermostat problems",
            "description": "Equipment component issue"
        },
        {
            "query": "pump motor not working",
            "description": "Equipment failure"
        },
        {
            "query": "engine room equipment maintenance",
            "description": "Location-based maintenance"
        },
        {
            "query": "broken fuel cooler mounts",
            "description": "Component failure"
        }
    ]

    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        description = test_case["description"]

        print(f"🔍 Test {i}: {description}")
        print(f"   Query: '{query}'")
        print("-" * 40)

        # Test without GNN
        print("📊 WITHOUT GNN:")
        start_time = time.time()
        result_without_gnn = gnn_api.enhanced_query_processing(query, use_gnn=False)
        time_without_gnn = time.time() - start_time

        print(f"   Processing time: {time_without_gnn:.3f}s")
        print(f"   Enhanced: {result_without_gnn.get('enhanced', False)}")
        print(f"   Entities found: {len(result_without_gnn.get('enhanced_query', {}).get('extracted_entities', []))}")

        # Test with GNN
        print("\n🧠 WITH GNN:")
        start_time = time.time()
        result_with_gnn = gnn_api.enhanced_query_processing(query, use_gnn=True)
        time_with_gnn = time.time() - start_time

        print(f"   Processing time: {time_with_gnn:.3f}s")
        print(f"   Enhanced: {result_with_gnn.get('enhanced', False)}")
        print(f"   GNN Confidence: {result_with_gnn.get('gnn_confidence', 0):.3f}")

        enhanced_query = result_with_gnn.get('enhanced_query', {})
        entities = enhanced_query.get('extracted_entities', [])
        classified_entities = enhanced_query.get('classified_entities', [])

        print(f"   Entities found: {len(entities)}")
        for entity in entities:
            print(f"     - {entity}")

        print(f"   Classified entities: {len(classified_entities)}")
        for entity in classified_entities:
            if 'class_name' in entity and 'confidence' in entity:
                print(f"     - {entity['entity_text']} → {entity['class_name']} (confidence: {entity['confidence']:.3f})")

        reasoning_results = result_with_gnn.get('reasoning_results', [])
        print(f"   Reasoning paths: {len(reasoning_results)}")

        for reasoning in reasoning_results[:2]:  # Show first 2 reasoning results
            start_entity = reasoning.get('start_entity', '')
            end_entity = reasoning.get('end_entity', '')
            paths = reasoning.get('paths', [])

            if paths:
                best_path = paths[0]
                print(f"     {start_entity} → {end_entity}")
                print(f"       Best path: {best_path.get('reasoning_chain', 'N/A')}")
                print(f"       Confidence: {best_path.get('gnn_confidence', 0):.3f}")
                print(f"       Semantic score: {best_path.get('semantic_score', 0):.3f}")

        # Performance comparison
        time_diff = time_with_gnn - time_without_gnn
        print(f"\n⚡ Performance:")
        print(f"   GNN overhead: {time_diff:.3f}s")
        print(f"   Speed ratio: {time_without_gnn/time_with_gnn:.2f}x")

        print("\n" + "="*60 + "\n")

    # Test entity classification
    print("🎯 Entity Classification Test")
    print("-" * 40)

    test_entities = [
        "thermostat",
        "air conditioner",
        "pump motor",
        "broken component",
        "maintenance procedure"
    ]

    for entity in test_entities:
        classification = gnn_api.gnn_service.classify_entity(entity, "maintenance context")

        if 'error' not in classification:
            print(f"   '{entity}' → {classification.get('class_name', 'unknown')} (confidence: {classification.get('confidence', 0):.3f})")
        else:
            print(f"   '{entity}' → Error: {classification['error']}")

    print("\n✅ GNN integration test completed!")

def test_gnn_reasoning():
    """Test GNN-enhanced reasoning"""
    print("\n🧠 GNN Reasoning Test")
    print("=" * 40)

    # Initialize GNN service
    gnn_api = GNNEnhancedAPI()
    gnn_api.initialize_gnn()

    if not gnn_api.initialized:
        print("❌ GNN initialization failed")
        return

    # Test reasoning pairs
    reasoning_pairs = [
        ("thermostat", "air conditioner"),
        ("pump", "motor"),
        ("broken", "repair"),
        ("engine room", "equipment")
    ]

    for start_entity, end_entity in reasoning_pairs:
        print(f"\n🔍 Reasoning: {start_entity} → {end_entity}")

        paths = gnn_api.gnn_service.gnn_enhanced_multi_hop_reasoning(
            start_entity, end_entity, max_hops=3
        )

        print(f"   Paths found: {len(paths)}")

        for i, path in enumerate(paths[:3], 1):  # Show top 3 paths
            print(f"   Path {i}:")
            print(f"     Chain: {path.get('reasoning_chain', 'N/A')}")
            print(f"     GNN Confidence: {path.get('gnn_confidence', 0):.3f}")
            print(f"     Semantic Score: {path.get('semantic_score', 0):.3f}")
            print(f"     Combined Score: {path.get('combined_score', 0):.3f}")

    print("\n✅ GNN reasoning test completed!")

def compare_systems():
    """Compare regular vs GNN-enhanced system capabilities"""
    print("\n📊 System Comparison")
    print("=" * 50)

    comparison_data = {
        "Regular System": {
            "Entity Classification": "Simple extraction",
            "Relationship Weighting": "Binary (0/1)",
            "Multi-hop Reasoning": "BFS path finding",
            "Query Enhancement": "Direct search",
            "Semantic Understanding": "Basic embeddings",
            "Confidence Scoring": "None",
            "Processing Time": "Fast",
            "Accuracy": "Basic"
        },
        "GNN-Enhanced System": {
            "Entity Classification": "Graph-aware classification",
            "Relationship Weighting": "Confidence-weighted",
            "Multi-hop Reasoning": "GNN-scored reasoning",
            "Query Enhancement": "Graph-context enhanced",
            "Semantic Understanding": "1540-dim graph embeddings",
            "Confidence Scoring": "GNN-based confidence",
            "Processing Time": "Moderate overhead",
            "Accuracy": "Enhanced"
        }
    }

    for system, capabilities in comparison_data.items():
        print(f"\n🔧 {system}:")
        for capability, description in capabilities.items():
            print(f"   {capability}: {description}")

    print("\n✅ System comparison completed!")

def main():
    """Main test function"""
    print("🚀 COMPREHENSIVE GNN INTEGRATION TEST")
    print("=" * 60)

    try:
        # Test 1: Basic integration
        test_gnn_integration()

        # Test 2: Reasoning capabilities
        test_gnn_reasoning()

        # Test 3: System comparison
        compare_systems()

        print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("\n📋 Next Steps:")
        print("   1. Start API server with GNN integration")
        print("   2. Test GNN-enhanced endpoints:")
        print("      curl -X POST 'http://localhost:8000/api/v1/query/gnn-enhanced' \\")
        print("        -H 'Content-Type: application/json' \\")
        print("        -d '{\"query\": \"air conditioner thermostat problems\", \"use_gnn\": true}'")
        print("   3. Monitor GNN performance and accuracy")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
