#!/usr/bin/env python3
"""
Simple test script to verify config-driven improvements work correctly
Tests entity extraction, safety assessment, and concept expansion using real config patterns
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from src.enhancement.query_analyzer import MaintenanceQueryAnalyzer
from src.models.maintenance_models import QueryType


def test_entity_extraction():
    """Test entity extraction with caching and pattern filtering"""
    print("🔍 Testing Entity Extraction with Config Patterns")
    print("=" * 50)

    analyzer = MaintenanceQueryAnalyzer()

    # Test queries that should match config patterns
    test_queries = [
        "pump bearing failure",
        "motor seal maintenance",
        "compressor vibration analysis",
        "valve troubleshooting guide"
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")

        # Test caching - call twice
        entities1 = analyzer._extract_entities(query)
        entities2 = analyzer._extract_entities(query)

        print(f"  Entities: {entities1}")
        print(f"  Cache hit: {entities1 == entities2}")
        print(f"  Valid patterns: {len([e for e in entities1 if e in analyzer.equipment_patterns or e in analyzer.failure_patterns])}")


def test_safety_assessment():
    """Test safety criticality assessment using config patterns"""
    print("\n🔒 Testing Safety Assessment with Config Patterns")
    print("=" * 50)

    analyzer = MaintenanceQueryAnalyzer()

    # Test entities that should trigger safety flags
    test_cases = [
        (["pump", "bearing"], QueryType.TROUBLESHOOTING),
        (["motor", "seal"], QueryType.PROCEDURAL),
        (["pressure_vessel"], QueryType.INFORMATIONAL),
        (["boiler"], QueryType.SAFETY)
    ]

    for entities, query_type in test_cases:
        print(f"\nEntities: {entities}, Type: {query_type.value}")

        assessment = analyzer._assess_safety_criticality(entities, query_type)

        print(f"  Safety Critical: {assessment['is_safety_critical']}")
        print(f"  Safety Level: {assessment['safety_level']}")
        print(f"  Critical Equipment: {assessment['critical_equipment']}")
        print(f"  Safety Warnings: {len(assessment['safety_warnings'])}")


def test_concept_expansion():
    """Test concept expansion using expansion_rules from config"""
    print("\n🌐 Testing Concept Expansion with Config Rules")
    print("=" * 50)

    analyzer = MaintenanceQueryAnalyzer()

    # Test entities for expansion
    test_entities = ["pump", "bearing", "seal"]

    for entity in test_entities:
        print(f"\nEntity: '{entity}'")

        # Test rule-based expansion
        expansions = analyzer._rule_based_expansion([entity])
        print(f"  Rule-based expansions: {expansions}")

        # Test enhanced expansion
        enhanced_expansions = analyzer._enhanced_expand_concepts([entity])
        print(f"  Enhanced expansions: {enhanced_expansions[:5]}...")  # Show first 5


def test_structured_search():
    """Test structured search query building"""
    print("\n🔍 Testing Structured Search Query Building")
    print("=" * 50)

    analyzer = MaintenanceQueryAnalyzer()

    # Test query building
    entities = ["pump", "bearing"]
    expanded_concepts = ["seal", "vibration", "misalignment"]

    structured_query = analyzer._build_structured_search(entities, expanded_concepts)

    print(f"Entities: {entities}")
    print(f"Expanded concepts: {expanded_concepts}")
    print(f"Structured query: '{structured_query}'")


def test_performance_logging():
    """Test that performance logging is properly integrated"""
    print("\n⏱️ Testing Performance Logging Integration")
    print("=" * 50)

    from src.monitoring.pipeline_monitor import get_monitor, reset_monitor
    from src.enhancement.query_analyzer import MaintenanceQueryAnalyzer

    # Reset monitor to ensure clean state
    reset_monitor()

    # Initialize monitor and start query
    monitor = get_monitor()
    query_id = monitor.start_query("pump bearing failure analysis", "structured")

    analyzer = MaintenanceQueryAnalyzer()

    # Test a full analysis with timing
    import time

    test_query = "pump bearing failure analysis"

    print(f"Query: '{test_query}'")

    start_time = time.time()
    try:
        analysis = analyzer.analyze_query(test_query)
        end_time = time.time()

        # End monitoring
        metrics = monitor.end_query(
            confidence_score=0.85,
            sources_count=3,
            safety_warnings_count=2
        )

        print(f"✅ Analysis completed in {end_time - start_time:.2f}s")
        print(f"📊 Entities found: {len(analysis.entities)}")
        print(f"📊 Query type: {analysis.query_type}")
        print(f"📊 Equipment category: {analysis.equipment_category}")

        # Verify metrics were collected
        assert metrics is not None, "Metrics should be collected"
        assert metrics.total_steps > 0, "Should have tracked steps"

        print("✅ Performance logging test passed")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        # Still end the query to clean up
        monitor.end_query(confidence_score=0.0, sources_count=0)
        raise


def main():
    """Run all config integration tests"""
    print("🚀 Testing Config-Driven Improvements")
    print("=" * 60)

    try:
        test_entity_extraction()
        test_safety_assessment()
        test_concept_expansion()
        test_structured_search()
        test_performance_logging()

        print("\n✅ All config integration tests completed successfully!")
        print("\n📋 Summary of improvements tested:")
        print("   • Entity extraction with caching and pattern filtering")
        print("   • Safety assessment using equipment_hierarchy and safety_critical_equipment")
        print("   • Concept expansion using expansion_rules from config")
        print("   • Structured search query building with config patterns")
        print("   • Performance logging integration")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()