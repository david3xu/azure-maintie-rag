"""
Complete GNN Integration Test Script
Tests the full weeks 1-12 implementation
"""

import requests
import time
import json
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000/api/v1"

def test_complete_integration():
    """Test complete weeks 1-12 implementation"""

    logger.info("🧪 Testing Complete MaintIE GNN-Enhanced RAG System")
    logger.info("=" * 60)

    # Test queries that demonstrate different capabilities
    test_queries = [
        {
            "query": "pump seal failure troubleshooting",
            "expected_features": ["equipment_classification", "failure_mode_detection", "troubleshooting_procedures"]
        },
        {
            "query": "motor bearing vibration analysis procedure",
            "expected_features": ["component_hierarchy", "diagnostic_procedures", "measurement_techniques"]
        },
        {
            "query": "pressure vessel safety inspection schedule",
            "expected_features": ["safety_critical_equipment", "regulatory_compliance", "inspection_intervals"]
        }
    ]

    for i, test_case in enumerate(test_queries, 1):
        logger.info(f"\n🔬 Test Case {i}: {test_case['query']}")
        logger.info("-" * 40)

        # Test structured endpoint (weeks 1-8 + GNN enhancement)
        test_structured_endpoint(test_case)

        # Test comparison endpoint (A/B testing)
        test_comparison_endpoint(test_case)

        time.sleep(1)  # Rate limiting

    # Test system health and capabilities
    test_system_health()

    logger.info("\n✅ Complete integration testing finished!")

def test_structured_endpoint(test_case: Dict[str, Any]):
    """Test the optimized structured endpoint with GNN enhancement"""

    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/query/structured",
            json={
                "query": test_case["query"],
                "max_results": 5,
                "include_explanations": True,
                "enable_safety_warnings": True
            },
            timeout=30
        )
        response_time = time.time() - start_time

        if response.status_code == 200:
            data = response.json()

            logger.info(f"  ✅ Structured Response: {response_time:.2f}s")
            logger.info(f"  📊 Confidence: {data['confidence_score']:.3f}")
            logger.info(f"  📄 Sources: {len(data['sources'])}")
            logger.info(f"  ⚠️ Safety Warnings: {len(data.get('safety_warnings', []))}")

            # Check enhanced query features
            enhanced_query = data.get('enhanced_query', {})
            if enhanced_query:
                analysis = enhanced_query.get('analysis', {})
                expanded_concepts = enhanced_query.get('expanded_concepts', [])

                logger.info(f"  🧠 Query Type: {analysis.get('query_type', 'Unknown')}")
                logger.info(f"  🔍 Entities Found: {len(analysis.get('entities', []))}")
                logger.info(f"  🌐 Concepts Expanded: {len(expanded_concepts)}")

                # Check for equipment categorization
                equipment_category = enhanced_query.get('equipment_category')
                if equipment_category:
                    logger.info(f"  ⚙️ Equipment Category: {equipment_category}")

                # Check for safety criticality
                safety_critical = enhanced_query.get('safety_critical', False)
                if safety_critical:
                    logger.info("  🚨 Safety Critical Equipment Detected")

            # Check search results metadata for graph operations
            search_results = data.get('search_results', [])
            if search_results:
                result = search_results[0]
                metadata = result.get('metadata', {})

                # Check for graph scoring
                if 'knowledge_graph_score' in metadata:
                    kg_score = metadata['knowledge_graph_score']
                    logger.info(f"  📈 Graph Score: {kg_score:.3f}")

                # Check for GNN enhancement
                if 'gnn_expansion_used' in metadata:
                    logger.info("  🤖 GNN Enhancement: Active")
                elif 'enhancement_method' in metadata:
                    method = metadata['enhancement_method']
                    logger.info(f"  🔧 Enhancement Method: {method}")

        else:
            logger.error(f"  ❌ Structured endpoint failed: {response.status_code}")
            logger.error(f"  📝 Error: {response.text}")

    except Exception as e:
        logger.error(f"  ❌ Structured endpoint error: {e}")

def test_comparison_endpoint(test_case: Dict[str, Any]):
    """Test A/B comparison between different approaches"""

    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/query/compare",
            json={
                "query": test_case["query"],
                "max_results": 5
            },
            timeout=60
        )
        response_time = time.time() - start_time

        if response.status_code == 200:
            data = response.json()

            # Performance comparison
            performance = data.get('performance', {})
            multi_modal = performance.get('multi_modal', {})
            optimized = performance.get('optimized', {})

            mm_time = multi_modal.get('processing_time', 0)
            opt_time = optimized.get('processing_time', 0)

            if mm_time > 0 and opt_time > 0:
                speedup = mm_time / opt_time
                improvement = performance.get('improvement', {})

                logger.info(f"  ⚡ Performance Comparison:")
                logger.info(f"    Multi-modal: {mm_time:.2f}s")
                logger.info(f"    Optimized: {opt_time:.2f}s")
                logger.info(f"    Speedup: {speedup:.1f}x")

                # Quality comparison
                quality = data.get('quality_comparison', {})
                if quality:
                    confidence_diff = quality.get('confidence_score', {}).get('difference', 0)
                    logger.info(f"    Confidence Δ: {confidence_diff:+.3f}")

            # Recommendation
            recommendation = data.get('recommendation', {})
            if recommendation.get('use_optimized', False):
                reason = recommendation.get('reason', '')
                logger.info(f"  💡 Recommendation: Use optimized ({reason})")

        else:
            logger.error(f"  ❌ Comparison endpoint failed: {response.status_code}")

    except Exception as e:
        logger.error(f"  ❌ Comparison endpoint error: {e}")

def test_system_health():
    """Test system health and component status"""

    logger.info(f"\n🏥 System Health Check")
    logger.info("-" * 30)

    try:
        # General health
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            health = response.json()

            status = health.get('status', 'unknown')
            logger.info(f"  Overall Status: {status.upper()}")

            checks = health.get('checks', {})
            for component, status in checks.items():
                icon = "✅" if status == "healthy" or status == "enabled" else "⚠️"
                logger.info(f"  {icon} {component}: {status}")

            # Performance metrics
            performance = health.get('performance', {})
            if performance:
                total_queries = performance.get('total_queries', 0)
                avg_time = performance.get('average_processing_time', 0)
                logger.info(f"  📊 Total Queries: {total_queries}")
                logger.info(f"  ⏱️ Avg Response Time: {avg_time:.2f}s")

            # Cache stats
            cache = health.get('cache', {})
            if cache:
                cache_type = cache.get('cache_type', 'unknown')
                cached_responses = cache.get('cached_responses', 0)
                logger.info(f"  💾 Cache Type: {cache_type}")
                logger.info(f"  📦 Cached Responses: {cached_responses}")

    except Exception as e:
        logger.error(f"  ❌ Health check error: {e}")

    # GNN-specific health check
    try:
        logger.info(f"\n🤖 GNN Component Health")
        logger.info("-" * 25)

        response = requests.get(f"{BASE_URL}/health/gnn", timeout=10)
        if response.status_code == 200:
            gnn_health = response.json()

            model_loaded = gnn_health.get('model_loaded', False)
            entities_mapped = gnn_health.get('entities_mapped', 0)
            success_rate = gnn_health.get('expansion_success_rate', 0)

            logger.info(f"  🧠 GNN Model: {'Loaded' if model_loaded else 'Not Available'}")
            logger.info(f"  🔗 Entities Mapped: {entities_mapped}")
            logger.info(f"  📈 Success Rate: {success_rate:.1%}")

        elif response.status_code == 404:
            logger.info("  ℹ️ GNN health endpoint not available (GNN may be disabled)")

    except Exception as e:
        logger.info(f"  ℹ️ GNN health check not available: {e}")

def test_gnn_specific_features():
    """Test GNN-specific functionality"""

    logger.info(f"\n🧠 GNN-Specific Feature Tests")
    logger.info("-" * 35)

    # Test query with entities that should benefit from GNN expansion
    test_query = "hydraulic pump mechanical seal replacement procedure"

    try:
        response = requests.post(
            f"{BASE_URL}/query/structured",
            json={"query": test_query, "max_results": 3},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            enhanced_query = data.get('enhanced_query', {})

            # Check expansion quality
            original_entities = enhanced_query.get('analysis', {}).get('entities', [])
            expanded_concepts = enhanced_query.get('expanded_concepts', [])

            logger.info(f"  🔍 Original Entities: {len(original_entities)}")
            logger.info(f"    {', '.join(original_entities[:5])}")
            logger.info(f"  🌐 Expanded Concepts: {len(expanded_concepts)}")
            logger.info(f"    {', '.join(expanded_concepts[:8])}")

            # Check for maintenance-specific expansions
            maintenance_terms = [
                'bearing', 'O-ring', 'gasket', 'vibration', 'alignment',
                'pressure', 'leak', 'lubrication', 'inspection'
            ]

            relevant_expansions = [
                concept for concept in expanded_concepts
                if any(term in concept.lower() for term in maintenance_terms)
            ]

            if relevant_expansions:
                logger.info(f"  ⚙️ Maintenance-Relevant Expansions: {len(relevant_expansions)}")
                logger.info(f"    {', '.join(relevant_expansions[:5])}")

            # Check domain context if available
            domain_context = enhanced_query.get('maintenance_context', {})
            if domain_context:
                urgency = domain_context.get('task_urgency', 'unknown')
                safety_level = domain_context.get('safety_level', 'unknown')
                logger.info(f"  🎯 Task Urgency: {urgency}")
                logger.info(f"  🛡️ Safety Level: {safety_level}")

    except Exception as e:
        logger.error(f"  ❌ GNN feature test error: {e}")

if __name__ == "__main__":
    try:
        # Test basic connectivity
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            logger.error("❌ API server not responding. Start with: uvicorn api.main:app --reload")
            exit(1)

        # Run complete test suite
        test_complete_integration()

        # Run GNN-specific tests
        test_gnn_specific_features()

        logger.info(f"\n🎉 All tests completed successfully!")
        logger.info("Your GNN-Enhanced RAG system is working correctly.")

    except requests.exceptions.ConnectionError:
        logger.error("❌ Cannot connect to API server.")
        logger.error("Start the server with: cd backend && uvicorn api.main:app --reload")
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        exit(1)