#!/usr/bin/env python3
"""
Debug script for testing granular pipeline monitoring
Demonstrates detailed sub-step tracking and performance metrics
"""

import sys
import time
import json
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline.rag_structured import MaintIEStructuredRAG
from src.monitoring.pipeline_monitor import get_monitor, reset_monitor


def test_monitoring_system():
    """Test the granular monitoring system with a real query"""
    print("🔍 Testing Granular Pipeline Monitoring System")
    print("=" * 60)

    # Reset monitor for clean test
    reset_monitor()

    # Initialize RAG system
    print("\n1. Initializing RAG System...")
    rag_system = MaintIEStructuredRAG()

    # Test query
    test_query = "pump bearing failure analysis"
    print(f"\n2. Processing Query: '{test_query}'")

    try:
        # Process query with monitoring
        start_time = time.time()
        response = rag_system.process_query(test_query)
        end_time = time.time()

        print(f"\n3. Query Processing Results:")
        print(f"   • Processing Time: {response.processing_time:.2f}s")
        print(f"   • Confidence Score: {response.confidence_score:.3f}")
        print(f"   • Sources Found: {len(response.sources)}")
        print(f"   • Safety Warnings: {len(response.safety_warnings)}")

        # Get detailed metrics
        monitor = get_monitor()
        metrics = monitor.get_performance_summary()

        print(f"\n4. Granular Performance Metrics:")
        print(f"   • Total Steps: {metrics['summary']['total_steps']}")
        print(f"   • Successful Steps: {metrics['summary']['successful_steps']}")
        print(f"   • Failed Steps: {metrics['summary']['failed_steps']}")
        print(f"   • Total API Calls: {metrics['summary']['total_api_calls']}")
        print(f"   • Cache Hits: {metrics['summary']['cache_hits']}")

        # Detailed step breakdown
        print(f"\n5. Detailed Step Breakdown:")
        for step_name, step_metrics in metrics['step_performance'].items():
            print(f"   • {step_name}:")
            print(f"     - Total Calls: {step_metrics['total_calls']}")
            print(f"     - Avg Duration: {step_metrics['avg_duration_ms']:.1f}ms")
            print(f"     - API Calls: {step_metrics['api_calls']}")
            print(f"     - Cache Hits: {step_metrics['cache_hits']}")

        # Show sample response
        print(f"\n6. Sample Response (first 200 chars):")
        print(f"   {response.generated_response[:200]}...")

        return True

    except Exception as e:
        print(f"❌ Error during monitoring test: {e}")
        return False


def test_monitoring_with_multiple_queries():
    """Test monitoring with multiple queries to show patterns"""
    print("\n🔍 Testing Monitoring with Multiple Queries")
    print("=" * 60)

    # Reset monitor
    reset_monitor()

    # Initialize RAG system
    rag_system = MaintIEStructuredRAG()

    # Test queries
    test_queries = [
        "pump seal failure",
        "motor bearing maintenance",
        "valve troubleshooting guide"
    ]

    results = []

    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Processing: '{query}'")

        try:
            # Reset monitor for each query
            reset_monitor()

            # Process query
            response = rag_system.process_query(query)

            # Get metrics
            monitor = get_monitor()
            metrics = monitor.get_performance_summary()

            result = {
                'query': query,
                'processing_time': response.processing_time,
                'confidence': response.confidence_score,
                'sources': len(response.sources),
                'total_steps': metrics['summary']['total_steps'],
                'api_calls': metrics['summary']['total_api_calls'],
                'cache_hits': metrics['summary']['cache_hits']
            }

            results.append(result)

            print(f"   ✅ Completed in {response.processing_time:.2f}s")
            print(f"   📊 Steps: {result['total_steps']}, API calls: {result['api_calls']}")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'query': query,
                'error': str(e)
            })

    # Summary
    print(f"\n📈 Summary of {len(results)} Queries:")
    successful = [r for r in results if 'error' not in r]
    if successful:
        avg_time = sum(r['processing_time'] for r in successful) / len(successful)
        avg_steps = sum(r['total_steps'] for r in successful) / len(successful)
        avg_api_calls = sum(r['api_calls'] for r in successful) / len(successful)

        print(f"   • Average Processing Time: {avg_time:.2f}s")
        print(f"   • Average Steps: {avg_steps:.1f}")
        print(f"   • Average API Calls: {avg_api_calls:.1f}")
        print(f"   • Success Rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")


def test_monitoring_file_output():
    """Test that monitoring files are being saved correctly"""
    print("\n🔍 Testing Monitoring File Output")
    print("=" * 60)

    # Reset monitor
    reset_monitor()

    # Initialize RAG system
    rag_system = MaintIEStructuredRAG()

    # Process a query
    test_query = "compressor maintenance procedure"
    print(f"Processing: '{test_query}'")

    try:
        response = rag_system.process_query(test_query)

        # Check for metrics files
        metrics_dir = Path("data/metrics")
        if metrics_dir.exists():
            metric_files = list(metrics_dir.glob("pipeline_metrics_*.json"))
            print(f"\n📁 Found {len(metric_files)} metric files:")

            # Show most recent file
            if metric_files:
                latest_file = max(metric_files, key=lambda f: f.stat().st_mtime)
                print(f"   Latest: {latest_file.name}")

                # Read and display sample metrics
                with open(latest_file, 'r') as f:
                    metrics_data = json.load(f)

                print(f"\n📊 Sample Metrics from {latest_file.name}:")
                print(f"   • Query ID: {metrics_data['query_id'][:8]}...")
                print(f"   • Total Duration: {metrics_data['total_duration_ms']:.1f}ms")
                print(f"   • Total Steps: {metrics_data['total_steps']}")
                print(f"   • API Calls: {metrics_data['total_api_calls']}")
                print(f"   • Cache Hits: {metrics_data['cache_hits']}")

                # Show sub-steps
                if metrics_data.get('sub_steps'):
                    print(f"\n🔍 Sub-steps tracked:")
                    for step in metrics_data['sub_steps'][:5]:  # Show first 5
                        print(f"   • {step['step_name']}: {step['duration_ms']:.1f}ms")
                    if len(metrics_data['sub_steps']) > 5:
                        print(f"   ... and {len(metrics_data['sub_steps']) - 5} more steps")
        else:
            print("❌ No metrics directory found")

    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run all monitoring tests"""
    print("🚀 MaintIE Enhanced RAG - Granular Monitoring System Test")
    print("=" * 70)

    # Test 1: Basic monitoring
    success1 = test_monitoring_system()

    # Test 2: Multiple queries
    test_monitoring_with_multiple_queries()

    # Test 3: File output
    test_monitoring_file_output()

    print(f"\n{'='*70}")
    if success1:
        print("✅ Monitoring system test completed successfully!")
        print("\n📋 Key Features Demonstrated:")
        print("   • Granular sub-step tracking with timing")
        print("   • API call monitoring and counting")
        print("   • Cache hit tracking")
        print("   • Custom metrics per step")
        print("   • Performance summary generation")
        print("   • Metrics file persistence")
        print("   • Error handling and recovery")
    else:
        print("❌ Monitoring system test failed!")

    print(f"\n📁 Check 'data/metrics/' directory for detailed metric files")
    print("📊 Use the monitoring system in production for:")
    print("   • Performance bottleneck identification")
    print("   • Error isolation and debugging")
    print("   • Cache performance analysis")
    print("   • API usage optimization")


if __name__ == "__main__":
    main()