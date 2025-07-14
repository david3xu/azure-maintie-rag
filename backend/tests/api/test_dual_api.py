#!/usr/bin/env python3
"""
Test script for dual API approach - comparing multi-modal vs structured RAG

This script demonstrates:
1. Both API endpoints working side-by-side
2. Performance comparison between methods
3. A/B testing capabilities
4. Quality comparison metrics

Usage:
    python -m pytest backend/tests/test_dual_api.py -v
    python backend/tests/test_dual_api.py
"""

import asyncio
import time
import requests
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# API configuration
BASE_URL = "http://localhost:8000/api/v1"
TEST_QUERIES = [
    "pump seal failure troubleshooting",
    "motor bearing replacement procedure",
    "compressor vibration analysis",
    "valve maintenance schedule",
    "electrical safety procedures"
]

def test_individual_endpoints():
    """Test both endpoints individually"""
    print("🔍 Testing Individual Endpoints")
    print("=" * 50)

    for i, query in enumerate(TEST_QUERIES[:2], 1):  # Test first 2 queries
        print(f"\n📝 Test {i}: {query}")

        # Test multi-modal endpoint
        print("  🔄 Testing multi-modal endpoint...")
        start_time = time.time()
        try:
            response = requests.post(
                f"{BASE_URL}/query/multi-modal",
                json={"query": query, "max_results": 5},
                timeout=30
            )
            multi_modal_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                print(f"    ✅ Multi-modal: {multi_modal_time:.2f}s, Confidence: {data['confidence_score']:.3f}")
            else:
                print(f"    ❌ Multi-modal failed: {response.status_code}")
                continue

        except Exception as e:
            print(f"    ❌ Multi-modal error: {e}")
            continue

        # Test structured endpoint
        print("  ⚡ Testing structured endpoint...")
        start_time = time.time()
        try:
            response = requests.post(
                f"{BASE_URL}/query/structured",
                json={"query": query, "max_results": 5},
                timeout=30
            )
            structured_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                print(f"    ✅ Structured: {structured_time:.2f}s, Confidence: {data['confidence_score']:.3f}")

                # Calculate improvement
                if multi_modal_time > 0:
                    improvement = ((multi_modal_time - structured_time) / multi_modal_time) * 100
                    speedup = multi_modal_time / structured_time
                    print(f"    📊 Improvement: {improvement:.1f}% faster ({speedup:.1f}x speedup)")

            else:
                print(f"    ❌ Structured failed: {response.status_code}")

        except Exception as e:
            print(f"    ❌ Structured error: {e}")

def test_comparison_endpoint():
    """Test the comparison endpoint for A/B testing"""
    print("\n🔬 Testing Comparison Endpoint (A/B Testing)")
    print("=" * 50)

    for i, query in enumerate(TEST_QUERIES[:2], 1):
        print(f"\n📝 Comparison Test {i}: {query}")

        try:
            response = requests.post(
                f"{BASE_URL}/query/compare",
                json={"query": query, "max_results": 5},
                timeout=60  # Longer timeout for comparison
            )

            if response.status_code == 200:
                data = response.json()

                # Display performance comparison
                perf = data['performance']
                print(f"  ⏱️  Performance:")
                print(f"    Multi-modal: {perf['multi_modal']['processing_time']:.2f}s ({perf['multi_modal']['api_calls_estimated']} API calls)")
                print(f"    Structured: {perf['optimized']['processing_time']:.2f}s ({perf['optimized']['api_calls_estimated']} API calls)")
                print(f"    Improvement: {perf['improvement']['time_reduction_percent']:.1f}% faster, {perf['improvement']['speedup_factor']:.1f}x speedup")

                # Display quality comparison
                quality = data['quality_comparison']
                print(f"  📊 Quality:")
                print(f"    Confidence: {quality['confidence_score']['multi_modal']:.3f} → {quality['confidence_score']['optimized']:.3f} ({quality['confidence_score']['difference']:+.3f})")
                print(f"    Results: {quality['search_results_count']['multi_modal']} → {quality['search_results_count']['optimized']}")
                print(f"    Safety warnings: {quality['safety_warnings_count']['multi_modal']} → {quality['safety_warnings_count']['optimized']}")

                # Display recommendation
                rec = data['recommendation']
                print(f"  💡 Recommendation: {'✅ Use structured' if rec['use_optimized'] else '⚠️  Consider multi-modal'}")
                print(f"    Reason: {rec['reason']}")

            else:
                print(f"  ❌ Comparison failed: {response.status_code}")
                print(f"    Response: {response.text}")

        except Exception as e:
            print(f"  ❌ Comparison error: {e}")

def test_api_health():
    """Test API health and endpoints availability"""
    print("🏥 Testing API Health")
    print("=" * 50)

    try:
        # Test health endpoint
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health endpoint: OK")
        else:
            print(f"❌ Health endpoint: {response.status_code}")

        # Test root endpoint
        response = requests.get("http://localhost:8000/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API root: {data.get('message', 'Unknown')}")
            print(f"   Version: {data.get('version', 'Unknown')}")
        else:
            print(f"❌ API root: {response.status_code}")

    except Exception as e:
        print(f"❌ Health check error: {e}")

def test_endpoint_availability():
    """Test that all endpoints are available"""
    print("\n🔗 Testing Endpoint Availability")
    print("=" * 50)

    endpoints = [
        ("Multi-modal", f"{BASE_URL}/query/multi-modal"),
        ("Structured", f"{BASE_URL}/query/structured"),
        ("Comparison", f"{BASE_URL}/query/compare"),
        ("Health", f"{BASE_URL}/health"),
        ("Docs", "http://localhost:8000/docs")
    ]

    for name, url in endpoints:
        try:
            if "docs" in url:
                response = requests.get(url, timeout=5)
            else:
                response = requests.post(url, json={"query": "test"}, timeout=5)

            if response.status_code in [200, 405]:  # 405 is OK for POST to GET endpoints
                print(f"✅ {name}: Available")
            else:
                print(f"⚠️  {name}: Status {response.status_code}")

        except Exception as e:
            print(f"❌ {name}: {str(e)[:50]}...")

def main():
    """Main test function"""
    print("🚀 MaintIE Dual API Test Suite")
    print("Testing multi-modal vs structured RAG approaches")
    print("=" * 60)

    # Test API health first
    test_api_health()

    # Test endpoint availability
    test_endpoint_availability()

    # Test individual endpoints
    test_individual_endpoints()

    # Test comparison endpoint
    test_comparison_endpoint()

    print("\n" + "=" * 60)
    print("✅ Test suite completed!")
    print("\n📋 Available endpoints:")
    print("  POST /api/v1/query/multi-modal - Original multi-modal RAG")
    print("  POST /api/v1/query/structured  - Optimized structured RAG")
    print("  POST /api/v1/query/compare     - A/B testing comparison")
    print("  GET  /api/v1/health            - System health check")
    print("\n🔗 API Documentation: http://localhost:8000/docs")

if __name__ == "__main__":
    main()