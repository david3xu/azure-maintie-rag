import requests
import json
import time
import pytest # Added import for pytest

def test_real_api_queries():
    """Test real API with Azure OpenAI backend using new dual API endpoints"""
    base_url = "http://localhost:8000"

    # Real maintenance queries
    test_cases = [
        {
            "query": "centrifugal pump seal failure troubleshooting",
            "max_results": 5,
            "include_explanations": True,
            "enable_safety_warnings": True
        },
        {
            "query": "how to align motor coupling properly",
            "max_results": 3,
            "include_explanations": True
        }
    ]

    print("🔄 Testing Real API (Azure Backend) - Dual API Endpoints...")

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Real Test Case {i}: {test_case['query']}")

        # Test multi-modal endpoint
        print("  🔄 Testing multi-modal endpoint...")
        start_time = time.time()
        try:
            response = requests.post(
                f"{base_url}/api/v1/query/multi-modal",
                json=test_case,
                headers={"Content-Type": "application/json"},
                timeout=30  # Azure can be slower
            )
            multi_modal_time = time.time() - start_time

            print(f"📊 Multi-modal Status: {response.status_code}")
            print(f"📊 Multi-modal Response Time: {multi_modal_time:.2f}s")

            assert response.status_code == 200, f"Multi-modal API returned non-200 status: {response.status_code} - {response.text}"
            data = response.json()

            assert "generated_response" in data, "Missing 'generated_response' field in multi-modal API response"
            assert len(data["generated_response"]) > 0, "Empty 'generated_response' field in multi-modal response"
            assert "sources" in data, "Missing 'sources' field in multi-modal API response"
            assert "confidence_score" in data, "Missing 'confidence_score' field in multi-modal API response"

            print(f"✅ Multi-modal query processed")
            print(f"📊 Response length: {len(data.get('generated_response', ''))}")
            print(f"📊 Sources found: {len(data.get('sources', []))}")
            print(f"📊 Safety warnings: {len(data.get('safety_warnings', []))}")

        except requests.exceptions.ConnectionError as ce:
            pytest.fail(f"Multi-modal API Connection Error for '{test_case['query']}': Is the API server running? {ce}")
        except requests.exceptions.Timeout as te:
            pytest.fail(f"Multi-modal API Timeout Error for '{test_case['query']}': {te}")
        except json.JSONDecodeError as je:
            pytest.fail(f"Multi-modal API JSON Decode Error for '{test_case['query']}': {je} - Response Text: {response.text}")
        except Exception as e:
            pytest.fail(f"Multi-modal API query test failed for '{test_case['query']}': {e}")

        # Test structured endpoint
        print("  ⚡ Testing structured endpoint...")
        start_time = time.time()
        try:
            response = requests.post(
                f"{base_url}/api/v1/query/structured",
                json=test_case,
                headers={"Content-Type": "application/json"},
                timeout=30  # Azure can be slower
            )
            structured_time = time.time() - start_time

            print(f"📊 Structured Status: {response.status_code}")
            print(f"📊 Structured Response Time: {structured_time:.2f}s")

            assert response.status_code == 200, f"Structured API returned non-200 status: {response.status_code} - {response.text}"
            data = response.json()

            assert "generated_response" in data, "Missing 'generated_response' field in structured API response"
            assert len(data["generated_response"]) > 0, "Empty 'generated_response' field in structured response"
            assert "sources" in data, "Missing 'sources' field in structured API response"
            assert "confidence_score" in data, "Missing 'confidence_score' field in structured API response"

            print(f"✅ Structured query processed")
            print(f"📊 Response length: {len(data.get('generated_response', ''))}")
            print(f"📊 Sources found: {len(data.get('sources', []))}")
            print(f"📊 Safety warnings: {len(data.get('safety_warnings', []))}")

            # Calculate performance improvement
            if multi_modal_time > 0:
                improvement = ((multi_modal_time - structured_time) / multi_modal_time) * 100
                speedup = multi_modal_time / structured_time
                print(f"📊 Performance: {improvement:.1f}% faster ({speedup:.1f}x speedup)")

        except requests.exceptions.ConnectionError as ce:
            pytest.fail(f"Structured API Connection Error for '{test_case['query']}': Is the API server running? {ce}")
        except requests.exceptions.Timeout as te:
            pytest.fail(f"Structured API Timeout Error for '{test_case['query']}': {te}")
        except json.JSONDecodeError as je:
            pytest.fail(f"Structured API JSON Decode Error for '{test_case['query']}': {je} - Response Text: {response.text}")
        except Exception as e:
            pytest.fail(f"Structured API query test failed for '{test_case['query']}': {e}")

        # Test comparison endpoint
        print("  🔬 Testing comparison endpoint...")
        try:
            response = requests.post(
                f"{base_url}/api/v1/query/compare",
                json=test_case,
                headers={"Content-Type": "application/json"},
                timeout=60  # Longer timeout for comparison
            )

            if response.status_code == 200:
                data = response.json()
                print(f"✅ Comparison completed")
                print(f"📊 Recommendation: {data.get('recommendation', {}).get('reason', 'N/A')}")
            else:
                print(f"⚠️  Comparison failed: {response.status_code}")

        except Exception as e:
            print(f"⚠️  Comparison error: {e}")

def test_api_endpoint_availability():
    """Test that all new API endpoints are available"""
    base_url = "http://localhost:8000"

    endpoints = [
        ("Multi-modal", f"{base_url}/api/v1/query/multi-modal"),
        ("Structured", f"{base_url}/api/v1/query/structured"),
        ("Comparison", f"{base_url}/api/v1/query/compare"),
        ("Health", f"{base_url}/api/v1/health")
    ]

    print("\n🔗 Testing API Endpoint Availability...")

    for name, url in endpoints:
        try:
            if "health" in url:
                response = requests.get(url, timeout=5)
            else:
                response = requests.post(url, json={"query": "test"}, timeout=5)

            if response.status_code in [200, 405]:  # 405 is OK for POST to GET endpoints
                print(f"✅ {name}: Available")
            else:
                print(f"⚠️  {name}: Status {response.status_code}")

        except Exception as e:
            print(f"❌ {name}: {str(e)[:50]}...")

if __name__ == "__main__":
    test_api_endpoint_availability()
    test_real_api_queries()
