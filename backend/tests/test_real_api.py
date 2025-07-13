import requests
import json
import time
import pytest # Added import for pytest

def test_real_api_queries():
    """Test real API with Azure OpenAI backend"""
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

    print("🔄 Testing Real API (Azure Backend)...")

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Real Test Case {i}: {test_case['query']}")

        start_time = time.time()
        try:
            response = requests.post(
                f"{base_url}/api/v1/query",
                json=test_case,
                headers={"Content-Type": "application/json"},
                timeout=30  # Azure can be slower
            )
            end_time = time.time()

            print(f"📊 Status: {response.status_code}")
            print(f"📊 Response Time: {end_time - start_time:.2f}s")

            assert response.status_code == 200, f"API returned non-200 status: {response.status_code} - {response.text}"
            data = response.json()

            assert "generated_response" in data, "Missing 'generated_response' field in API response"
            assert len(data["generated_response"]) > 0, "Empty 'generated_response' field"
            assert "sources" in data, "Missing 'sources' field in API response"
            assert "confidence_score" in data, "Missing 'confidence_score' field in API response"

            print(f"✅ Azure-powered query processed")
            print(f"📊 Response length: {len(data.get('generated_response', ''))}")
            print(f"📊 Sources found: {len(data.get('sources', []))}")
            print(f"📊 Safety warnings: {len(data.get('safety_warnings', []))}")

            # Check for Azure-specific indicators
            response_text = data.get('generated_response', '').lower()
            assert any(keyword in response_text for keyword in ['maintenance', 'procedure', 'equipment']), "Domain-specific response not generated"

        except requests.exceptions.ConnectionError as ce:
            pytest.fail(f"API Connection Error for '{test_case['query']}': Is the API server running? {ce}")
        except requests.exceptions.Timeout as te:
            pytest.fail(f"API Timeout Error for '{test_case['query']}': {te}")
        except json.JSONDecodeError as je:
            pytest.fail(f"API JSON Decode Error for '{test_case['query']}': {je} - Response Text: {response.text}")
        except Exception as e:
            pytest.fail(f"API query test failed for '{test_case['query']}': {e}")

if __name__ == "__main__":
    test_real_api_queries()
