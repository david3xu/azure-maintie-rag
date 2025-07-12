import requests
import json
import time

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
        response = requests.post(
            f"{base_url}/api/v1/query",
            json=test_case,
            headers={"Content-Type": "application/json"},
            timeout=30  # Azure can be slower
        )
        end_time = time.time()

        print(f"📊 Status: {response.status_code}")
        print(f"📊 Response Time: {end_time - start_time:.2f}s")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Azure-powered query processed")
            print(f"📊 Response length: {len(data.get('generated_response', ''))}")
            print(f"📊 Sources found: {len(data.get('sources', []))}")
            print(f"📊 Safety warnings: {len(data.get('safety_warnings', []))}")

            # Check for Azure-specific indicators
            response_text = data.get('generated_response', '').lower()
            if any(keyword in response_text for keyword in ['maintenance', 'procedure', 'equipment']):
                print("✅ Domain-specific response generated")

        else:
            print(f"❌ Error: {response.text}")
            return False

    return True

if __name__ == "__main__":
    test_real_api_queries()
