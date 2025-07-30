#!/usr/bin/env python3
"""
Debug the workflow endpoint error
"""

import asyncio
import sys
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Add backend to Python path
sys.path.insert(0, '/workspace/azure-maintie-rag/backend')

def debug_workflow_endpoint():
    """Debug the workflow endpoint error"""
    print("🔧 DEBUGGING WORKFLOW ENDPOINT ERROR")
    print("=" * 50)
    
    try:
        from api.endpoints.workflow_endpoint import router, retrieve_workflow_evidence
        
        # Create test app
        test_app = FastAPI()
        test_app.include_router(router)
        client = TestClient(test_app)
        
        print("   ✅ Workflow router imported successfully")
        
        # Test the endpoint that's failing
        print("   🔍 Testing /api/v1/workflow/test_workflow_123/evidence...")
        response = client.get("/api/v1/workflow/test_workflow_123/evidence")
        
        print(f"   📊 Status code: {response.status_code}")
        
        if response.status_code != 200:
            try:
                error_detail = response.json()
                print(f"   📋 Error detail: {error_detail}")
            except:
                print(f"   📋 Response text: {response.text}")
        
        # Test the function directly
        print("\n   🔍 Testing retrieve_workflow_evidence function directly...")
        
        async def test_function():
            try:
                result = await retrieve_workflow_evidence("test_workflow_123")
                print(f"   ✅ Function result: {result}")
                return True
            except Exception as e:
                print(f"   ❌ Function error: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        function_result = asyncio.run(test_function())
        
        return response.status_code == 200 or function_result
        
    except Exception as e:
        print(f"   ❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_workflow_endpoint()