#!/usr/bin/env python3
"""
Direct test of health endpoint functionality
Tests the endpoint logic without starting full FastAPI server
"""

import asyncio
import sys
import os

# Add backend to Python path
sys.path.insert(0, '/workspace/azure-maintie-rag/backend')

async def test_health_endpoint_direct():
    """Test health endpoint functions directly"""
    print("🏥 Testing Health Endpoint Logic")
    print("=" * 50)
    
    try:
        # Import health endpoint functions
        from api.endpoints.health_endpoint import health_check, basic_health_check, detailed_health_check
        
        print("   ✅ Health endpoint functions imported successfully")
        
        # Test basic health check
        print("\n📋 Testing basic health check...")
        basic_result = await basic_health_check()
        print(f"   ✅ Basic health: {basic_result.body.decode()}")
        
        # Test main health check
        print("\n🔍 Testing main health check...")
        health_result = await health_check()
        
        # Validate health check structure
        assert "status" in health_result
        assert "timestamp" in health_result
        assert "components" in health_result
        assert "capabilities" in health_result
        
        print(f"   ✅ Main health check: {health_result['status']}")
        print(f"   📊 Response time: {health_result['response_time_ms']}ms")
        print(f"   🔧 Components: {len(health_result['components'])} checked")
        
        # Test detailed health check  
        print("\n🔬 Testing detailed health check...")
        detailed_result = await detailed_health_check()
        
        # Validate detailed health structure
        assert "overall_status" in detailed_result
        assert "component_diagnostics" in detailed_result
        assert "performance_metrics" in detailed_result
        
        print(f"   ✅ Detailed health: {detailed_result['overall_status']}")
        print(f"   📊 Components diagnosed: {len(detailed_result['component_diagnostics'])}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Health endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_health_endpoint_with_fastapi():
    """Test health endpoint with FastAPI test client"""
    print("\n🚀 Testing Health Endpoint with FastAPI")
    print("=" * 50)
    
    try:
        from fastapi.testclient import TestClient
        from api.endpoints.health_endpoint import router
        from fastapi import FastAPI
        
        # Create minimal test app
        test_app = FastAPI()
        test_app.include_router(router)
        
        client = TestClient(test_app)
        
        # Test basic health endpoint
        print("   📋 Testing /api/v1/health...")
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        
        print(f"   ✅ Basic health endpoint: {response.status_code} - {data['message']}")
        
        # Test main health endpoint
        print("   🔍 Testing /health...")
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        
        print(f"   ✅ Main health endpoint: {response.status_code} - {data['status']}")
        print(f"   🔧 System: {data['system']} v{data['version']}")
        
        # Test detailed health endpoint
        print("   🔬 Testing /health/detailed...")
        response = client.get("/health/detailed")
        
        assert response.status_code == 200
        data = response.json()
        assert "overall_status" in data
        
        print(f"   ✅ Detailed health endpoint: {response.status_code} - {data['overall_status']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ FastAPI health test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run health endpoint tests"""
    print("🏥 HEALTH ENDPOINT TESTING")
    print("=" * 70)
    print("Testing health endpoints after architecture fixes")
    print("=" * 70)
    
    tests = [
        ("Direct Function Tests", test_health_endpoint_direct),
        ("FastAPI Client Tests", test_health_endpoint_with_fastapi)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        try:
            result = await test_func()
            if result:
                passed += 1
                print(f"   ✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"   ❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"   ❌ {test_name} FAILED: {e}")
    
    print("\n" + "=" * 70)
    print("HEALTH ENDPOINT TEST SUMMARY")
    print("=" * 70)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL HEALTH ENDPOINT TESTS PASSED!")
        print("✅ Health endpoints working correctly after architecture fixes")
        print("✅ Basic health check functional")
        print("✅ Detailed health check operational")
        print("✅ FastAPI integration working")
    else:
        print(f"\n⚠️  {failed} health endpoint test(s) failed")
    
    return failed == 0

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)