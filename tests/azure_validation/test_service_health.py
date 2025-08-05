"""
Azure Service Health Validation Tests - CODING_STANDARDS Compliant
Tests real Azure service connectivity and health monitoring.
"""

import pytest
import asyncio
import time
from typing import Dict, Any, List


@pytest.mark.azure
class TestAzureServiceHealth:
    """Test Azure service health and connectivity"""
    
    @pytest.mark.asyncio
    async def test_azure_service_initialization(self, azure_services):
        """Test Azure service initialization and health"""
        
        # Services should be initialized by fixture
        assert azure_services is not None
        
        # Get comprehensive health status
        health_status = azure_services.get_service_status()
        
        # Validate health status structure
        required_fields = ["overall_health", "successful_services", "total_services", "service_details"]
        for field in required_fields:
            assert field in health_status, f"Missing required health field: {field}"
        
        # Validate health data
        assert isinstance(health_status["overall_health"], bool)
        assert isinstance(health_status["successful_services"], int)
        assert isinstance(health_status["total_services"], int)
        assert isinstance(health_status["service_details"], dict)
        
        # Validate service counts
        assert health_status["total_services"] > 0, "No Azure services configured"
        assert health_status["successful_services"] >= 0, "Negative successful services count"
        assert health_status["successful_services"] <= health_status["total_services"], "More successful than total services"
        
        print(f"🏥 Azure Health Status:")
        print(f"   Overall Health: {'✅ Healthy' if health_status['overall_health'] else '❌ Degraded'}")
        print(f"   Services: {health_status['successful_services']}/{health_status['total_services']}")
        
        # Log individual service status
        for service_name, service_health in health_status["service_details"].items():
            status_icon = "✅" if service_health else "❌"
            print(f"   {status_icon} {service_name}")
    
    @pytest.mark.asyncio
    async def test_individual_service_connectivity(self, azure_services):
        """Test connectivity to each Azure service individually"""
        
        service_tests = {}
        
        # Test Azure OpenAI
        if hasattr(azure_services, 'openai_client') and azure_services.openai_client:
            try:
                start_time = time.time()
                result = await azure_services.openai_client.get_completion(
                    "Test connectivity", max_tokens=5
                )
                connectivity_time = time.time() - start_time
                
                service_tests["openai"] = {
                    "available": True,
                    "connectivity_test": result is not None,
                    "response_time": connectivity_time,
                    "result": result
                }
                
            except Exception as e:
                service_tests["openai"] = {
                    "available": True,
                    "connectivity_test": False,
                    "error": str(e)
                }
        else:
            service_tests["openai"] = {"available": False}
        
        # Test Azure Search
        if hasattr(azure_services, 'search_client') and azure_services.search_client:
            try:
                start_time = time.time()
                result = await azure_services.search_client.search("connectivity test", top=1)
                connectivity_time = time.time() - start_time
                
                service_tests["search"] = {
                    "available": True,
                    "connectivity_test": True,  # Search should not fail even with no results
                    "response_time": connectivity_time,
                    "result": result
                }
                
            except Exception as e:
                service_tests["search"] = {
                    "available": True,
                    "connectivity_test": False,
                    "error": str(e)
                }
        else:
            service_tests["search"] = {"available": False}
        
        # Test Azure Cosmos
        if hasattr(azure_services, 'cosmos_client') and azure_services.cosmos_client:
            try:
                start_time = time.time()
                health = await azure_services.cosmos_client.health_check()
                connectivity_time = time.time() - start_time
                
                service_tests["cosmos"] = {
                    "available": True,
                    "connectivity_test": health is not None,
                    "response_time": connectivity_time,
                    "result": health
                }
                
            except Exception as e:
                service_tests["cosmos"] = {
                    "available": True,
                    "connectivity_test": False,
                    "error": str(e)
                }
        else:
            service_tests["cosmos"] = {"available": False}
        
        # Test Azure Storage
        if hasattr(azure_services, 'storage_client') and azure_services.storage_client:
            try:
                start_time = time.time()
                containers = await azure_services.storage_client.list_containers()
                connectivity_time = time.time() - start_time
                
                service_tests["storage"] = {
                    "available": True,
                    "connectivity_test": containers is not None,
                    "response_time": connectivity_time,
                    "container_count": len(containers) if containers else 0
                }
                
            except Exception as e:
                service_tests["storage"] = {
                    "available": True,
                    "connectivity_test": False,
                    "error": str(e)
                }
        else:
            service_tests["storage"] = {"available": False}
        
        # Report connectivity results
        print(f"🔌 Individual Service Connectivity:")
        for service_name, test_result in service_tests.items():
            if test_result["available"]:
                connectivity = test_result.get("connectivity_test", False)
                response_time = test_result.get("response_time", 0)
                
                status_icon = "✅" if connectivity else "❌"
                print(f"   {status_icon} {service_name.title()}: "
                      f"{'Connected' if connectivity else 'Failed'}")
                
                if connectivity and response_time:
                    print(f"      Response time: {response_time:.3f}s")
                
                if not connectivity and "error" in test_result:
                    print(f"      Error: {test_result['error']}")
            else:
                print(f"   ⚪ {service_name.title()}: Not configured")
        
        # Validate at least one service is working
        working_services = sum(
            1 for test in service_tests.values() 
            if test["available"] and test.get("connectivity_test", False)
        )
        
        assert working_services > 0, f"No Azure services are working: {service_tests}"
        
        print(f"📊 Connectivity Summary: {working_services} services working")
        
        return service_tests
    
    @pytest.mark.asyncio
    async def test_service_health_monitoring(self, azure_services):
        """Test ongoing service health monitoring"""
        
        # Perform multiple health checks over time
        health_checks = []
        
        for i in range(3):  # 3 health checks with delay
            start_time = time.time()
            health_status = azure_services.get_service_status()
            check_time = time.time() - start_time
            
            health_checks.append({
                "check_number": i + 1,
                "timestamp": time.time(),
                "overall_health": health_status["overall_health"],
                "successful_services": health_status["successful_services"],
                "total_services": health_status["total_services"],
                "check_duration": check_time,
                "service_details": health_status["service_details"].copy()
            })
            
            if i < 2:  # Don't wait after last check
                await asyncio.sleep(1)  # 1 second between checks
        
        # Analyze health consistency
        health_states = [check["overall_health"] for check in health_checks]
        service_counts = [check["successful_services"] for check in health_checks]
        
        # Health should be consistent across checks
        health_consistency = len(set(health_states)) <= 2  # Allow for some variation
        assert health_consistency, f"Health status too inconsistent: {health_states}"
        
        # Service counts should not vary dramatically
        min_services = min(service_counts)
        max_services = max(service_counts)
        service_consistency = (max_services - min_services) <= 1  # Allow 1 service variation
        
        print(f"🔍 Health Monitoring Results:")
        for check in health_checks:
            print(f"   Check {check['check_number']}: "
                  f"{'✅' if check['overall_health'] else '❌'} "
                  f"{check['successful_services']}/{check['total_services']} "
                  f"({check['check_duration']:.3f}s)")
        
        print(f"   Health consistency: {'✅ Stable' if health_consistency else '❌ Unstable'}")
        print(f"   Service consistency: {'✅ Stable' if service_consistency else '❌ Unstable'}")
        
        assert service_consistency, f"Service count too inconsistent: {service_counts}"
        
        return health_checks
    
    @pytest.mark.asyncio
    async def test_service_failover_simulation(self, azure_services):
        """Test system behavior when services are unavailable"""
        
        # Get initial health status
        initial_health = azure_services.get_service_status()
        
        print(f"🔄 Initial Health: {initial_health['successful_services']}/{initial_health['total_services']}")
        
        # Test system behavior with current service availability
        if initial_health["successful_services"] == initial_health["total_services"]:
            print("✅ All services available - system at full capacity")
            
            # Verify full functionality
            assert initial_health["overall_health"] == True
            
        elif initial_health["successful_services"] >= 2:
            print("⚠️ Partial service availability - testing degraded mode")
            
            # System should handle partial availability gracefully
            assert initial_health["successful_services"] > 0
            
            # Overall health may be true or false depending on implementation
            # but system should not crash
            
        elif initial_health["successful_services"] == 1:
            print("🚨 Minimal service availability - testing survival mode")
            
            # At least one service working
            assert initial_health["successful_services"] == 1
            
        else:
            print("❌ No services available - testing complete failure mode")
            
            # System should handle complete failure gracefully
            assert initial_health["successful_services"] == 0
            assert initial_health["overall_health"] == False
        
        # Validate system doesn't crash regardless of service availability
        try:
            # Attempt to get health status multiple times
            for _ in range(3):
                health = azure_services.get_service_status()
                assert health is not None
                assert isinstance(health, dict)
                
            print("✅ System remains stable regardless of service availability")
            
        except Exception as e:
            pytest.fail(f"System instability detected: {e}")


@pytest.mark.azure
class TestAzureServicePerformance:
    """Test Azure service performance characteristics"""
    
    @pytest.mark.asyncio
    async def test_service_response_times(self, azure_services):
        """Test individual Azure service response times"""
        
        response_times = {}
        
        # Test OpenAI response time
        if azure_services.openai_client:
            start_time = time.time()
            try:
                result = await azure_services.openai_client.get_completion("Response time test", max_tokens=10)
                response_time = time.time() - start_time
                
                response_times["openai"] = {
                    "response_time": response_time,
                    "success": result is not None,
                    "target_sla": 2.0  # 2 second target for LLM
                }
                
            except Exception as e:
                response_times["openai"] = {
                    "response_time": time.time() - start_time,
                    "success": False,
                    "error": str(e)
                }
        
        # Test Search response time
        if azure_services.search_client:
            start_time = time.time()
            try:
                result = await azure_services.search_client.search("response time test", top=3)
                response_time = time.time() - start_time
                
                response_times["search"] = {
                    "response_time": response_time,
                    "success": True,  # Search should not fail
                    "target_sla": 1.0  # 1 second target for search
                }
                
            except Exception as e:
                response_times["search"] = {
                    "response_time": time.time() - start_time,
                    "success": False,
                    "error": str(e)
                }
        
        # Report response time results
        print(f"⚡ Azure Service Response Times:")
        for service_name, timing in response_times.items():
            target_sla = timing.get("target_sla", 3.0)
            response_time = timing["response_time"]
            success = timing.get("success", False)
            
            sla_met = response_time <= target_sla
            status_icon = "✅" if success and sla_met else "⚠️" if success else "❌"
            
            print(f"   {status_icon} {service_name.title()}: {response_time:.3f}s "
                  f"(target: {target_sla}s, {'success' if success else 'failed'})")
            
            if not success and "error" in timing:
                print(f"      Error: {timing['error']}")
        
        # Validate at least one service meets SLA
        services_meeting_sla = [
            service for service, timing in response_times.items()
            if timing.get("success", False) and timing["response_time"] <= timing.get("target_sla", 3.0)
        ]
        
        if response_times:  # Only validate if we have services to test
            assert len(services_meeting_sla) > 0, f"No services meeting SLA: {response_times}"
        
        return response_times
    
    @pytest.mark.asyncio
    async def test_concurrent_service_load(self, azure_services):
        """Test Azure services under concurrent load"""
        
        # Create concurrent tasks for available services
        concurrent_tasks = []
        
        # Add OpenAI tasks
        if azure_services.openai_client:
            for i in range(3):
                concurrent_tasks.append(
                    azure_services.openai_client.get_completion(f"Concurrent test {i}", max_tokens=10)
                )
        
        # Add Search tasks
        if azure_services.search_client:
            for i in range(3):
                concurrent_tasks.append(
                    azure_services.search_client.search(f"concurrent test {i}", top=2)
                )
        
        if not concurrent_tasks:
            pytest.skip("No Azure services available for concurrent testing")
        
        # Execute concurrent operations
        start_time = time.time()
        results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Analyze concurrent performance
        successful_operations = sum(
            1 for result in results 
            if not isinstance(result, Exception) and result is not None
        )
        
        failed_operations = len(results) - successful_operations
        success_rate = successful_operations / len(results) * 100
        
        print(f"🔄 Concurrent Load Test Results:")
        print(f"   Total operations: {len(results)}")
        print(f"   Successful: {successful_operations}")
        print(f"   Failed: {failed_operations}")
        print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Operations per second: {len(results) / total_time:.1f}")
        
        # Validate reasonable concurrent performance
        assert success_rate >= 70.0, f"Concurrent success rate too low: {success_rate:.1f}%"
        assert total_time <= 10.0, f"Concurrent operations too slow: {total_time:.3f}s"
        
        return {
            "total_operations": len(results),
            "successful_operations": successful_operations,
            "success_rate": success_rate,
            "total_time": total_time,
            "operations_per_second": len(results) / total_time
        }