"""
Azure Services Integration Tests - CODING_STANDARDS Compliant
Tests ConsolidatedAzureServices with real Azure backends.
"""

import pytest
import asyncio
from typing import Dict, Any


@pytest.mark.azure
@pytest.mark.integration
class TestConsolidatedAzureServices:
    """Test ConsolidatedAzureServices with real Azure infrastructure"""

    @pytest.mark.asyncio
    async def test_service_initialization(self, azure_services):
        """Test real Azure service initialization"""
        # Services should be initialized by fixture
        assert azure_services is not None

        # Get service status
        status = azure_services.get_service_status()

        # Validate service health
        assert status["total_services"] > 0
        assert status["successful_services"] >= 2  # At least 2 services working
        assert isinstance(status["overall_health"], bool)

        print(
            f"✅ Azure Services Status: {status['successful_services']}/{status['total_services']}"
        )

    @pytest.mark.asyncio
    async def test_openai_client_connectivity(self, azure_services):
        """Test real Azure OpenAI client connectivity"""
        openai_client = azure_services.openai_client

        if not openai_client:
            pytest.skip("Azure OpenAI client not available")

        # Test basic completion request
        try:
            result = await openai_client.get_completion(
                "Test prompt for connectivity", max_tokens=10
            )

            assert result is not None
            assert isinstance(result, dict)
            print("✅ Azure OpenAI connectivity verified")

        except Exception as e:
            pytest.fail(f"Azure OpenAI connectivity failed: {e}")

    @pytest.mark.asyncio
    async def test_search_client_connectivity(self, azure_services):
        """Test real Azure Cognitive Search connectivity"""
        search_client = azure_services.search_client

        if not search_client:
            pytest.skip("Azure Search client not available")

        # Test basic search operation
        try:
            # Simple search test (should not fail even with no results)
            result = await search_client.search("test query", top=1)

            assert result is not None
            print("✅ Azure Search connectivity verified")

        except Exception as e:
            pytest.fail(f"Azure Search connectivity failed: {e}")

    @pytest.mark.asyncio
    async def test_cosmos_client_connectivity(self, azure_services):
        """Test real Azure Cosmos DB connectivity"""
        cosmos_client = azure_services.cosmos_client

        if not cosmos_client:
            pytest.skip("Azure Cosmos client not available")

        # Test basic cosmos operation
        try:
            # Simple health check operation
            health = await cosmos_client.health_check()

            assert health is not None
            print("✅ Azure Cosmos connectivity verified")

        except Exception as e:
            pytest.fail(f"Azure Cosmos connectivity failed: {e}")

    @pytest.mark.asyncio
    async def test_storage_client_connectivity(self, azure_services):
        """Test real Azure Blob Storage connectivity"""
        storage_client = azure_services.storage_client

        if not storage_client:
            pytest.skip("Azure Storage client not available")

        # Test basic storage operation
        try:
            # Simple container list operation
            containers = await storage_client.list_containers()

            assert containers is not None
            assert isinstance(containers, list)
            print("✅ Azure Storage connectivity verified")

        except Exception as e:
            pytest.fail(f"Azure Storage connectivity failed: {e}")

    @pytest.mark.asyncio
    async def test_service_health_monitoring(self, azure_services):
        """Test service health monitoring functionality"""
        # Get comprehensive health status
        health = azure_services.get_service_status()

        # Validate health structure
        assert "overall_health" in health
        assert "successful_services" in health
        assert "total_services" in health
        assert "service_details" in health

        # Validate health data types
        assert isinstance(health["overall_health"], bool)
        assert isinstance(health["successful_services"], int)
        assert isinstance(health["total_services"], int)
        assert isinstance(health["service_details"], dict)

        # Log health status
        print(f"📊 Health Status: {health}")

        # Validate reasonable service counts
        assert health["total_services"] >= 3  # At least openai, search, cosmos
        assert health["successful_services"] <= health["total_services"]

    @pytest.mark.asyncio
    async def test_concurrent_service_usage(self, azure_services):
        """Test concurrent usage of multiple Azure services"""
        tasks = []

        # Create concurrent tasks for different services
        if azure_services.openai_client:
            tasks.append(
                azure_services.openai_client.get_completion(
                    "concurrent test", max_tokens=5
                )
            )

        if azure_services.search_client:
            tasks.append(azure_services.search_client.search("concurrent test", top=1))

        if len(tasks) < 2:
            pytest.skip("Need at least 2 Azure services for concurrent testing")

        # Execute concurrent operations
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Validate results
            assert len(results) == len(tasks)

            # Check that at least one operation succeeded
            successes = sum(
                1 for result in results if not isinstance(result, Exception)
            )
            assert successes > 0, f"All concurrent operations failed: {results}"

            print(f"✅ Concurrent operations: {successes}/{len(tasks)} successful")

        except Exception as e:
            pytest.fail(f"Concurrent Azure operations failed: {e}")


@pytest.mark.azure
@pytest.mark.integration
class TestAzureServiceErrorHandling:
    """Test error handling and resilience with real Azure services"""

    @pytest.mark.asyncio
    async def test_service_unavailable_handling(self, azure_services):
        """Test handling when some services are unavailable"""
        status = azure_services.get_service_status()

        # Test should handle partial service availability gracefully
        if status["successful_services"] < status["total_services"]:
            print(
                f"⚠️ Testing with partial service availability: "
                f"{status['successful_services']}/{status['total_services']}"
            )

            # System should still function with reduced capability
            assert status["successful_services"] > 0

            # Overall health should reflect partial availability
            expected_health = status["successful_services"] >= 2
            assert status["overall_health"] == expected_health

    @pytest.mark.asyncio
    async def test_invalid_request_handling(self, azure_services):
        """Test handling of invalid requests to Azure services"""

        # Test invalid OpenAI request
        if azure_services.openai_client:
            try:
                # Request with invalid parameters
                result = await azure_services.openai_client.get_completion(
                    "", max_tokens=-1  # Invalid token count
                )
                # Should handle gracefully, not crash
                assert result is not None

            except Exception as e:
                # Exception is acceptable, crash is not
                assert "error" in str(e).lower() or "invalid" in str(e).lower()
                print(f"✅ Invalid request handled gracefully: {type(e).__name__}")

    @pytest.mark.asyncio
    async def test_timeout_handling(self, azure_services, performance_monitor):
        """Test timeout handling for Azure operations"""

        async with performance_monitor.measure_operation(
            "timeout_test", sla_target=5.0
        ):
            # Test that operations complete within reasonable time
            if azure_services.openai_client:
                result = await azure_services.openai_client.get_completion(
                    "This is a timeout test prompt", max_tokens=50
                )

                assert result is not None
                print("✅ Operation completed within timeout")
            else:
                pytest.skip("No Azure OpenAI client available for timeout testing")
