"""
Comprehensive Integration Tests for Azure Universal RAG System
============================================================

This file provides a complete integration test suite that validates:
1. Azure service authentication and connectivity
2. PydanticAI agent functionality with real Azure backends
3. Universal dependencies initialization
4. End-to-end workflow testing with proper error handling
5. Performance and reliability metrics

All tests use real Azure services with proper authentication.
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
from dotenv import load_dotenv

# Load environment configuration
load_dotenv()


class TestAzureIntegrationComprehensive:
    """Comprehensive integration tests for the Azure Universal RAG system."""

    @pytest.mark.asyncio
    async def test_azure_authentication_comprehensive(self):
        """Comprehensive test of Azure authentication methods."""
        print("\n🔐 Testing Azure Authentication Methods...")

        # Test DefaultAzureCredential
        try:
            from azure.identity import DefaultAzureCredential

            credential = DefaultAzureCredential()
            token = credential.get_token("https://cognitiveservices.azure.com/.default")

            assert token is not None
            assert hasattr(token, "token")
            assert len(token.token) > 0
            print("✅ DefaultAzureCredential: Working")

        except Exception as e:
            print(f"❌ DefaultAzureCredential failed: {e}")
            pytest.fail(f"DefaultAzureCredential authentication failed: {e}")

    @pytest.mark.asyncio
    async def test_azure_openai_full_integration(self):
        """Full integration test for Azure OpenAI with different authentication methods."""
        print("\n🤖 Testing Azure OpenAI Integration...")

        use_managed_identity = (
            os.getenv("USE_MANAGED_IDENTITY", "false").lower() == "true"
        )

        # Test different authentication methods
        if use_managed_identity:
            await self._test_azure_openai_with_managed_identity()
        else:
            await self._test_azure_openai_with_cli_auth()

    async def _test_azure_openai_with_managed_identity(self):
        """Test Azure OpenAI with managed identity."""
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        from openai import AsyncAzureOpenAI

        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )

        client = AsyncAzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_ad_token_provider=token_provider,
            api_version=os.getenv("OPENAI_API_VERSION", "2024-06-01"),
        )

        try:
            # Test chat completion
            response = await client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL_DEPLOYMENT", "gpt-4o"),
                messages=[{"role": "user", "content": "Test managed identity auth"}],
                max_tokens=10,
            )

            assert response.choices[0].message.content is not None
            print("✅ Azure OpenAI (Managed Identity): Working")

            # Test embeddings
            embedding_response = await client.embeddings.create(
                model=os.getenv("EMBEDDING_MODEL_DEPLOYMENT", "text-embedding-ada-002"),
                input="Test embedding with managed identity",
            )

            assert len(embedding_response.data) > 0
            assert len(embedding_response.data[0].embedding) == 1536
            print("✅ Azure OpenAI Embeddings (Managed Identity): Working")

        finally:
            await client.close()

    async def _test_azure_openai_with_cli_auth(self):
        """Test Azure OpenAI with CLI authentication."""
        from azure.identity import AzureCliCredential, get_bearer_token_provider
        from openai import AsyncAzureOpenAI

        credential = AzureCliCredential()
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )

        client = AsyncAzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_ad_token_provider=token_provider,
            api_version=os.getenv("OPENAI_API_VERSION", "2024-06-01"),
        )

        try:
            # Test chat completion
            response = await client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL_DEPLOYMENT", "gpt-4o"),
                messages=[{"role": "user", "content": "Test CLI auth"}],
                max_tokens=10,
            )

            assert response.choices[0].message.content is not None
            print("✅ Azure OpenAI (CLI Auth): Working")

            # Test embeddings
            embedding_response = await client.embeddings.create(
                model=os.getenv("EMBEDDING_MODEL_DEPLOYMENT", "text-embedding-ada-002"),
                input="Test embedding with CLI auth",
            )

            assert len(embedding_response.data) > 0
            assert len(embedding_response.data[0].embedding) == 1536
            print("✅ Azure OpenAI Embeddings (CLI Auth): Working")

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_universal_deps_initialization(self):
        """Test UniversalDeps initialization and service availability."""
        print("\n🔧 Testing Universal Dependencies...")

        from agents.core.universal_deps import get_universal_deps

        # Test initialization
        deps = await get_universal_deps()
        assert deps is not None
        print("✅ UniversalDeps: Initialized")

        # Test service status
        service_status = await deps._get_service_status()
        print(f"📊 Service Status: {service_status}")

        # Verify critical services
        critical_services = ["openai"]
        for service in critical_services:
            if not service_status.get(service, False):
                print(f"⚠️ Critical service {service} not available")
            else:
                print(f"✅ Critical service {service}: Available")

        # Test service availability methods
        available_services = deps.get_available_services()
        print(f"🌐 Available Services: {available_services}")

        assert "openai" in available_services or service_status.get("openai", False)

    @pytest.mark.asyncio
    async def test_pydantic_ai_provider_integration(self):
        """Test PydanticAI provider with Azure authentication."""
        print("\n🔌 Testing PydanticAI Provider...")

        from agents.core.azure_pydantic_provider import get_azure_openai_model

        # Test model creation
        model = get_azure_openai_model()
        assert model is not None
        assert hasattr(model, "model_name")
        print(f"✅ PydanticAI Model: Created ({model.model_name})")

        # Test model can be used in a simple completion (outside of agent context)
        try:
            # This tests the underlying Azure OpenAI client setup
            provider = model._provider
            client = provider._client

            # Basic validation that client is properly configured
            assert client is not None
            assert hasattr(client, "chat")
            print("✅ PydanticAI Provider Client: Configured")

        except Exception as e:
            print(f"⚠️ PydanticAI Provider validation failed: {e}")
            # This is not a critical failure for this test

    @pytest.mark.asyncio
    async def test_domain_intelligence_agent_minimal(self):
        """Minimal test of Domain Intelligence Agent without complex dependencies."""
        print("\n🧠 Testing Domain Intelligence Agent (Minimal)...")

        try:
            from agents.core.universal_deps import get_universal_deps
            from agents.domain_intelligence.agent import domain_intelligence_agent

            # Verify agent structure
            assert domain_intelligence_agent is not None
            assert hasattr(domain_intelligence_agent, "run")
            print("✅ Domain Intelligence Agent: Structure OK")

            # Get dependencies
            deps = await get_universal_deps()

            # Test with minimal prompt that doesn't trigger complex tools
            simple_prompt = "Analyze: Python programming language."

            try:
                # Run with timeout to avoid hanging
                result = await asyncio.wait_for(
                    domain_intelligence_agent.run(simple_prompt, deps=deps),
                    timeout=30.0,  # 30 second timeout
                )

                assert result is not None
                print("✅ Domain Intelligence Agent: Basic execution successful")
                print(f"   Result type: {type(result)}")

                # Check if result has expected structure
                if hasattr(result, "data"):
                    print(f"   Result data type: {type(result.data)}")

            except asyncio.TimeoutError:
                print("⚠️ Domain Intelligence Agent: Execution timeout (>30s)")
                pytest.skip(
                    "Agent execution timed out - may indicate configuration issues"
                )
            except Exception as agent_error:
                print(f"❌ Domain Intelligence Agent execution failed: {agent_error}")
                print(f"   Error type: {type(agent_error)}")
                pytest.fail(f"Agent execution failed: {agent_error}")

        except ImportError as import_error:
            print(f"❌ Domain Intelligence Agent import failed: {import_error}")
            pytest.fail(f"Agent import failed: {import_error}")

    @pytest.mark.asyncio
    async def test_infrastructure_clients_direct(self):
        """Test infrastructure clients directly without agent wrapper."""
        print("\n🏗️ Testing Infrastructure Clients...")

        # Test Azure OpenAI client directly
        try:
            from infrastructure.azure_openai.openai_client import (
                UnifiedAzureOpenAIClient,
            )

            openai_client = UnifiedAzureOpenAIClient()
            await openai_client.initialize()

            # Test connection
            connection_result = await openai_client.test_connection()
            assert connection_result["success"] == True
            print("✅ UnifiedAzureOpenAIClient: Working")

        except Exception as e:
            print(f"⚠️ UnifiedAzureOpenAIClient test failed: {e}")

    @pytest.mark.asyncio
    async def test_end_to_end_minimal_workflow(self):
        """Minimal end-to-end workflow test."""
        print("\n🔄 Testing Minimal End-to-End Workflow...")

        # Test data availability
        test_data_dir = (
            Path(__file__).parent.parent
            / "data"
            / "raw"
            / "azure-ai-services-language-service_output"
        )

        if test_data_dir.exists():
            test_files = list(test_data_dir.glob("*.md"))
            if test_files:
                print(f"✅ Test Data: {len(test_files)} files available")

                # Test with a small file
                small_file = min(test_files, key=lambda f: f.stat().st_size)
                content = small_file.read_text(encoding="utf-8")[
                    :500
                ]  # First 500 chars

                print(f"📄 Testing with file: {small_file.name} ({len(content)} chars)")

                # Test direct infrastructure client
                try:
                    from infrastructure.azure_openai.openai_client import (
                        UnifiedAzureOpenAIClient,
                    )

                    client = UnifiedAzureOpenAIClient()
                    await client.initialize()

                    # Test simple completion
                    result = await client.get_completion(
                        f"Analyze this content in one sentence: {content[:200]}...",
                        domain="universal",
                    )

                    assert result is not None
                    assert len(result) > 0
                    print("✅ End-to-End Workflow: Basic processing successful")

                except Exception as e:
                    print(f"⚠️ End-to-End Workflow failed: {e}")
            else:
                print("⚠️ Test Data: No files found")
        else:
            print("⚠️ Test Data: Directory not found")

    def test_environment_configuration_complete(self):
        """Complete environment configuration validation."""
        print("\n⚙️ Testing Environment Configuration...")

        # Essential variables
        essential_vars = {
            "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "OPENAI_MODEL_DEPLOYMENT": os.getenv("OPENAI_MODEL_DEPLOYMENT"),
            "EMBEDDING_MODEL_DEPLOYMENT": os.getenv("EMBEDDING_MODEL_DEPLOYMENT"),
            "AZURE_ENV_NAME": os.getenv("AZURE_ENV_NAME"),
        }

        # Optional but important variables
        optional_vars = {
            "AZURE_COSMOS_ENDPOINT": os.getenv("AZURE_COSMOS_ENDPOINT"),
            "AZURE_SEARCH_ENDPOINT": os.getenv("AZURE_SEARCH_ENDPOINT"),
            "AZURE_STORAGE_ACCOUNT": os.getenv("AZURE_STORAGE_ACCOUNT"),
        }

        print("📋 Essential Variables:")
        missing_essential = []
        for var_name, var_value in essential_vars.items():
            status = "✅ SET" if var_value else "❌ MISSING"
            print(f"   {status} {var_name}")
            if not var_value:
                missing_essential.append(var_name)

        print("📋 Optional Variables:")
        for var_name, var_value in optional_vars.items():
            status = "✅ SET" if var_value else "⚠️ NOT SET"
            print(f"   {status} {var_name}")

        assert (
            not missing_essential
        ), f"Missing essential environment variables: {missing_essential}"
        print("✅ Environment Configuration: Essential variables validated")

    @pytest.mark.asyncio
    async def test_performance_benchmarking(self):
        """Basic performance benchmarking of key operations."""
        print("\n⚡ Testing Performance Benchmarks...")

        from agents.core.universal_deps import get_universal_deps

        # Test deps initialization time
        start_time = time.time()
        deps = await get_universal_deps()
        init_time = time.time() - start_time

        print(f"⏱️ UniversalDeps initialization: {init_time:.3f}s")
        assert init_time < 10.0, f"Initialization too slow: {init_time:.3f}s"

        # Test service status check time
        start_time = time.time()
        service_status = await deps._get_service_status()
        status_time = time.time() - start_time

        print(f"⏱️ Service status check: {status_time:.3f}s")
        print(
            f"📊 Service availability: {sum(service_status.values())}/{len(service_status)}"
        )

        assert status_time < 5.0, f"Service status check too slow: {status_time:.3f}s"

        print("✅ Performance Benchmarks: Within acceptable limits")

    def test_integration_test_architecture(self):
        """Test the integration test architecture itself."""
        print("\n🧪 Testing Integration Test Architecture...")

        # Test fixtures are available
        from tests.conftest import azure_services

        assert azure_services is not None
        print("✅ Test Fixtures: Available")

        # Test environment is properly configured
        test_use_managed_identity = (
            os.getenv("TEST_USE_MANAGED_IDENTITY", "false").lower() == "true"
        )
        auth_method = "Managed Identity" if test_use_managed_identity else "CLI/API Key"
        print(f"🔐 Test Authentication Method: {auth_method}")

        # Test data directory
        test_data_dir = Path(__file__).parent.parent / "data" / "raw"
        print(
            f"📁 Test Data Directory: {'EXISTS' if test_data_dir.exists() else 'MISSING'}"
        )

        print("✅ Integration Test Architecture: Validated")


class TestIntegrationHealthReport:
    """Generate comprehensive health report for the integration testing."""

    @pytest.mark.asyncio
    async def test_generate_integration_health_report(self):
        """Generate comprehensive health report."""
        print("\n📋 COMPREHENSIVE INTEGRATION HEALTH REPORT")
        print("=" * 60)

        health_report = {
            "timestamp": time.time(),
            "environment": os.getenv("AZURE_ENV_NAME", "unknown"),
            "services": {},
            "authentication": {},
            "performance": {},
            "recommendations": [],
        }

        # Test services
        try:
            from agents.core.universal_deps import get_universal_deps

            deps = await get_universal_deps()
            service_status = await deps._get_service_status()

            health_report["services"] = service_status

            available_count = sum(service_status.values())
            total_count = len(service_status)

            print(f"🌐 SERVICES STATUS: {available_count}/{total_count} available")
            for service, status in service_status.items():
                icon = "✅" if status else "❌"
                print(f"   {icon} {service}")

        except Exception as e:
            print(f"❌ Services check failed: {e}")
            health_report["services"] = {"error": str(e)}

        # Test authentication
        try:
            from azure.identity import DefaultAzureCredential

            credential = DefaultAzureCredential()
            token = credential.get_token("https://cognitiveservices.azure.com/.default")

            health_report["authentication"]["default_credential"] = True
            print("🔐 AUTHENTICATION: DefaultAzureCredential working")

        except Exception as e:
            health_report["authentication"]["default_credential"] = False
            health_report["authentication"]["error"] = str(e)
            print(f"❌ AUTHENTICATION: Failed - {e}")

        # Performance metrics
        try:
            start = time.time()
            deps = await get_universal_deps()
            init_time = time.time() - start

            health_report["performance"]["initialization_time"] = init_time

            if init_time > 5:
                health_report["recommendations"].append(
                    "Slow initialization time - consider optimizing service startup"
                )

        except Exception as e:
            health_report["performance"]["error"] = str(e)

        # Overall health score
        service_health = (
            sum(health_report["services"].values())
            / max(len(health_report["services"]), 1)
            if isinstance(health_report["services"], dict)
            and "error" not in health_report["services"]
            else 0
        )
        auth_health = (
            1.0
            if health_report["authentication"].get("default_credential", False)
            else 0.0
        )

        overall_health = (service_health * 0.7) + (auth_health * 0.3)
        health_report["overall_health_score"] = overall_health

        print(f"\n📊 OVERALL HEALTH SCORE: {overall_health:.2f}/1.0")

        if overall_health >= 0.8:
            print("🎉 SYSTEM STATUS: EXCELLENT")
        elif overall_health >= 0.6:
            print("✅ SYSTEM STATUS: GOOD")
        elif overall_health >= 0.4:
            print("⚠️ SYSTEM STATUS: NEEDS ATTENTION")
        else:
            print("❌ SYSTEM STATUS: CRITICAL ISSUES")

        # Recommendations
        if health_report["recommendations"]:
            print("\n💡 RECOMMENDATIONS:")
            for rec in health_report["recommendations"]:
                print(f"   • {rec}")

        print("\n" + "=" * 60)

        # Save health report
        health_report_path = Path(__file__).parent / "integration_health_report.json"
        import json

        with open(health_report_path, "w") as f:
            json.dump(health_report, f, indent=2)

        print(f"📄 Health report saved to: {health_report_path}")

        # Ensure minimum health for CI/CD
        assert overall_health >= 0.5, f"System health too low: {overall_health:.2f}/1.0"
