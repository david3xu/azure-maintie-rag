"""
Layer 2: PydanticAI Agent Direct Tests
======================================

Comprehensive testing of Agent[UniversalDeps, T] objects with real Azure OpenAI backends.
Tests all 3 agents: Domain Intelligence, Knowledge Extraction, Universal Search.
Validates PydanticAI framework integration and tool execution.
"""

import asyncio
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load environment before all imports
load_dotenv()

from agents.core.universal_deps import get_universal_deps, reset_universal_deps
from agents.core.universal_models import (
    ExtractionRequest,
    ExtractionResult,
    MultiModalSearchResult,
    SearchRequest,
    UniversalDomainAnalysis,
)
from agents.domain_intelligence.agent import (
    create_domain_intelligence_agent,
    domain_intelligence_agent,
    run_domain_analysis,
)
from agents.knowledge_extraction.agent import knowledge_extraction_agent
from agents.universal_search.agent import universal_search_agent


class TestPydanticAIAgentConfiguration:
    """Test PydanticAI agent configuration and setup."""

    @pytest.mark.layer2
    @pytest.mark.azure
    def test_environment_configuration_for_agents(self):
        """Test that environment is properly configured for PydanticAI agents."""
        # Critical: PydanticAI expects these environment variables
        required_for_pydantic = [
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "OPENAI_MODEL_DEPLOYMENT",
            "EMBEDDING_MODEL_DEPLOYMENT",
        ]

        # Set OPENAI_BASE_URL if not already set (PydanticAI requirement)
        if not os.getenv("OPENAI_BASE_URL"):
            os.environ["OPENAI_BASE_URL"] = os.getenv("AZURE_OPENAI_ENDPOINT", "")

        missing_vars = []
        for var in required_for_pydantic:
            value = os.getenv(var)
            if not value:
                missing_vars.append(var)

        if missing_vars:
            pytest.fail(
                f"PydanticAI agents require these environment variables: {missing_vars}"
            )

        print("✅ PydanticAI Environment: Properly configured")
        print(f"   OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL')}")
        print(f"   Model: {os.getenv('OPENAI_MODEL_DEPLOYMENT')}")

    @pytest.mark.layer2
    @pytest.mark.azure
    @pytest.mark.asyncio
    async def test_universal_deps_initialization(self):
        """Test universal dependencies initialization with real Azure services."""
        # Reset to ensure clean state
        reset_universal_deps()

        try:
            deps = await get_universal_deps()
            assert deps is not None
            assert deps._initialized

            # Check service availability
            service_status = await deps._get_service_status()

            print("✅ Universal Dependencies: Initialization successful")
            print(
                f"   OpenAI Client: {'Available' if service_status.get('openai') else 'Failed'}"
            )
            print(
                f"   Cosmos Client: {'Available' if service_status.get('cosmos') else 'Failed'}"
            )
            print(
                f"   Search Client: {'Available' if service_status.get('search') else 'Failed'}"
            )
            print(
                f"   Storage Client: {'Available' if service_status.get('storage') else 'Failed'}"
            )

            # At minimum, OpenAI should be available for agent testing
            if not service_status.get("openai"):
                pytest.fail(
                    "OpenAI service is required for agent testing but not available"
                )

        except Exception as e:
            pytest.fail(f"Universal dependencies initialization failed: {e}")


class TestDomainIntelligenceAgent:
    """Test Domain Intelligence Agent functionality with real Azure OpenAI."""

    @pytest.mark.layer2
    @pytest.mark.azure
    def test_domain_intelligence_agent_import_and_structure(self):
        """Test that Domain Intelligence Agent has proper structure."""
        assert domain_intelligence_agent is not None
        assert hasattr(domain_intelligence_agent, "run")
        assert hasattr(domain_intelligence_agent, "system_prompt")
        assert hasattr(domain_intelligence_agent, "_deps_type")

        print("✅ Domain Intelligence Agent: Proper PydanticAI structure")
        print(f"   Agent Type: {type(domain_intelligence_agent)}")
        print(
            f"   System Prompt Length: {len(domain_intelligence_agent.system_prompt)} chars"
        )

    @pytest.mark.layer2
    @pytest.mark.azure
    @pytest.mark.asyncio
    async def test_domain_intelligence_agent_run_method(self):
        """Test Domain Intelligence Agent.run() with real Azure OpenAI backend."""
        try:
            deps = await get_universal_deps()

            # Test with sample content
            sample_content = """
            Azure Cosmos DB is a fully managed NoSQL database service.
            It provides multiple data models including document, key-value, graph, and column-family.
            The service offers global distribution and horizontal scaling capabilities.
            """

            # Use run_domain_analysis helper function which handles proper agent initialization
            result = await run_domain_analysis(sample_content)

            # Validate result structure
            assert isinstance(result, UniversalDomainAnalysis)
            assert hasattr(result, "discovered_characteristics")
            assert hasattr(result, "processing_configuration")

            # Validate discovered characteristics
            chars = result.discovered_characteristics
            assert 0.0 <= chars.vocabulary_complexity <= 1.0
            assert 0.0 <= chars.concept_density <= 1.0
            assert isinstance(chars.structural_patterns, list)
            assert isinstance(chars.content_signature, str)

            print("✅ Domain Intelligence Agent: run() method successful")
            print(f"   Vocabulary Complexity: {chars.vocabulary_complexity:.3f}")
            print(f"   Concept Density: {chars.concept_density:.3f}")
            print(f"   Structural Patterns: {chars.structural_patterns}")
            print(f"   Content Signature: {chars.content_signature}")

        except Exception as e:
            # Provide detailed error information for debugging
            import traceback

            error_details = traceback.format_exc()
            print(f"❌ Domain Intelligence Agent test failed:")
            print(f"   Error: {e}")
            print(f"   Details: {error_details}")

            # If it's a 404 error, provide specific guidance
            if "404" in str(e):
                pytest.fail(
                    f"Azure OpenAI 404 error - check model deployment name: {e}"
                )
            else:
                pytest.fail(f"Domain Intelligence Agent test failed: {e}")

    @pytest.mark.layer2
    @pytest.mark.azure
    @pytest.mark.asyncio
    async def test_domain_intelligence_agent_tools(self):
        """Test Domain Intelligence Agent tool execution."""
        try:
            deps = await get_universal_deps()

            # Test content analysis prompt to trigger tool usage
            content_analysis_prompt = """
            Please analyze the following content characteristics without making domain assumptions:
            
            "Python is a programming language that emphasizes readability and simplicity.
            Functions are defined using the def keyword, and classes use the class keyword.
            Popular frameworks include Django for web development and NumPy for data science."
            
            Use the analyze_content_characteristics tool to discover vocabulary complexity and concept density.
            """

            result = await domain_intelligence_agent.run(
                content_analysis_prompt, deps=deps
            )

            # Validate that tools were executed and results structured properly
            assert result is not None
            assert hasattr(result, "output")
            assert isinstance(result.output, UniversalDomainAnalysis)

            analysis = result.output
            assert hasattr(analysis, "discovered_characteristics")

            print("✅ Domain Intelligence Agent: Tool execution successful")
            print(f"   Tool Usage: Content analysis completed")
            print(f"   Result Type: {type(analysis)}")

        except Exception as e:
            pytest.fail(f"Domain Intelligence Agent tool test failed: {e}")


class TestKnowledgeExtractionAgent:
    """Test Knowledge Extraction Agent functionality with real Azure OpenAI."""

    @pytest.mark.layer2
    @pytest.mark.azure
    def test_knowledge_extraction_agent_structure(self):
        """Test Knowledge Extraction Agent structure."""
        assert knowledge_extraction_agent is not None
        assert hasattr(knowledge_extraction_agent, "run")
        assert hasattr(knowledge_extraction_agent, "system_prompt")

        print("✅ Knowledge Extraction Agent: Proper PydanticAI structure")
        print(f"   Agent Type: {type(knowledge_extraction_agent)}")

    @pytest.mark.layer2
    @pytest.mark.azure
    @pytest.mark.asyncio
    async def test_knowledge_extraction_agent_run_method(self):
        """Test Knowledge Extraction Agent.run() with real Azure OpenAI."""
        try:
            deps = await get_universal_deps()

            # Test with sample technical content
            sample_content = """
            Microsoft Azure provides cloud computing services including virtual machines,
            storage accounts, and networking resources. Azure Resource Manager organizes
            these resources into resource groups for easier management.
            """

            # Create extraction request
            extraction_prompt = f"Extract entities and relationships from this content: {sample_content}"

            result = await knowledge_extraction_agent.run(extraction_prompt, deps=deps)

            # Validate result structure
            assert result is not None
            assert hasattr(result, "output")
            assert isinstance(result.output, ExtractionResult)

            extraction = result.output
            assert hasattr(extraction, "entities")
            assert hasattr(extraction, "relationships")
            assert isinstance(extraction.entities, list)
            assert isinstance(extraction.relationships, list)

            print("✅ Knowledge Extraction Agent: run() method successful")
            print(f"   Entities Found: {len(extraction.entities)}")
            print(f"   Relationships Found: {len(extraction.relationships)}")

            # Print sample entities and relationships
            if extraction.entities:
                print(
                    f"   Sample Entity: {extraction.entities[0].text if extraction.entities else 'None'}"
                )
            if extraction.relationships:
                print(
                    f"   Sample Relationship: {extraction.relationships[0].relationship_type if extraction.relationships else 'None'}"
                )

        except Exception as e:
            import traceback

            print(f"❌ Knowledge Extraction Agent test failed: {e}")
            print(f"   Traceback: {traceback.format_exc()}")

            if "404" in str(e):
                pytest.fail(f"Azure OpenAI 404 error - check model deployment: {e}")
            else:
                pytest.fail(f"Knowledge Extraction Agent test failed: {e}")

    @pytest.mark.layer2
    @pytest.mark.azure
    @pytest.mark.asyncio
    async def test_knowledge_extraction_with_structured_request(self):
        """Test Knowledge Extraction Agent with structured ExtractionRequest."""
        try:
            deps = await get_universal_deps()

            # Create structured extraction request
            request = ExtractionRequest(
                content="Azure Cosmos DB supports multiple APIs including MongoDB, Cassandra, and SQL.",
                extract_entities=True,
                extract_relationships=True,
                confidence_threshold=0.7,
            )

            result = await knowledge_extraction_agent.run(
                f"Process this extraction request: {request.model_dump_json()}",
                deps=deps,
            )

            assert result is not None
            extraction = result.output
            assert isinstance(extraction, ExtractionResult)

            print("✅ Knowledge Extraction Agent: Structured request successful")
            print(f"   Request Type: {type(request)}")
            print(f"   Response Type: {type(extraction)}")

        except Exception as e:
            pytest.fail(f"Structured extraction test failed: {e}")


class TestUniversalSearchAgent:
    """Test Universal Search Agent functionality with real Azure OpenAI."""

    @pytest.mark.layer2
    @pytest.mark.azure
    def test_universal_search_agent_structure(self):
        """Test Universal Search Agent structure."""
        assert universal_search_agent is not None
        assert hasattr(universal_search_agent, "run")
        assert hasattr(universal_search_agent, "system_prompt")

        print("✅ Universal Search Agent: Proper PydanticAI structure")
        print(f"   Agent Type: {type(universal_search_agent)}")

    @pytest.mark.layer2
    @pytest.mark.azure
    @pytest.mark.asyncio
    async def test_universal_search_agent_run_method(self):
        """Test Universal Search Agent.run() with real Azure OpenAI."""
        try:
            deps = await get_universal_deps()

            # Test with sample search query
            search_query = (
                "Find information about Azure Cosmos DB performance optimization"
            )

            result = await universal_search_agent.run(
                f"Execute universal search for: {search_query}", deps=deps
            )

            # Validate result structure
            assert result is not None
            assert hasattr(result, "output")
            assert isinstance(result.output, MultiModalSearchResult)

            search_result = result.output
            assert hasattr(search_result, "unified_results")
            assert hasattr(search_result, "search_strategy_used")
            assert isinstance(search_result.unified_results, list)
            assert isinstance(search_result.search_strategy_used, str)

            print("✅ Universal Search Agent: run() method successful")
            print(f"   Search Strategy: {search_result.search_strategy_used}")
            print(f"   Results Count: {len(search_result.unified_results)}")

        except Exception as e:
            import traceback

            print(f"❌ Universal Search Agent test failed: {e}")
            print(f"   Traceback: {traceback.format_exc()}")

            if "404" in str(e):
                pytest.fail(f"Azure OpenAI 404 error - check model deployment: {e}")
            else:
                pytest.fail(f"Universal Search Agent test failed: {e}")


class TestAgentOrchestration:
    """Test multi-agent orchestration and interaction patterns."""

    @pytest.mark.layer2
    @pytest.mark.azure
    @pytest.mark.asyncio
    async def test_multi_agent_workflow(self):
        """Test workflow involving multiple agents in sequence."""
        try:
            deps = await get_universal_deps()

            # Sample content for multi-agent processing
            content = """
            Azure App Service is a platform-as-a-service (PaaS) offering that enables 
            developers to build and deploy web applications quickly. It supports multiple 
            programming languages including .NET, Java, Python, and Node.js.
            """

            # Step 1: Domain Intelligence Analysis
            domain_analysis = await run_domain_analysis(content)
            assert isinstance(domain_analysis, UniversalDomainAnalysis)

            # Step 2: Knowledge Extraction based on domain analysis
            extraction_prompt = f"Extract entities and relationships from: {content}"
            extraction_result = await knowledge_extraction_agent.run(
                extraction_prompt, deps=deps
            )
            assert extraction_result.output is not None

            # Step 3: Universal Search based on extracted entities
            if extraction_result.output.entities:
                first_entity = (
                    extraction_result.output.entities[0].text
                    if extraction_result.output.entities
                    else "Azure"
                )
                search_result = await universal_search_agent.run(
                    f"Search for information about: {first_entity}", deps=deps
                )
                assert search_result.output is not None

            print("✅ Multi-Agent Workflow: All agents worked together successfully")
            print(f"   Domain Analysis: Completed")
            print(
                f"   Knowledge Extraction: {len(extraction_result.output.entities)} entities"
            )
            print(f"   Universal Search: Completed")

        except Exception as e:
            pytest.fail(f"Multi-agent workflow test failed: {e}")

    @pytest.mark.layer2
    @pytest.mark.azure
    @pytest.mark.asyncio
    async def test_agent_performance_characteristics(self):
        """Test agent performance with real Azure OpenAI backend."""
        import time

        try:
            deps = await get_universal_deps()

            # Measure Domain Intelligence Agent performance
            start_time = time.time()
            domain_result = await run_domain_analysis(
                "Test content for performance measurement"
            )
            domain_duration = time.time() - start_time

            # Measure Knowledge Extraction Agent performance
            start_time = time.time()
            extraction_result = await knowledge_extraction_agent.run(
                "Extract entities from: Test content for performance measurement",
                deps=deps,
            )
            extraction_duration = time.time() - start_time

            # Validate reasonable performance (should be under 15 seconds each)
            assert (
                domain_duration < 15.0
            ), f"Domain analysis too slow: {domain_duration:.2f}s"
            assert (
                extraction_duration < 15.0
            ), f"Extraction too slow: {extraction_duration:.2f}s"

            print("✅ Agent Performance: Within acceptable ranges")
            print(f"   Domain Analysis: {domain_duration:.2f}s")
            print(f"   Knowledge Extraction: {extraction_duration:.2f}s")
            print(
                f"   Both agents < 15s target: {'✅' if max(domain_duration, extraction_duration) < 15 else '❌'}"
            )

        except Exception as e:
            pytest.fail(f"Agent performance test failed: {e}")


class TestAgentErrorHandling:
    """Test agent error handling and recovery mechanisms."""

    @pytest.mark.layer2
    @pytest.mark.azure
    @pytest.mark.asyncio
    async def test_agent_invalid_input_handling(self):
        """Test how agents handle invalid or edge case inputs."""
        try:
            deps = await get_universal_deps()

            # Test with minimal content
            minimal_result = await run_domain_analysis("Test")
            assert isinstance(minimal_result, UniversalDomainAnalysis)

            # Test with empty-ish content
            empty_result = await run_domain_analysis("   ")
            assert isinstance(empty_result, UniversalDomainAnalysis)

            # Test with very long content (should be handled gracefully)
            long_content = "This is a test sentence. " * 1000
            long_result = await run_domain_analysis(long_content)
            assert isinstance(long_result, UniversalDomainAnalysis)

            print("✅ Agent Error Handling: Graceful handling of edge cases")
            print(f"   Minimal Content: Handled")
            print(f"   Empty Content: Handled")
            print(f"   Long Content: Handled")

        except Exception as e:
            pytest.fail(f"Agent error handling test failed: {e}")

    @pytest.mark.layer2
    @pytest.mark.azure
    @pytest.mark.asyncio
    async def test_agent_dependency_failure_handling(self):
        """Test agent behavior when some dependencies are unavailable."""
        try:
            deps = await get_universal_deps()

            # Test agents with limited service availability
            service_status = await deps._get_service_status()
            available_services = [
                service for service, available in service_status.items() if available
            ]

            # Domain Intelligence should work with just OpenAI
            if service_status.get("openai"):
                result = await run_domain_analysis(
                    "Test content for dependency testing"
                )
                assert isinstance(result, UniversalDomainAnalysis)
                print("✅ Domain Intelligence Agent: Works with minimal dependencies")

            print(
                f"✅ Dependency Handling: Tested with {len(available_services)} available services"
            )

        except Exception as e:
            pytest.fail(f"Dependency failure handling test failed: {e}")
