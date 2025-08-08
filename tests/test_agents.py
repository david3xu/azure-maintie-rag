"""
Test PydanticAI Agents
======================

Tests all three Universal RAG agents with real Azure OpenAI integration.
Validates agent functionality, tool usage, and universal domain processing.
"""

import asyncio
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class TestDomainIntelligenceAgent:
    """Test Domain Intelligence Agent functionality."""

    @pytest.mark.asyncio
    async def test_agent_import(self):
        """Test that Domain Intelligence Agent can be imported successfully."""
        try:
            from agents.domain_intelligence.agent import domain_intelligence_agent

            assert domain_intelligence_agent is not None
            print("✅ Domain Intelligence Agent: Import successful")
        except Exception as e:
            pytest.fail(f"Failed to import Domain Intelligence Agent: {e}")

    @pytest.mark.asyncio
    async def test_pydantic_ai_agent_direct(self):
        """Test the PydanticAI Agent[UniversalDeps, UniversalDomainAnalysis] object directly."""
        try:
            from agents.core.universal_deps import get_universal_deps
            from agents.domain_intelligence.agent import domain_intelligence_agent

            # Test agent object properties
            assert domain_intelligence_agent is not None
            assert hasattr(domain_intelligence_agent, "run")
            assert hasattr(domain_intelligence_agent, "system_prompt")

            # Test agent can run with real dependencies
            deps = await get_universal_deps()

            sample_prompt = "Analyze these content characteristics: Python programming language with code examples"

            result = await domain_intelligence_agent.run(sample_prompt, deps=deps)

            assert result is not None
            assert hasattr(result.output, "characteristics")
            assert hasattr(result.output, "domain_signature")
            assert hasattr(result.output.characteristics, "vocabulary_complexity_ratio")
            assert hasattr(result.output, "processing_config")

            print(
                "✅ PydanticAI Agent[UniversalDeps, UniversalDomainAnalysis]: Direct test successful"
            )
            print(f"   Agent Type: {type(domain_intelligence_agent)}")
            print(f"   Output Type: {type(result.output)}")
            print(f"   Domain Signature: {result.output.domain_signature}")
            print(
                f"   Vocabulary Complexity: {result.output.characteristics.vocabulary_complexity_ratio}"
            )

        except Exception as e:
            print(f"❌ PydanticAI Agent direct test failed: {e}")
            # Don't skip - we have real Azure services, let's see what's actually wrong
            import traceback

            traceback.print_exc()
            raise

    @pytest.mark.asyncio
    async def test_content_analysis(self):
        """Test domain intelligence content analysis."""
        from agents.domain_intelligence.agent import run_domain_analysis

        sample_content = """
        Python is a programming language that emphasizes readability and simplicity.
        Functions are defined using the def keyword, and classes use the class keyword.
        Popular frameworks include Django for web development and NumPy for data science.
        
        Example code:
        def hello_world():
            print("Hello, World!")
        """

        try:
            result = await run_domain_analysis(sample_content)

            assert result is not None
            assert hasattr(result, "characteristics")
            assert hasattr(result.characteristics, "vocabulary_complexity_ratio")
            assert hasattr(result.characteristics, "vocabulary_richness")
            assert 0 <= result.characteristics.vocabulary_complexity_ratio <= 1
            assert 0 <= result.characteristics.vocabulary_richness <= 1

            print("✅ Domain Intelligence Agent: Content analysis working")
            print(
                f"   Vocabulary Complexity: {result.characteristics.vocabulary_complexity_ratio:.3f}"
            )
            print(
                f"   Vocabulary Richness: {result.characteristics.vocabulary_richness:.3f}"
            )
            print(f"   Domain Signature: {result.domain_signature}")
            print(
                f"   Processing Complexity: {result.processing_config.processing_complexity}"
            )

        except Exception as e:
            print(f"⚠️ Domain Intelligence Agent analysis failed: {e}")
            # Don't fail the test if it's just an Azure connection issue
            if "404" in str(e) or "Resource not found" in str(e):
                pytest.skip(
                    "Azure OpenAI model configuration issue - service deployed but model access needs configuration"
                )
            else:
                raise


class TestKnowledgeExtractionAgent:
    """Test Knowledge Extraction Agent functionality."""

    @pytest.mark.asyncio
    async def test_agent_import(self):
        """Test that Knowledge Extraction Agent can be imported successfully."""
        try:
            from agents.knowledge_extraction.agent import knowledge_extraction_agent

            assert knowledge_extraction_agent is not None
            print("✅ Knowledge Extraction Agent: Import successful")
        except Exception as e:
            pytest.fail(f"Failed to import Knowledge Extraction Agent: {e}")

    @pytest.mark.asyncio
    async def test_pydantic_ai_agent_direct(self):
        """Test the PydanticAI Agent[UniversalDeps, ExtractionResult] object directly."""
        try:
            from agents.core.universal_deps import get_universal_deps
            from agents.knowledge_extraction.agent import knowledge_extraction_agent

            # Test agent object properties
            assert knowledge_extraction_agent is not None
            assert hasattr(knowledge_extraction_agent, "run")
            assert hasattr(knowledge_extraction_agent, "system_prompt")

            # Test agent can run with real dependencies
            deps = await get_universal_deps()

            sample_prompt = "Extract entities and relationships from: Azure Cosmos DB supports multiple data models"

            result = await knowledge_extraction_agent.run(sample_prompt, deps=deps)

            assert result is not None
            assert hasattr(result.output, "entities")
            assert hasattr(result.output, "relationships")

            print(
                "✅ PydanticAI Agent[UniversalDeps, ExtractionResult]: Direct test successful"
            )
            print(f"   Agent Type: {type(knowledge_extraction_agent)}")
            print(f"   Output Type: {type(result.output)}")

        except Exception as e:
            print(f"❌ PydanticAI Knowledge Extraction Agent direct test failed: {e}")
            import traceback

            traceback.print_exc()
            raise

    @pytest.mark.asyncio
    async def test_entity_extraction(self):
        """Test entity extraction functionality."""
        try:
            from agents.core.universal_deps import get_universal_deps
            from agents.core.universal_models import ExtractionRequest
            from agents.knowledge_extraction.agent import knowledge_extraction_agent

            # Test with sample text
            sample_text = """
            Microsoft Azure is a cloud computing platform. 
            It provides services like Azure OpenAI and Azure Cosmos DB.
            The platform supports Python applications and REST APIs.
            """

            deps = await get_universal_deps()

            # Format prompt as string for PydanticAI agent
            extraction_prompt = f"Extract entities and relationships from this content:\n\n{sample_text}"

            # Note: This may fail with current Azure OpenAI model configuration
            result = await knowledge_extraction_agent.run(extraction_prompt, deps=deps)

            assert result is not None
            print("✅ Knowledge Extraction Agent: Entity extraction working")

        except Exception as e:
            print(f"⚠️ Knowledge Extraction Agent failed: {e}")
            if "404" in str(e) or "Resource not found" in str(e):
                pytest.skip("Azure OpenAI model configuration issue")
            else:
                raise


class TestUniversalSearchAgent:
    """Test Universal Search Agent functionality."""

    @pytest.mark.asyncio
    async def test_agent_import(self):
        """Test that Universal Search Agent can be imported successfully."""
        try:
            from agents.universal_search.agent import universal_search_agent

            assert universal_search_agent is not None
            print("✅ Universal Search Agent: Import successful")
        except Exception as e:
            pytest.fail(f"Failed to import Universal Search Agent: {e}")

    @pytest.mark.asyncio
    async def test_pydantic_ai_agent_direct(self):
        """Test the PydanticAI Agent[UniversalDeps, MultiModalSearchResult] object directly."""
        try:
            from agents.core.universal_deps import get_universal_deps
            from agents.universal_search.agent import universal_search_agent

            # Test agent object properties
            assert universal_search_agent is not None
            assert hasattr(universal_search_agent, "run")
            assert hasattr(universal_search_agent, "system_prompt")

            # Test agent can run with real dependencies
            deps = await get_universal_deps()

            sample_prompt = (
                "Search for: Azure Cosmos DB performance optimization techniques"
            )

            result = await universal_search_agent.run(sample_prompt, deps=deps)

            assert result is not None
            assert hasattr(result.output, "unified_results")
            assert hasattr(result.output, "search_strategy_used")

            print(
                "✅ PydanticAI Agent[UniversalDeps, MultiModalSearchResult]: Direct test successful"
            )
            print(f"   Agent Type: {type(universal_search_agent)}")
            print(f"   Output Type: {type(result.output)}")

        except Exception as e:
            print(f"❌ PydanticAI Universal Search Agent direct test failed: {e}")
            import traceback

            traceback.print_exc()
            raise


class TestAgentIntegration:
    """Test agent integration and orchestration."""

    def test_universal_deps_initialization(self):
        """Test that universal dependencies can be initialized."""
        try:
            from agents.core.universal_deps import (
                UniversalDeps,
                get_universal_deps_sync,
            )

            deps = get_universal_deps_sync()
            assert isinstance(deps, UniversalDeps)
            assert hasattr(deps, "openai_client")  # Correct attribute name
            assert hasattr(deps, "search_client")
            assert hasattr(deps, "cosmos_client")
            assert hasattr(deps, "storage_client")

            print("✅ Universal Dependencies: Initialization successful")

        except Exception as e:
            pytest.fail(f"Failed to initialize universal dependencies: {e}")

    def test_universal_models_import(self):
        """Test that universal models can be imported."""
        try:
            from agents.core.universal_models import (
                ExtractionRequest,
                ExtractionResult,
                SearchRequest,
                SearchResult,
                UniversalDomainAnalysis,
                UniversalDomainCharacteristics,
            )

            # Test that models can be instantiated with correct required fields
            characteristics = UniversalDomainCharacteristics(
                avg_document_length=1500,
                document_count=100,
                vocabulary_richness=0.75,
                sentence_complexity=12.5,
                lexical_diversity=0.65,
                vocabulary_complexity_ratio=0.45,
                structural_consistency=0.85,
            )

            assert characteristics.avg_document_length == 1500
            assert characteristics.document_count == 100
            assert characteristics.vocabulary_richness == 0.75

            print("✅ Universal Models: All models imported and working")

        except Exception as e:
            pytest.fail(f"Failed to import universal models: {e}")


class TestDataProcessing:
    """Test data processing capabilities with real test data."""

    def test_test_data_availability(self):
        """Test that test data corpus is available."""
        data_dir = Path("/workspace/azure-maintie-rag/data/raw")
        assert data_dir.exists(), "Test data directory not found"

        # Check for Azure AI Language Service files
        azure_files = list(data_dir.rglob("*.md"))
        assert len(azure_files) > 0, "No markdown files found in test data"

        print(f"✅ Test Data: Found {len(azure_files)} test files")

        # Test that files can be read
        sample_file = azure_files[0]
        content = sample_file.read_text(encoding="utf-8")
        assert len(content) > 100, "Test files appear to be empty or too small"

        print(f"✅ Test Data: Files readable, sample size: {len(content)} chars")

    def test_data_processing_pipeline_structure(self):
        """Test that data processing pipeline components exist."""
        pipeline_scripts = [
            "scripts/dataflow/00_check_azure_state.py",
            "scripts/dataflow/01_data_ingestion.py",
            "scripts/dataflow/02_knowledge_extraction.py",
            "scripts/dataflow/07_unified_search.py",
        ]

        base_path = Path("/workspace/azure-maintie-rag")
        missing_scripts = []

        for script in pipeline_scripts:
            script_path = base_path / script
            if not script_path.exists():
                missing_scripts.append(script)

        assert not missing_scripts, f"Missing pipeline scripts: {missing_scripts}"
        print("✅ Data Pipeline: All core scripts present")
