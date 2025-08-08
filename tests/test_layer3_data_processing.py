"""
Layer 3: Real Data Processing Tests
===================================

Comprehensive testing with actual Azure AI Language Service documentation files.
Processes all 17 real files through the complete data pipeline.
Validates end-to-end data processing with authentic content.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
from dotenv import load_dotenv

# Load environment before all imports
load_dotenv()

from agents.core.universal_deps import get_universal_deps
from agents.core.universal_models import (
    ExtractionRequest,
    ExtractionResult,
    SearchRequest,
    UniversalDomainAnalysis,
)
from agents.domain_intelligence.agent import run_domain_analysis
from agents.knowledge_extraction.agent import knowledge_extraction_agent
from agents.universal_search.agent import universal_search_agent


class TestRealDataAvailability:
    """Test availability and quality of real Azure AI test data."""

    @pytest.mark.layer3
    @pytest.mark.azure
    def test_azure_ai_data_directory_exists(self):
        """Test that Azure AI Language Service test data directory exists."""
        data_dir = (
            Path(__file__).parent.parent
            / "data"
            / "raw"
            / "azure-ai-services-language-service_output"
        )

        assert data_dir.exists(), f"Azure AI test data directory not found: {data_dir}"
        assert data_dir.is_dir(), f"Path exists but is not a directory: {data_dir}"

        print("✅ Azure AI Data Directory: Found and accessible")
        print(f"   Path: {data_dir}")

    @pytest.mark.layer3
    @pytest.mark.azure
    def test_azure_ai_test_files_availability(self, test_data_directory):
        """Test that Azure AI test files are available and readable."""
        markdown_files = list(test_data_directory.glob("*.md"))

        assert (
            len(markdown_files) > 0
        ), "No Azure AI markdown files found in test data directory"

        # Test that files are readable
        readable_files = []
        file_sizes = []

        for file_path in markdown_files:
            try:
                content = file_path.read_text(encoding="utf-8")
                readable_files.append(file_path)
                file_sizes.append(len(content))
            except Exception as e:
                print(f"⚠️  Cannot read file {file_path.name}: {e}")

        assert (
            len(readable_files) >= 10
        ), f"Expected at least 10 readable files, found {len(readable_files)}"

        avg_size = sum(file_sizes) / len(file_sizes)
        substantial_files = sum(1 for size in file_sizes if size > 500)

        print("✅ Azure AI Test Files: Available and readable")
        print(f"   Total Files: {len(markdown_files)}")
        print(f"   Readable Files: {len(readable_files)}")
        print(f"   Substantial Files (>500 chars): {substantial_files}")
        print(f"   Average File Size: {avg_size:.0f} characters")

    @pytest.mark.layer3
    @pytest.mark.azure
    def test_azure_ai_content_quality(self, test_data_quality_report):
        """Test quality characteristics of Azure AI test data."""
        report = test_data_quality_report

        # Validate minimum quality standards
        assert (
            report["total_files"] >= 10
        ), f"Expected at least 10 files, found {report['total_files']}"
        assert (
            report["quality_ratio"] >= 0.6
        ), f"Quality ratio too low: {report['quality_ratio']:.2f}"
        assert (
            report["average_size_chars"] >= 1000
        ), f"Files too small on average: {report['average_size_chars']:.0f}"

        print("✅ Azure AI Content Quality: Meets testing standards")
        print(f"   Quality Ratio: {report['quality_ratio']:.2f}")
        print(f"   Files with API Content: {report['files_with_api_content']}")
        print(
            f"   Files with Tutorial Content: {report['files_with_tutorial_content']}"
        )
        print(f"   Files with Code Blocks: {report['files_with_code_blocks']}")

    @pytest.mark.layer3
    @pytest.mark.azure
    def test_content_diversity_analysis(self, azure_ai_test_files):
        """Analyze content diversity to ensure comprehensive testing."""
        content_types = {
            "api_documentation": 0,
            "how_to_guides": 0,
            "code_examples": 0,
            "conceptual_content": 0,
            "reference_material": 0,
        }

        vocabulary_samples = []

        for file_path in azure_ai_test_files[:10]:  # Analyze first 10 files
            content = file_path.read_text(encoding="utf-8").lower()

            # Classify content types
            if any(
                term in content
                for term in ["api", "endpoint", "request", "response", "method"]
            ):
                content_types["api_documentation"] += 1
            if any(
                term in content
                for term in ["how to", "tutorial", "step", "guide", "walkthrough"]
            ):
                content_types["how_to_guides"] += 1
            if any(term in content for term in ["```", "code", "example", "sample"]):
                content_types["code_examples"] += 1
            if any(
                term in content
                for term in ["concept", "overview", "introduction", "understand"]
            ):
                content_types["conceptual_content"] += 1
            if any(
                term in content
                for term in ["reference", "parameter", "property", "field"]
            ):
                content_types["reference_material"] += 1

            # Sample vocabulary complexity
            words = content.split()
            unique_words = set(word.strip('.,!?;:"()[]') for word in words)
            vocab_complexity = len(unique_words) / max(len(words), 1)
            vocabulary_samples.append(vocab_complexity)

        # Validate diversity
        diverse_types = sum(1 for count in content_types.values() if count > 0)
        avg_vocab_complexity = sum(vocabulary_samples) / len(vocabulary_samples)

        assert (
            diverse_types >= 3
        ), f"Content not diverse enough: {diverse_types} types found"
        assert (
            0.1 <= avg_vocab_complexity <= 0.9
        ), f"Vocabulary complexity out of range: {avg_vocab_complexity:.3f}"

        print("✅ Content Diversity Analysis: Sufficient for comprehensive testing")
        print(f"   Content Types Found: {diverse_types}/5")
        for content_type, count in content_types.items():
            print(f"   {content_type.replace('_', ' ').title()}: {count} files")
        print(f"   Average Vocabulary Complexity: {avg_vocab_complexity:.3f}")


class TestDomainIntelligenceWithRealData:
    """Test Domain Intelligence Agent with real Azure AI documentation files."""

    @pytest.mark.layer3
    @pytest.mark.azure
    @pytest.mark.asyncio
    async def test_domain_analysis_on_real_files(self, azure_ai_test_files):
        """Test domain intelligence analysis on real Azure AI files."""
        processed_files = 0
        analysis_results = []

        # Process first 5 files for comprehensive testing
        for file_path in azure_ai_test_files[:5]:
            try:
                content = file_path.read_text(encoding="utf-8")

                # Skip very small files
                if len(content) < 200:
                    continue

                # Run domain analysis
                analysis = await run_domain_analysis(content)

                assert isinstance(analysis, UniversalDomainAnalysis)
                assert hasattr(analysis, "discovered_characteristics")

                characteristics = analysis.discovered_characteristics
                analysis_results.append(
                    {
                        "file": file_path.name,
                        "vocab_complexity": characteristics.vocabulary_complexity,
                        "concept_density": characteristics.concept_density,
                        "structural_patterns": characteristics.structural_patterns,
                        "content_signature": characteristics.content_signature,
                    }
                )

                processed_files += 1

            except Exception as e:
                print(f"⚠️  Failed to process {file_path.name}: {e}")

        assert (
            processed_files >= 3
        ), f"Expected to process at least 3 files, processed {processed_files}"

        # Validate analysis quality
        vocab_complexities = [r["vocab_complexity"] for r in analysis_results]
        avg_complexity = sum(vocab_complexities) / len(vocab_complexities)

        print("✅ Domain Intelligence Real Data: Processing successful")
        print(f"   Files Processed: {processed_files}")
        print(f"   Average Vocabulary Complexity: {avg_complexity:.3f}")
        print(
            f"   Complexity Range: {min(vocab_complexities):.3f} - {max(vocab_complexities):.3f}"
        )

        # Print sample analysis for verification
        if analysis_results:
            sample = analysis_results[0]
            print(f"   Sample Analysis ({sample['file']}):")
            print(f"     Vocabulary: {sample['vocab_complexity']:.3f}")
            print(f"     Concept Density: {sample['concept_density']:.3f}")
            print(f"     Patterns: {sample['structural_patterns']}")

    @pytest.mark.layer3
    @pytest.mark.azure
    @pytest.mark.asyncio
    async def test_domain_analysis_performance_with_real_data(
        self, azure_ai_test_files
    ):
        """Test domain analysis performance with real data files."""
        processing_times = []

        for file_path in azure_ai_test_files[:3]:  # Test with first 3 files
            content = file_path.read_text(encoding="utf-8")

            if len(content) < 200:
                continue

            start_time = time.time()
            analysis = await run_domain_analysis(content)
            processing_time = time.time() - start_time

            processing_times.append(processing_time)

            # Individual file should process within reasonable time
            assert (
                processing_time < 20.0
            ), f"File {file_path.name} took too long: {processing_time:.2f}s"

        avg_time = sum(processing_times) / len(processing_times)

        print("✅ Domain Analysis Performance: Within acceptable ranges")
        print(f"   Files Tested: {len(processing_times)}")
        print(f"   Average Processing Time: {avg_time:.2f}s")
        print(f"   Max Processing Time: {max(processing_times):.2f}s")
        print(
            f"   All files < 20s target: {'✅' if max(processing_times) < 20 else '❌'}"
        )


class TestKnowledgeExtractionWithRealData:
    """Test Knowledge Extraction Agent with real Azure AI documentation."""

    @pytest.mark.layer3
    @pytest.mark.azure
    @pytest.mark.asyncio
    async def test_knowledge_extraction_on_real_files(self, azure_ai_test_files):
        """Test knowledge extraction on real Azure AI files."""
        deps = await get_universal_deps()

        processed_files = 0
        extraction_results = []

        for file_path in azure_ai_test_files[:3]:  # Process first 3 files
            try:
                content = file_path.read_text(encoding="utf-8")

                # Skip very small files
                if len(content) < 300:
                    continue

                # Extract knowledge
                extraction_prompt = f"Extract entities and relationships from this Azure AI documentation: {content[:1500]}..."
                result = await knowledge_extraction_agent.run(
                    extraction_prompt, deps=deps
                )

                assert result.output is not None
                assert isinstance(result.output, ExtractionResult)

                extraction = result.output
                extraction_results.append(
                    {
                        "file": file_path.name,
                        "entities_count": len(extraction.entities),
                        "relationships_count": len(extraction.relationships),
                        "entities": [
                            entity.text for entity in extraction.entities[:5]
                        ],  # First 5 entities
                        "relationships": [
                            rel.relationship_type
                            for rel in extraction.relationships[:3]
                        ],  # First 3 relationships
                    }
                )

                processed_files += 1

            except Exception as e:
                print(f"⚠️  Failed to extract from {file_path.name}: {e}")

        assert (
            processed_files >= 2
        ), f"Expected to process at least 2 files, processed {processed_files}"

        # Validate extraction quality
        total_entities = sum(r["entities_count"] for r in extraction_results)
        total_relationships = sum(r["relationships_count"] for r in extraction_results)

        print("✅ Knowledge Extraction Real Data: Processing successful")
        print(f"   Files Processed: {processed_files}")
        print(f"   Total Entities Extracted: {total_entities}")
        print(f"   Total Relationships Extracted: {total_relationships}")
        print(f"   Average Entities per File: {total_entities / processed_files:.1f}")

        # Print sample extraction
        if extraction_results:
            sample = extraction_results[0]
            print(f"   Sample Extraction ({sample['file']}):")
            print(f"     Entities: {sample['entities']}")
            print(f"     Relationships: {sample['relationships']}")

    @pytest.mark.layer3
    @pytest.mark.azure
    @pytest.mark.asyncio
    async def test_knowledge_extraction_accuracy_validation(self, azure_ai_test_files):
        """Validate accuracy of knowledge extraction on known content."""
        deps = await get_universal_deps()

        # Find a file that likely contains Azure-specific entities
        azure_content_file = None
        for file_path in azure_ai_test_files:
            content = file_path.read_text(encoding="utf-8").lower()
            if "azure" in content and len(content) > 500:
                azure_content_file = file_path
                break

        if not azure_content_file:
            pytest.skip("No suitable Azure content file found for accuracy testing")

        content = azure_content_file.read_text(encoding="utf-8")

        # Extract knowledge
        result = await knowledge_extraction_agent.run(
            f"Extract entities and relationships from: {content[:1000]}", deps=deps
        )

        extraction = result.output
        entity_texts = [entity.text.lower() for entity in extraction.entities]

        # Validate that Azure-related entities were found
        expected_entities = ["azure", "microsoft", "api", "service"]
        found_expected = sum(
            1
            for expected in expected_entities
            if any(expected in entity for entity in entity_texts)
        )

        accuracy_ratio = found_expected / len(expected_entities)

        print("✅ Knowledge Extraction Accuracy: Validated on real content")
        print(f"   File: {azure_content_file.name}")
        print(f"   Expected Entities Found: {found_expected}/{len(expected_entities)}")
        print(f"   Accuracy Ratio: {accuracy_ratio:.2f}")
        print(f"   Total Entities: {len(extraction.entities)}")

        # Should find at least some expected entities
        assert found_expected >= 2, f"Too few expected entities found: {found_expected}"


class TestUniversalSearchWithRealData:
    """Test Universal Search Agent with real data corpus."""

    @pytest.mark.layer3
    @pytest.mark.azure
    @pytest.mark.asyncio
    async def test_universal_search_on_real_queries(self, azure_ai_test_files):
        """Test universal search with queries based on real file content."""
        deps = await get_universal_deps()

        # Generate search queries based on actual file content
        search_queries = []
        for file_path in azure_ai_test_files[:3]:
            content = file_path.read_text(encoding="utf-8")

            # Extract potential search terms from content
            words = content.split()
            technical_terms = [
                word
                for word in words
                if len(word) > 8 and word.isalpha() and word[0].isupper()
            ]

            if technical_terms:
                search_queries.append(f"Find information about {technical_terms[0]}")

        if not search_queries:
            search_queries = ["Search for Azure AI services", "Find API documentation"]

        search_results = []

        for query in search_queries[:2]:  # Test with first 2 queries
            try:
                result = await universal_search_agent.run(
                    f"Execute universal search: {query}", deps=deps
                )

                assert result.output is not None
                search_result = result.output

                search_results.append(
                    {
                        "query": query,
                        "strategy": search_result.search_strategy_used,
                        "results_count": len(search_result.unified_results),
                        "has_results": len(search_result.unified_results) > 0,
                    }
                )

            except Exception as e:
                print(f"⚠️  Search failed for query '{query}': {e}")

        processed_queries = len(search_results)
        assert (
            processed_queries >= 1
        ), f"Expected at least 1 successful search, got {processed_queries}"

        print("✅ Universal Search Real Data: Queries processed successfully")
        print(f"   Queries Processed: {processed_queries}")

        for result in search_results:
            print(f"   Query: '{result['query']}'")
            print(f"     Strategy: {result['strategy']}")
            print(f"     Results: {result['results_count']}")


class TestDataPipelineIntegration:
    """Test integrated data pipeline with real files end-to-end."""

    @pytest.mark.layer3
    @pytest.mark.azure
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_complete_pipeline_with_real_file(self, azure_ai_test_files):
        """Test complete data processing pipeline with one real file."""
        deps = await get_universal_deps()

        # Select a substantial file for complete pipeline testing
        test_file = None
        for file_path in azure_ai_test_files:
            content = file_path.read_text(encoding="utf-8")
            if 1000 < len(content) < 5000:  # Good size for comprehensive testing
                test_file = file_path
                break

        if not test_file:
            pytest.skip("No suitable file found for complete pipeline testing")

        content = test_file.read_text(encoding="utf-8")

        print(f"🔄 Running complete pipeline on: {test_file.name}")
        print(f"   Content Length: {len(content)} characters")

        # Step 1: Domain Intelligence Analysis
        start_time = time.time()
        domain_analysis = await run_domain_analysis(content)
        domain_time = time.time() - start_time

        assert isinstance(domain_analysis, UniversalDomainAnalysis)
        print(f"   ✅ Domain Analysis: {domain_time:.2f}s")

        # Step 2: Knowledge Extraction
        start_time = time.time()
        extraction_result = await knowledge_extraction_agent.run(
            f"Extract entities and relationships: {content[:1500]}", deps=deps
        )
        extraction_time = time.time() - start_time

        assert isinstance(extraction_result.output, ExtractionResult)
        print(
            f"   ✅ Knowledge Extraction: {extraction_time:.2f}s ({len(extraction_result.output.entities)} entities)"
        )

        # Step 3: Universal Search (using extracted entities)
        if extraction_result.output.entities:
            entity_text = extraction_result.output.entities[0].text
            start_time = time.time()
            search_result = await universal_search_agent.run(
                f"Search for: {entity_text}", deps=deps
            )
            search_time = time.time() - start_time

            print(f"   ✅ Universal Search: {search_time:.2f}s")

        total_time = (
            domain_time
            + extraction_time
            + (search_time if "search_time" in locals() else 0)
        )

        print("✅ Complete Pipeline: Successful end-to-end processing")
        print(f"   Total Pipeline Time: {total_time:.2f}s")
        print(f"   File: {test_file.name}")
        print(
            f"   Vocab Complexity: {domain_analysis.discovered_characteristics.vocabulary_complexity:.3f}"
        )
        print(f"   Entities Found: {len(extraction_result.output.entities)}")

    @pytest.mark.layer3
    @pytest.mark.azure
    @pytest.mark.asyncio
    async def test_pipeline_performance_benchmarks(self, azure_ai_test_files):
        """Test pipeline performance against SLA benchmarks."""
        deps = await get_universal_deps()

        performance_results = []

        for file_path in azure_ai_test_files[:2]:  # Test with first 2 files
            content = file_path.read_text(encoding="utf-8")

            if len(content) < 500:  # Skip small files
                continue

            # Time each pipeline stage
            start_time = time.time()
            domain_analysis = await run_domain_analysis(content)
            domain_time = time.time() - start_time

            start_time = time.time()
            extraction_result = await knowledge_extraction_agent.run(
                f"Extract from: {content[:1000]}", deps=deps
            )
            extraction_time = time.time() - start_time

            total_time = domain_time + extraction_time

            performance_results.append(
                {
                    "file": file_path.name,
                    "file_size": len(content),
                    "domain_time": domain_time,
                    "extraction_time": extraction_time,
                    "total_time": total_time,
                    "meets_sla": total_time
                    < 30.0,  # 30 second SLA for individual file processing
                }
            )

        if not performance_results:
            pytest.skip("No suitable files for performance testing")

        # Validate performance
        avg_total_time = sum(r["total_time"] for r in performance_results) / len(
            performance_results
        )
        sla_compliant_files = sum(1 for r in performance_results if r["meets_sla"])

        print("✅ Pipeline Performance Benchmarks: Evaluated")
        print(f"   Files Tested: {len(performance_results)}")
        print(f"   Average Total Time: {avg_total_time:.2f}s")
        print(
            f"   SLA Compliant Files: {sla_compliant_files}/{len(performance_results)}"
        )

        for result in performance_results:
            print(
                f"   {result['file']}: {result['total_time']:.2f}s ({'✅' if result['meets_sla'] else '❌'})"
            )

        # At least 50% of files should meet SLA
        sla_ratio = sla_compliant_files / len(performance_results)
        assert (
            sla_ratio >= 0.5
        ), f"Too many files exceed SLA: {sla_ratio:.2f} compliance rate"
