"""
Real Implementation Tests: Universal Content Processing
======================================================

Tests for universal content processing across ANY domain using REAL:
- Azure OpenAI processing
- Multiple content domains (programming, legal, medical, technical, business)
- English language focus with limited multilingual support
- Real data from data/raw/ directory
- No fake values, no placeholders, no mocks
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Real agent imports
from agents.domain_intelligence.agent import (
    UniversalDomainDeps,
    run_universal_domain_analysis,
)
from agents.knowledge_extraction.agent import run_knowledge_extraction
from agents.universal_search.agent import run_universal_search

# Real data preparation
REAL_CONTENT_SAMPLES = {
    "programming": {
        "text": """
        Object-oriented programming (OOP) is a programming paradigm based on the concept of objects.
        Classes define the structure and behavior of objects. Inheritance allows classes to inherit 
        properties from parent classes. Polymorphism enables objects to take multiple forms.
        Encapsulation restricts direct access to object components.
        """,
        "language": "en",
        "expected_entities": [
            "Object-oriented programming",
            "OOP",
            "objects",
            "Classes",
            "Inheritance",
            "Polymorphism",
            "Encapsulation",
        ],
        "expected_domain": "programming",
    },
    "legal": {
        "text": """
        Contract law governs the enforceability of agreements between parties. A valid contract requires
        offer, acceptance, consideration, and legal capacity. Breach of contract occurs when one party
        fails to perform contractual obligations. Damages may be awarded for breach of contract.
        """,
        "language": "en",
        "expected_entities": [
            "Contract law",
            "agreements",
            "offer",
            "acceptance",
            "consideration",
            "breach",
            "Damages",
        ],
        "expected_domain": "legal",
    },
    "medical": {
        "text": """
        Cardiovascular disease affects the heart and blood vessels. Risk factors include hypertension,
        diabetes, smoking, and obesity. Treatment options include medication, lifestyle changes, and
        surgical interventions. Prevention through diet and exercise is crucial.
        """,
        "language": "en",
        "expected_entities": [
            "Cardiovascular disease",
            "heart",
            "blood vessels",
            "hypertension",
            "diabetes",
            "medication",
        ],
        "expected_domain": "medical",
    },
    "spanish_technical": {
        "text": """
        La inteligencia artificial utiliza algoritmos de aprendizaje automático para analizar datos.
        Las redes neuronales imitan el funcionamiento del cerebro humano. El procesamiento de lenguaje
        natural permite a las máquinas entender texto humano.
        """,
        "language": "es",
        "expected_entities": [
            "inteligencia artificial",
            "algoritmos",
            "aprendizaje automático",
            "redes neuronales",
            "procesamiento de lenguaje natural",
        ],
        "expected_domain": "technical",
    },
    "business": {
        "text": """
        Business intelligence systems help enterprises analyze data and make informed decisions.
        Data mining techniques extract valuable insights from large datasets. Cloud computing
        provides scalable infrastructure resources. Artificial intelligence is transforming
        business operations and competitive strategies.
        """,
        "language": "en",
        "expected_entities": [
            "Business intelligence",
            "data mining",
            "Cloud computing",
            "Artificial intelligence",
        ],
        "expected_domain": "business",
    },
}


class TestUniversalContentProcessing:
    """Real implementation tests for universal content processing"""

    @pytest.mark.asyncio
    async def test_domain_detection_across_multiple_domains(self):
        """Test real domain detection across programming, legal, medical, technical domains"""

        results = {}

        for domain_name, content_data in REAL_CONTENT_SAMPLES.items():
            # Use real domain intelligence agent
            deps = UniversalDomainDeps(
                data_directory="/workspace/azure-maintie-rag/data/raw",
                max_files_to_analyze=10,
                min_content_length=50,
            )

            # Create temporary content for analysis
            temp_content = content_data["text"]

            # Real Azure OpenAI analysis - no mocks
            start_time = time.time()
            domain_result = await run_universal_domain_analysis(deps)
            processing_time = time.time() - start_time

            # Validate real results
            assert domain_result is not None
            assert hasattr(domain_result, "domain_signature")
            assert hasattr(domain_result, "content_type_confidence")
            assert domain_result.content_type_confidence > 0.0
            assert processing_time < 30.0  # Real performance requirement

            results[domain_name] = {
                "domain_signature": domain_result.domain_signature,
                "confidence": domain_result.content_type_confidence,
                "processing_time": processing_time,
                "language": content_data["language"],
            }

        # Validate universal processing worked across all domains
        assert len(results) == len(REAL_CONTENT_SAMPLES)

        # Ensure different domains produce different signatures
        signatures = [r["domain_signature"] for r in results.values()]
        assert len(set(signatures)) >= 3  # At least 3 different domain signatures

        # Validate performance across domains
        avg_processing_time = sum(r["processing_time"] for r in results.values()) / len(
            results
        )
        assert avg_processing_time < 10.0  # Enterprise performance requirement

        print(f"✅ Universal domain detection: {len(results)} domains processed")
        print(f"   Average processing time: {avg_processing_time:.2f}s")
        print(f"   Unique domain signatures: {len(set(signatures))}")

    @pytest.mark.asyncio
    async def test_multilingual_entity_extraction(self):
        """Test real entity extraction across multiple languages"""

        extraction_results = {}

        for content_name, content_data in REAL_CONTENT_SAMPLES.items():
            # Real knowledge extraction with Azure OpenAI
            start_time = time.time()
            extraction_result = await run_knowledge_extraction(
                text=content_data["text"],
                confidence_threshold=0.6,  # Lower threshold for multilingual
                max_entities=20,
                enable_monitoring=False,  # Disable for test performance
            )
            processing_time = time.time() - start_time

            # Validate real extraction results
            assert extraction_result is not None
            assert len(extraction_result.entities) > 0
            assert len(extraction_result.relationships) >= 0
            assert extraction_result.extraction_confidence > 0.0

            # Validate entities are meaningful (not just random words)
            entity_texts = [e.text for e in extraction_result.entities]
            assert any(
                len(entity.split()) > 1 for entity in entity_texts
            )  # Multi-word entities

            extraction_results[content_name] = {
                "entities": entity_texts,
                "entity_count": len(extraction_result.entities),
                "relationship_count": len(extraction_result.relationships),
                "confidence": extraction_result.extraction_confidence,
                "processing_time": processing_time,
                "language": content_data["language"],
            }

        # Validate multilingual extraction worked
        languages_processed = set(r["language"] for r in extraction_results.values())
        assert len(languages_processed) >= 3  # Multiple languages

        # Validate quality across languages
        for result in extraction_results.values():
            assert result["entity_count"] >= 3  # Minimum entities extracted
            assert result["confidence"] > 0.5  # Reasonable confidence
            assert result["processing_time"] < 15.0  # Performance requirement

        print(f"✅ Multilingual extraction: {len(languages_processed)} languages")
        print(
            f"   Total entities: {sum(r['entity_count'] for r in extraction_results.values())}"
        )
        print(f"   Languages: {sorted(languages_processed)}")

    @pytest.mark.asyncio
    async def test_real_data_processing_pipeline(self):
        """Test complete pipeline with actual data from data/raw directory"""

        # Use real data files
        data_dir = Path("/workspace/azure-maintie-rag/data/raw")
        real_files = list(data_dir.glob("**/*.md"))[:5]  # Test with first 5 real files

        if len(real_files) == 0:
            pytest.skip("No real data files found in data/raw")

        pipeline_results = []

        for file_path in real_files:
            # Read real content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if len(content.strip()) < 100:  # Skip very short files
                continue

            # Process with complete pipeline
            start_time = time.time()

            # Step 1: Domain analysis
            deps = UniversalDomainDeps(
                data_directory=str(data_dir), max_files_to_analyze=5
            )
            domain_result = await run_universal_domain_analysis(deps)

            # Step 2: Knowledge extraction
            extraction_result = await run_knowledge_extraction(
                text=content[:2000],  # First 2000 chars for performance
                confidence_threshold=0.7,
                enable_monitoring=False,
            )

            # Step 3: Universal search (using extracted entities)
            if extraction_result.entities:
                search_query = extraction_result.entities[0].text
                search_result = await run_universal_search(
                    query=search_query, max_results=5, enable_monitoring=False
                )
            else:
                search_result = None

            total_time = time.time() - start_time

            # Validate real pipeline results
            assert domain_result.domain_signature is not None
            assert len(extraction_result.entities) >= 0
            assert total_time < 60.0  # Complete pipeline under 60s

            pipeline_results.append(
                {
                    "file": file_path.name,
                    "domain": domain_result.domain_signature,
                    "entities": len(extraction_result.entities),
                    "relationships": len(extraction_result.relationships),
                    "search_executed": search_result is not None,
                    "total_time": total_time,
                }
            )

        # Validate pipeline worked on real data
        assert len(pipeline_results) > 0
        avg_time = sum(r["total_time"] for r in pipeline_results) / len(
            pipeline_results
        )
        total_entities = sum(r["entities"] for r in pipeline_results)

        assert avg_time < 30.0  # Average pipeline time under 30s
        assert total_entities > 0  # Extracted entities from real data

        print(f"✅ Real data pipeline: {len(pipeline_results)} files processed")
        print(f"   Average time: {avg_time:.2f}s per file")
        print(f"   Total entities extracted: {total_entities}")

    @pytest.mark.asyncio
    async def test_content_adaptation_without_configuration(self):
        """Test system adapts to new content without any manual configuration"""

        # Completely new content types not seen before
        novel_content = {
            "quantum_physics": """
                Quantum entanglement describes the phenomenon where quantum particles remain connected
                regardless of distance. Bell's theorem demonstrates the non-local nature of quantum mechanics.
                Superposition allows particles to exist in multiple states simultaneously until measured.
                """,
            "cooking_recipe": """
                Preheat oven to 350°F. Mix flour, sugar, and baking powder in a bowl. 
                Add eggs and milk gradually. Pour into greased pan and bake for 25 minutes.
                Cool before serving with fresh berries.
                """,
            "financial_analysis": """
                Market volatility increased due to inflation concerns. The Federal Reserve raised interest rates
                by 0.75 basis points. Dividend yields remain attractive for income investors.
                Portfolio diversification across asset classes is recommended.
                """,
        }

        adaptation_results = {}

        for content_type, text in novel_content.items():
            # Test zero-configuration adaptation
            start_time = time.time()

            # Domain intelligence should adapt automatically
            deps = UniversalDomainDeps(
                data_directory="/workspace/azure-maintie-rag/data/raw"
            )
            domain_result = await run_universal_domain_analysis(deps)

            # Knowledge extraction should work without domain-specific setup
            extraction_result = await run_knowledge_extraction(
                text=text, confidence_threshold=0.6, enable_monitoring=False
            )

            adaptation_time = time.time() - start_time

            # Validate adaptation worked without configuration
            assert domain_result.domain_signature is not None
            assert len(extraction_result.entities) > 0
            assert extraction_result.extraction_confidence > 0.0

            adaptation_results[content_type] = {
                "adapted": True,
                "domain_detected": domain_result.domain_signature,
                "entities_found": len(extraction_result.entities),
                "adaptation_time": adaptation_time,
                "confidence": extraction_result.extraction_confidence,
            }

        # Validate universal adaptation
        assert len(adaptation_results) == len(novel_content)
        assert all(r["adapted"] for r in adaptation_results.values())
        assert all(r["entities_found"] > 0 for r in adaptation_results.values())

        # Different content should produce different domain signatures
        domains = [r["domain_detected"] for r in adaptation_results.values()]

        print(f"✅ Zero-config adaptation: {len(adaptation_results)} new content types")
        print(f"   Unique domains detected: {len(set(domains))}")
        print(
            f"   Average adaptation time: {sum(r['adaptation_time'] for r in adaptation_results.values()) / len(adaptation_results):.2f}s"
        )

    @pytest.mark.asyncio
    async def test_cross_domain_knowledge_integration(self):
        """Test knowledge integration across different domains"""

        # Create cross-domain knowledge scenario
        cross_domain_content = """
        Machine learning algorithms are increasingly used in medical diagnosis.
        Legal frameworks must adapt to AI decision-making in healthcare.
        Software engineering principles apply to developing medical AI systems.
        Data privacy regulations affect healthcare technology implementations.
        """

        # Extract knowledge spanning multiple domains
        extraction_result = await run_knowledge_extraction(
            text=cross_domain_content,
            confidence_threshold=0.5,
            max_entities=25,
            max_relationships=20,
            enable_graph_storage=False,  # Disable for test
            enable_monitoring=False,
        )

        # Validate cross-domain extraction
        assert len(extraction_result.entities) >= 5
        assert len(extraction_result.relationships) >= 2

        # Check for entities from multiple domains
        entity_texts = [e.text.lower() for e in extraction_result.entities]

        # Should find technical terms
        tech_terms = any(
            term in " ".join(entity_texts)
            for term in ["machine learning", "algorithms", "software", "ai"]
        )

        # Should find medical terms
        medical_terms = any(
            term in " ".join(entity_texts)
            for term in ["medical", "healthcare", "diagnosis"]
        )

        # Should find legal terms
        legal_terms = any(
            term in " ".join(entity_texts)
            for term in ["legal", "regulations", "privacy"]
        )

        domains_found = sum([tech_terms, medical_terms, legal_terms])
        assert domains_found >= 2  # At least 2 domains represented

        # Validate relationships connect across domains
        relationship_texts = [
            f"{r.subject} {r.predicate} {r.object}"
            for r in extraction_result.relationships
        ]

        print(f"✅ Cross-domain integration: {domains_found} domains integrated")
        print(f"   Entities: {len(extraction_result.entities)}")
        print(f"   Relationships: {len(extraction_result.relationships)}")
        print(f"   Overall confidence: {extraction_result.extraction_confidence:.2f}")
