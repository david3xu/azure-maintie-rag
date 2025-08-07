"""
Real Implementation Tests: Knowledge Graph Intelligence
======================================================

Tests for knowledge graph intelligence and reasoning using REAL:
- Graph Neural Network (GNN) training and inference
- Knowledge graph construction and validation
- Relationship extraction and entity linking
- Graph-based reasoning and pattern recognition
- Cross-domain knowledge integration
- No fake values, no placeholders, no mocks
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from agents.domain_intelligence.agent import (
    UniversalDomainDeps,
    run_universal_domain_analysis,
)

# Real agent imports
from agents.knowledge_extraction.agent import run_knowledge_extraction
from agents.universal_search.agent import run_universal_search

# Real infrastructure imports
from infrastructure.azure_cosmos import SimpleCosmosGremlinClient
from infrastructure.azure_ml import GNNInferenceClient, GNNTrainingClient
from infrastructure.azure_monitoring import AppInsightsClient
from infrastructure.azure_search import UnifiedSearchClient

# Knowledge graph intelligence benchmarks (real production requirements)
GRAPH_INTELLIGENCE_SLA = {
    "entity_extraction_accuracy": 0.85,  # 85% entity extraction accuracy
    "relationship_precision": 0.80,  # 80% relationship precision
    "graph_construction_time": 30.0,  # Graph construction under 30s
    "gnn_training_time": 300.0,  # GNN training under 5 minutes
    "gnn_inference_time": 2.0,  # GNN inference under 2s
    "graph_traversal_time": 1.5,  # Graph queries under 1.5s
    "knowledge_integration_score": 0.75,  # 75% knowledge integration quality
    "reasoning_accuracy": 0.70,  # 70% reasoning accuracy
}


class TestKnowledgeGraphIntelligence:
    """Real implementation tests for knowledge graph intelligence"""

    @pytest.mark.asyncio
    async def test_knowledge_graph_construction_from_real_data(self):
        """Test knowledge graph construction from real document data"""

        # Use real documents for graph construction
        data_dir = Path("/workspace/azure-maintie-rag/data/raw")
        real_files = list(data_dir.glob("**/*.md"))[:10]  # Use first 10 real files

        if len(real_files) == 0:
            pytest.skip("No real data files found for graph construction")

        graph_construction_results = []
        cosmos_client = SimpleCosmosGremlinClient()
        monitoring_client = AppInsightsClient()

        total_entities = 0
        total_relationships = 0

        for file_path in real_files:
            # Read real content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if len(content.strip()) < 200:  # Skip very short files
                continue

            construction_start = time.time()

            # Extract knowledge from real content
            extraction_result = await run_knowledge_extraction(
                text=content[:3000],  # First 3000 chars for performance
                confidence_threshold=0.7,
                max_entities=30,
                max_relationships=25,
                enable_graph_storage=True,  # Enable real graph storage
                enable_monitoring=True,
                enable_entity_linking=True,  # Enable entity linking
            )

            construction_time = time.time() - construction_start

            # Validate knowledge extraction quality
            assert extraction_result is not None
            assert len(extraction_result.entities) >= 5  # Minimum entities
            assert len(extraction_result.relationships) >= 3  # Minimum relationships
            assert extraction_result.extraction_confidence >= 0.6
            assert (
                construction_time <= GRAPH_INTELLIGENCE_SLA["graph_construction_time"]
            )

            # Validate entity quality (entities should have meaningful names)
            entity_texts = [e.text for e in extraction_result.entities]
            meaningful_entities = [
                e for e in entity_texts if len(e.split()) >= 1 and len(e) >= 3
            ]
            entity_quality = (
                len(meaningful_entities) / len(entity_texts) if entity_texts else 0
            )

            assert entity_quality >= 0.8  # 80% of entities should be meaningful

            # Validate relationship quality
            relationship_quality = 0.0
            if extraction_result.relationships:
                valid_relationships = [
                    r
                    for r in extraction_result.relationships
                    if r.subject
                    and r.predicate
                    and r.object
                    and len(r.subject) >= 2
                    and len(r.object) >= 2
                ]
                relationship_quality = len(valid_relationships) / len(
                    extraction_result.relationships
                )

            assert relationship_quality >= 0.7  # 70% of relationships should be valid

            total_entities += len(extraction_result.entities)
            total_relationships += len(extraction_result.relationships)

            # Track graph construction metrics
            await monitoring_client.track_custom_event(
                event_name="knowledge_graph_constructed",
                properties={
                    "source_file": file_path.name,
                    "content_length": len(content),
                    "extraction_method": "azure_openai",
                },
                measurements={
                    "construction_time_ms": construction_time * 1000,
                    "entities_extracted": len(extraction_result.entities),
                    "relationships_extracted": len(extraction_result.relationships),
                    "extraction_confidence": extraction_result.extraction_confidence,
                    "entity_quality_score": entity_quality,
                    "relationship_quality_score": relationship_quality,
                },
            )

            graph_construction_results.append(
                {
                    "file": file_path.name,
                    "entities": len(extraction_result.entities),
                    "relationships": len(extraction_result.relationships),
                    "construction_time": construction_time,
                    "confidence": extraction_result.extraction_confidence,
                    "entity_quality": entity_quality,
                    "relationship_quality": relationship_quality,
                }
            )

        # Validate overall graph construction
        assert len(graph_construction_results) >= 3  # Minimum processed files
        assert total_entities >= 50  # Minimum total entities
        assert total_relationships >= 30  # Minimum total relationships

        avg_construction_time = sum(
            r["construction_time"] for r in graph_construction_results
        ) / len(graph_construction_results)
        avg_confidence = sum(r["confidence"] for r in graph_construction_results) / len(
            graph_construction_results
        )
        avg_entity_quality = sum(
            r["entity_quality"] for r in graph_construction_results
        ) / len(graph_construction_results)

        assert (
            avg_construction_time <= GRAPH_INTELLIGENCE_SLA["graph_construction_time"]
        )
        assert avg_confidence >= 0.7
        assert (
            avg_entity_quality >= GRAPH_INTELLIGENCE_SLA["entity_extraction_accuracy"]
        )

        print(
            f"✅ Knowledge graph construction: {len(graph_construction_results)} documents processed"
        )
        print(f"   Total entities: {total_entities}")
        print(f"   Total relationships: {total_relationships}")
        print(f"   Average construction time: {avg_construction_time:.2f}s")
        print(f"   Average confidence: {avg_confidence:.2f}")

    @pytest.mark.asyncio
    async def test_gnn_training_and_inference_pipeline(self):
        """Test Graph Neural Network training and inference with real graph data"""

        # First, create a knowledge graph with sufficient data
        training_content = [
            """
            Machine learning algorithms require training data to learn patterns and make predictions.
            Neural networks consist of interconnected nodes that process information. Deep learning
            uses multiple layers to extract hierarchical features. Supervised learning uses labeled
            data to train models. Unsupervised learning finds patterns without labels.
            """,
            """
            Software engineering involves designing, developing, and maintaining software systems.
            Version control systems track changes in code over time. Continuous integration
            automatically builds and tests code changes. Agile methodologies emphasize iterative
            development and collaboration. Code reviews improve software quality and knowledge sharing.
            """,
            """
            Data science combines statistics, programming, and domain expertise to extract insights
            from data. Data preprocessing cleans and transforms raw data for analysis. Feature
            engineering creates meaningful variables for machine learning models. Data visualization
            communicates findings through charts and graphs. Statistical analysis validates hypotheses.
            """,
        ]

        gnn_training_client = GNNTrainingClient()
        gnn_inference_client = GNNInferenceClient()
        monitoring_client = AppInsightsClient()

        # Step 1: Build knowledge graph for training
        all_entities = []
        all_relationships = []

        graph_construction_start = time.time()

        for i, content in enumerate(training_content):
            extraction_result = await run_knowledge_extraction(
                text=content,
                confidence_threshold=0.6,
                max_entities=25,
                max_relationships=20,
                enable_graph_storage=True,
                enable_monitoring=False,  # Disable for training performance
            )

            all_entities.extend(extraction_result.entities)
            all_relationships.extend(extraction_result.relationships)

        graph_construction_time = time.time() - graph_construction_start

        # Validate sufficient graph data for training
        assert len(all_entities) >= 20  # Minimum entities for training
        assert len(all_relationships) >= 15  # Minimum relationships for training
        assert graph_construction_time <= 60.0  # Construction under 60s

        # Step 2: Train GNN model on extracted graph
        training_start = time.time()

        training_result = await gnn_training_client.train_model(
            entities=[
                {"id": e.id, "text": e.text, "type": e.entity_type}
                for e in all_entities
            ],
            relationships=[
                {
                    "subject": r.subject,
                    "predicate": r.predicate,
                    "object": r.object,
                    "confidence": r.confidence,
                }
                for r in all_relationships
            ],
            training_config={
                "epochs": 10,  # Limited epochs for testing
                "learning_rate": 0.01,
                "hidden_dim": 64,  # Smaller model for testing
                "dropout": 0.1,
            },
        )

        training_time = time.time() - training_start

        # Validate GNN training
        assert training_result is not None
        assert training_result.get("model_id") is not None
        assert (
            training_result.get("training_loss", float("inf")) < 2.0
        )  # Reasonable training loss
        assert training_time <= GRAPH_INTELLIGENCE_SLA["gnn_training_time"]

        model_id = training_result["model_id"]

        # Step 3: Test GNN inference on new queries
        inference_queries = [
            "machine learning model training process",
            "software development best practices",
            "data analysis and visualization techniques",
        ]

        inference_results = []

        for query in inference_queries:
            inference_start = time.time()

            # Use GNN for intelligent search/reasoning
            inference_result = await gnn_inference_client.predict(
                model_id=model_id,
                query_embedding=query,
                max_results=10,
                confidence_threshold=0.5,
            )

            inference_time = time.time() - inference_start

            # Validate GNN inference
            assert inference_result is not None
            assert (
                len(inference_result.get("predictions", [])) >= 3
            )  # Minimum predictions
            assert (
                inference_result.get("confidence", 0.0) >= 0.4
            )  # Reasonable confidence
            assert inference_time <= GRAPH_INTELLIGENCE_SLA["gnn_inference_time"]

            inference_results.append(
                {
                    "query": query,
                    "predictions": len(inference_result.get("predictions", [])),
                    "confidence": inference_result.get("confidence", 0.0),
                    "inference_time": inference_time,
                }
            )

        # Track GNN pipeline performance
        await monitoring_client.track_custom_event(
            event_name="gnn_training_inference_completed",
            properties={
                "model_id": model_id,
                "training_entities": len(all_entities),
                "training_relationships": len(all_relationships),
                "inference_queries": len(inference_queries),
            },
            measurements={
                "graph_construction_time_ms": graph_construction_time * 1000,
                "training_time_ms": training_time * 1000,
                "avg_inference_time_ms": sum(
                    r["inference_time"] for r in inference_results
                )
                / len(inference_results)
                * 1000,
                "training_loss": training_result.get("training_loss", 0.0),
                "avg_inference_confidence": sum(
                    r["confidence"] for r in inference_results
                )
                / len(inference_results),
            },
        )

        # Validate overall GNN pipeline
        avg_inference_time = sum(r["inference_time"] for r in inference_results) / len(
            inference_results
        )
        avg_confidence = sum(r["confidence"] for r in inference_results) / len(
            inference_results
        )

        assert avg_inference_time <= GRAPH_INTELLIGENCE_SLA["gnn_inference_time"]
        assert avg_confidence >= 0.5

        print(f"✅ GNN training and inference pipeline:")
        print(f"   Graph construction: {graph_construction_time:.2f}s")
        print(f"   GNN training: {training_time:.2f}s")
        print(f"   Average inference time: {avg_inference_time:.3f}s")
        print(f"   Average inference confidence: {avg_confidence:.2f}")
        print(f"   Training entities: {len(all_entities)}")
        print(f"   Training relationships: {len(all_relationships)}")

    @pytest.mark.asyncio
    async def test_graph_based_reasoning_patterns(self):
        """Test graph-based reasoning and pattern recognition capabilities"""

        # Create knowledge graph with reasoning test patterns
        reasoning_scenarios = [
            {
                "content": """
                Python is a programming language used for web development. Django is a Python framework
                for building web applications. Flask is another Python web framework that is lightweight.
                Both Django and Flask use Python syntax and libraries.
                """,
                "expected_patterns": [
                    ("Python", "is_used_for", "web development"),
                    ("Django", "is_framework_for", "Python"),
                    ("Flask", "is_framework_for", "Python"),
                    ("Django", "used_for", "web applications"),
                ],
                "reasoning_type": "hierarchical_relationships",
            },
            {
                "content": """
                Regular maintenance prevents system failures and extends equipment lifespan.
                Preventive maintenance is scheduled before problems occur. Corrective maintenance
                fixes problems after they happen. Predictive maintenance uses data to forecast
                when maintenance is needed. All maintenance types reduce downtime.
                """,
                "expected_patterns": [
                    ("maintenance", "prevents", "system failures"),
                    ("preventive maintenance", "is_type_of", "maintenance"),
                    ("corrective maintenance", "is_type_of", "maintenance"),
                    ("predictive maintenance", "is_type_of", "maintenance"),
                ],
                "reasoning_type": "causal_relationships",
            },
            {
                "content": """
                Cardiovascular disease affects the heart and blood vessels. High blood pressure
                increases cardiovascular risk. Regular exercise reduces cardiovascular risk.
                Healthy diet also reduces cardiovascular risk. Smoking increases cardiovascular risk.
                Age and genetics are non-modifiable risk factors.
                """,
                "expected_patterns": [
                    ("high blood pressure", "increases", "cardiovascular risk"),
                    ("regular exercise", "reduces", "cardiovascular risk"),
                    ("healthy diet", "reduces", "cardiovascular risk"),
                    ("smoking", "increases", "cardiovascular risk"),
                ],
                "reasoning_type": "risk_factor_analysis",
            },
        ]

        reasoning_results = []
        cosmos_client = SimpleCosmosGremlinClient()

        for scenario in reasoning_scenarios:
            reasoning_start = time.time()

            # Extract knowledge and build reasoning graph
            extraction_result = await run_knowledge_extraction(
                text=scenario["content"],
                confidence_threshold=0.5,
                max_entities=30,
                max_relationships=25,
                enable_graph_storage=True,
                enable_entity_linking=True,
                enable_monitoring=False,
            )

            # Test graph-based search with reasoning
            if extraction_result.entities:
                primary_entity = extraction_result.entities[0].text

                # Search using graph reasoning
                reasoning_search = await run_universal_search(
                    query=primary_entity,
                    max_results=10,
                    enable_vector_search=False,
                    enable_graph_search=True,  # Focus on graph reasoning
                    enable_gnn_search=True,  # Enable GNN reasoning
                    graph_traversal_depth=3,  # Allow multi-hop reasoning
                    enable_monitoring=False,
                )
            else:
                reasoning_search = None

            reasoning_time = time.time() - reasoning_start

            # Analyze reasoning pattern detection
            extracted_relationships = [
                (r.subject.lower(), r.predicate.lower(), r.object.lower())
                for r in extraction_result.relationships
            ]

            pattern_matches = 0
            for expected_subject, expected_predicate, expected_object in scenario[
                "expected_patterns"
            ]:
                # Check for pattern matches (flexible matching)
                pattern_found = any(
                    expected_subject.lower() in subject
                    and expected_predicate.lower() in predicate
                    and expected_object.lower() in obj
                    for subject, predicate, obj in extracted_relationships
                )
                if pattern_found:
                    pattern_matches += 1

            pattern_detection_rate = pattern_matches / len(
                scenario["expected_patterns"]
            )

            # Validate reasoning capabilities
            assert (
                len(extraction_result.relationships) >= 3
            )  # Minimum reasoning relationships
            assert reasoning_time <= 15.0  # Reasoning under 15s
            assert pattern_detection_rate >= 0.4  # 40% pattern detection minimum

            reasoning_results.append(
                {
                    "reasoning_type": scenario["reasoning_type"],
                    "entities": len(extraction_result.entities),
                    "relationships": len(extraction_result.relationships),
                    "pattern_matches": pattern_matches,
                    "pattern_detection_rate": pattern_detection_rate,
                    "reasoning_time": reasoning_time,
                    "graph_search_successful": reasoning_search is not None
                    and len(reasoning_search.results) > 0,
                }
            )

        # Validate overall reasoning performance
        avg_pattern_detection = sum(
            r["pattern_detection_rate"] for r in reasoning_results
        ) / len(reasoning_results)
        avg_reasoning_time = sum(r["reasoning_time"] for r in reasoning_results) / len(
            reasoning_results
        )
        successful_graph_searches = sum(
            1 for r in reasoning_results if r["graph_search_successful"]
        )
        total_patterns_found = sum(r["pattern_matches"] for r in reasoning_results)

        assert avg_pattern_detection >= 0.5  # 50% average pattern detection
        assert avg_reasoning_time <= 10.0  # Average reasoning under 10s
        assert (
            successful_graph_searches >= len(reasoning_results) * 0.6
        )  # 60% successful graph searches
        assert total_patterns_found >= 5  # Minimum total patterns found

        print(f"✅ Graph-based reasoning patterns: {len(reasoning_results)} scenarios")
        print(f"   Average pattern detection rate: {avg_pattern_detection:.1%}")
        print(f"   Average reasoning time: {avg_reasoning_time:.2f}s")
        print(
            f"   Successful graph searches: {successful_graph_searches}/{len(reasoning_results)}"
        )
        print(f"   Total patterns found: {total_patterns_found}")

    @pytest.mark.asyncio
    async def test_cross_domain_knowledge_integration(self):
        """Test knowledge integration across different domains using graph intelligence"""

        # Cross-domain integration scenario
        integration_content = {
            "technical": """
                Machine learning models require large datasets for training. Data preprocessing
                involves cleaning and transforming raw data. Model validation ensures performance
                on unseen data. Deployment pipelines automate model release processes.
                """,
            "business": """
                Business intelligence systems analyze data to support decision making. Key performance
                indicators measure business success. Data governance ensures data quality and compliance.
                Stakeholder engagement is crucial for project success.
                """,
            "medical": """
                Medical diagnosis relies on patient symptoms and test results. Treatment protocols
                guide clinical decision making. Patient data must be protected according to regulations.
                Clinical trials validate treatment effectiveness.
                """,
        }

        # Extract knowledge from each domain
        domain_extractions = {}
        all_cross_domain_entities = []
        all_cross_domain_relationships = []

        integration_start = time.time()

        for domain, content in integration_content.items():
            extraction_result = await run_knowledge_extraction(
                text=content,
                confidence_threshold=0.6,
                max_entities=20,
                max_relationships=15,
                enable_graph_storage=True,
                enable_domain_tagging=True,  # Tag entities with domain
                enable_monitoring=False,
            )

            # Tag entities and relationships with domain
            for entity in extraction_result.entities:
                entity.domain = domain
                all_cross_domain_entities.append(entity)

            for relationship in extraction_result.relationships:
                relationship.domain = domain
                all_cross_domain_relationships.append(relationship)

            domain_extractions[domain] = extraction_result

        # Test cross-domain knowledge queries
        cross_domain_queries = [
            "data quality and validation processes",  # Should find tech + business + medical
            "system performance and monitoring",  # Should find tech + business
            "compliance and regulatory requirements",  # Should find business + medical
            "decision making and analysis methods",  # Should find all three
        ]

        integration_results = []

        for query in cross_domain_queries:
            query_start = time.time()

            # Search across integrated knowledge graph
            cross_domain_search = await run_universal_search(
                query=query,
                max_results=15,
                enable_vector_search=True,
                enable_graph_search=True,
                enable_cross_domain_search=True,  # Enable cross-domain search
                enable_monitoring=False,
                confidence_threshold=0.4,
            )

            query_time = time.time() - query_start

            # Analyze cross-domain integration in results
            result_domains = set()
            if hasattr(cross_domain_search, "results"):
                for result in cross_domain_search.results:
                    if hasattr(result, "domain"):
                        result_domains.add(result.domain)
                    # Infer domain from content if not tagged
                    elif hasattr(result, "content"):
                        content_lower = result.content.lower()
                        if any(
                            term in content_lower
                            for term in [
                                "machine learning",
                                "data preprocessing",
                                "model",
                            ]
                        ):
                            result_domains.add("technical")
                        if any(
                            term in content_lower
                            for term in ["business", "stakeholder", "kpi"]
                        ):
                            result_domains.add("business")
                        if any(
                            term in content_lower
                            for term in ["medical", "patient", "clinical"]
                        ):
                            result_domains.add("medical")

            domain_coverage = len(result_domains)
            cross_domain_integration_score = domain_coverage / 3.0  # 3 total domains

            # Validate cross-domain integration
            assert len(cross_domain_search.results) >= 3  # Minimum results
            assert query_time <= 5.0  # Performance requirement
            assert (
                cross_domain_integration_score >= 0.33
            )  # At least 1 domain represented

            integration_results.append(
                {
                    "query": query,
                    "results_count": len(cross_domain_search.results),
                    "domains_found": len(result_domains),
                    "domain_coverage": cross_domain_integration_score,
                    "query_time": query_time,
                    "confidence": cross_domain_search.overall_confidence,
                }
            )

        total_integration_time = time.time() - integration_start

        # Validate overall cross-domain integration
        avg_domain_coverage = sum(
            r["domain_coverage"] for r in integration_results
        ) / len(integration_results)
        avg_query_time = sum(r["query_time"] for r in integration_results) / len(
            integration_results
        )
        avg_confidence = sum(r["confidence"] for r in integration_results) / len(
            integration_results
        )
        multi_domain_queries = sum(
            1 for r in integration_results if r["domains_found"] >= 2
        )

        assert (
            len(all_cross_domain_entities) >= 30
        )  # Sufficient entities across domains
        assert (
            len(all_cross_domain_relationships) >= 20
        )  # Sufficient relationships across domains
        assert (
            avg_domain_coverage >= GRAPH_INTELLIGENCE_SLA["knowledge_integration_score"]
        )
        assert avg_query_time <= 3.0  # Average query performance
        assert (
            multi_domain_queries >= len(integration_results) * 0.5
        )  # 50% multi-domain queries

        print(
            f"✅ Cross-domain knowledge integration: {len(integration_results)} queries"
        )
        print(f"   Total integration time: {total_integration_time:.2f}s")
        print(f"   Average domain coverage: {avg_domain_coverage:.1%}")
        print(f"   Average query time: {avg_query_time:.3f}s")
        print(
            f"   Multi-domain queries: {multi_domain_queries}/{len(integration_results)}"
        )
        print(f"   Cross-domain entities: {len(all_cross_domain_entities)}")
        print(f"   Cross-domain relationships: {len(all_cross_domain_relationships)}")

    @pytest.mark.asyncio
    async def test_graph_traversal_performance_optimization(self):
        """Test graph traversal performance and optimization under different query patterns"""

        # Create a substantial knowledge graph for performance testing
        performance_content = """
        Enterprise software systems require robust architecture and scalable design patterns.
        Microservices architecture decomposes applications into small, independent services.
        API gateways manage communication between microservices and external clients.
        Load balancers distribute traffic across multiple service instances.
        Database sharding partitions data across multiple database instances.
        Caching layers improve response times and reduce database load.
        Monitoring systems track application performance and health metrics.
        Security frameworks protect against common vulnerabilities and attacks.
        Continuous integration pipelines automate testing and deployment processes.
        Container orchestration platforms manage containerized application lifecycle.
        Message queues enable asynchronous communication between services.
        Service discovery mechanisms allow services to locate and communicate with each other.
        Configuration management systems maintain consistent settings across environments.
        Logging aggregation collects and analyzes logs from distributed systems.
        Performance profiling identifies bottlenecks and optimization opportunities.
        """

        # Build comprehensive knowledge graph
        graph_construction_start = time.time()

        extraction_result = await run_knowledge_extraction(
            text=performance_content,
            confidence_threshold=0.6,
            max_entities=50,
            max_relationships=40,
            enable_graph_storage=True,
            enable_entity_linking=True,
            enable_monitoring=False,
        )

        graph_construction_time = time.time() - graph_construction_start

        # Test different graph traversal patterns
        traversal_patterns = [
            {
                "name": "single_hop_traversal",
                "query": "microservices architecture",
                "traversal_depth": 1,
                "expected_performance": 1.0,  # Under 1s
            },
            {
                "name": "multi_hop_traversal",
                "query": "enterprise software systems",
                "traversal_depth": 3,
                "expected_performance": 2.0,  # Under 2s
            },
            {
                "name": "deep_traversal",
                "query": "scalable design patterns",
                "traversal_depth": 5,
                "expected_performance": 3.0,  # Under 3s
            },
            {
                "name": "broad_traversal",
                "query": "system performance",
                "traversal_depth": 2,
                "max_results": 20,
                "expected_performance": 2.5,  # Under 2.5s
            },
        ]

        traversal_results = []
        cosmos_client = SimpleCosmosGremlinClient()

        for pattern in traversal_patterns:
            traversal_start = time.time()

            # Execute graph traversal with specific pattern
            traversal_result = await run_universal_search(
                query=pattern["query"],
                max_results=pattern.get("max_results", 10),
                enable_vector_search=False,  # Focus on graph traversal
                enable_graph_search=True,
                graph_traversal_depth=pattern["traversal_depth"],
                enable_traversal_optimization=True,  # Enable optimization
                enable_monitoring=False,
            )

            traversal_time = time.time() - traversal_start

            # Validate traversal performance
            assert traversal_result is not None
            assert len(traversal_result.results) >= 3  # Minimum results from traversal
            assert (
                traversal_time <= pattern["expected_performance"]
            )  # Performance requirement
            assert traversal_result.overall_confidence >= 0.4  # Reasonable confidence

            # Analyze traversal depth achieved
            traversal_depth_achieved = getattr(
                traversal_result, "actual_traversal_depth", 1
            )

            traversal_results.append(
                {
                    "pattern_name": pattern["name"],
                    "query": pattern["query"],
                    "requested_depth": pattern["traversal_depth"],
                    "achieved_depth": traversal_depth_achieved,
                    "results_count": len(traversal_result.results),
                    "traversal_time": traversal_time,
                    "performance_target": pattern["expected_performance"],
                    "performance_ratio": traversal_time
                    / pattern["expected_performance"],
                    "confidence": traversal_result.overall_confidence,
                }
            )

        # Test concurrent traversal performance
        concurrent_start = time.time()

        concurrent_tasks = [
            run_universal_search(
                query="system architecture patterns",
                max_results=8,
                enable_graph_search=True,
                graph_traversal_depth=2,
                enable_monitoring=False,
            )
            for _ in range(5)  # 5 concurrent traversals
        ]

        concurrent_results = await asyncio.gather(*concurrent_tasks)
        concurrent_time = time.time() - concurrent_start

        # Validate concurrent performance
        successful_concurrent = [
            r for r in concurrent_results if r is not None and len(r.results) > 0
        ]
        concurrent_success_rate = (
            len(successful_concurrent) / len(concurrent_results) * 100
        )

        assert concurrent_success_rate >= 80.0  # 80% success rate under load
        assert concurrent_time <= 8.0  # All concurrent traversals under 8s

        # Validate overall traversal optimization
        avg_traversal_time = sum(r["traversal_time"] for r in traversal_results) / len(
            traversal_results
        )
        avg_performance_ratio = sum(
            r["performance_ratio"] for r in traversal_results
        ) / len(traversal_results)
        patterns_meeting_target = sum(
            1 for r in traversal_results if r["performance_ratio"] <= 1.0
        )

        assert avg_traversal_time <= GRAPH_INTELLIGENCE_SLA["graph_traversal_time"]
        assert avg_performance_ratio <= 1.2  # Average performance within 120% of target
        assert (
            patterns_meeting_target >= len(traversal_results) * 0.75
        )  # 75% meet performance targets

        print(
            f"✅ Graph traversal performance optimization: {len(traversal_patterns)} patterns tested"
        )
        print(f"   Graph construction time: {graph_construction_time:.2f}s")
        print(f"   Average traversal time: {avg_traversal_time:.3f}s")
        print(f"   Average performance ratio: {avg_performance_ratio:.2f}x")
        print(
            f"   Patterns meeting target: {patterns_meeting_target}/{len(traversal_results)}"
        )
        print(f"   Concurrent success rate: {concurrent_success_rate:.1f}%")
        print(f"   Total entities in graph: {len(extraction_result.entities)}")
        print(
            f"   Total relationships in graph: {len(extraction_result.relationships)}"
        )
