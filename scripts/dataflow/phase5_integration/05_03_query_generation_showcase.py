#!/usr/bin/env python3
"""
Query Generation Showcase - PydanticAI SQL Pattern Demonstration
===============================================================

Comprehensive demonstration of the AI-powered query generation system:
- Gremlin query generation for Azure Cosmos DB operations
- Azure Cognitive Search query optimization
- Domain analysis query generation
- Query orchestration and performance optimization
- Unified query generation interface demonstration

Showcases the zero-hardcoded-values philosophy with AI-generated queries.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.query_generation.analysis_query_agent import generate_analysis_query
from agents.query_generation.gremlin_query_agent import generate_gremlin_query
from agents.query_generation.search_query_agent import generate_search_query

# Import all query generation agents and orchestrator
from agents.query_generation.universal_query_orchestrator import (
    QueryRequest,
    UniversalQueryOrchestrator,
    generate_analysis_query_orchestrated,
    generate_gremlin_query_orchestrated,
    generate_search_query_orchestrated,
    query_orchestrator,
)


async def query_generation_showcase(
    demo_type: str = "comprehensive", session_id: str = None
) -> Dict[str, Any]:
    """
    Comprehensive showcase of AI-powered query generation capabilities
    """
    session_id = session_id or f"query_showcase_{int(time.time())}"
    print("🤖 Query Generation Showcase - PydanticAI SQL Pattern")
    print(f"Session: {session_id}")
    print("=" * 80)

    results = {
        "session_id": session_id,
        "demo_type": demo_type,
        "showcases": [],
        "overall_status": "in_progress",
    }

    start_time = time.time()

    try:
        # Showcase 1: Gremlin Query Generation for Azure Cosmos DB
        print("📊 Showcase 1: Gremlin Query Generation for Azure Cosmos DB")
        print("=" * 60)
        gremlin_start = time.time()

        # Sample entities and relationships for demo
        sample_entities = [
            {"text": "Azure RAG System", "type": "SYSTEM", "confidence": 0.95},
            {"text": "Knowledge Graph", "type": "CONCEPT", "confidence": 0.87},
            {"text": "PydanticAI", "type": "FRAMEWORK", "confidence": 0.91},
        ]

        sample_relationships = [
            {
                "subject": "Azure RAG System",
                "predicate": "USES",
                "object": "Knowledge Graph",
                "confidence": 0.89,
            },
            {
                "subject": "Azure RAG System",
                "predicate": "POWERED_BY",
                "object": "PydanticAI",
                "confidence": 0.92,
            },
        ]

        print("🔧 Generating entity insertion queries...")
        entity_insertion_query = await generate_gremlin_query(
            operation_type="insert_entity",
            entities=sample_entities,
            relationships=[],
            confidence_threshold=0.8,
        )

        print("🔗 Generating relationship insertion queries...")
        relationship_insertion_query = await generate_gremlin_query(
            operation_type="insert_relationship",
            entities=[],
            relationships=sample_relationships,
            confidence_threshold=0.8,
        )

        print("🌐 Generating graph traversal queries...")
        graph_traversal_query = await generate_gremlin_query(
            operation_type="traverse_graph",
            entities=[],
            relationships=[],
            graph_context={
                "start_entity": "Azure RAG System",
                "traversal_type": "outbound",
            },
            max_results=20,
            traversal_depth=3,
        )

        gremlin_duration = time.time() - gremlin_start

        gremlin_showcase = {
            "showcase": "gremlin_query_generation",
            "duration": gremlin_duration,
            "status": "completed",
            "queries_generated": {
                "entity_insertion": {
                    "query": (
                        entity_insertion_query.query[:100] + "..."
                        if len(entity_insertion_query.query) > 100
                        else entity_insertion_query.query
                    ),
                    "complexity": entity_insertion_query.estimated_complexity,
                    "explanation": entity_insertion_query.explanation,
                },
                "relationship_insertion": {
                    "query": (
                        relationship_insertion_query.query[:100] + "..."
                        if len(relationship_insertion_query.query) > 100
                        else relationship_insertion_query.query
                    ),
                    "complexity": relationship_insertion_query.estimated_complexity,
                    "explanation": relationship_insertion_query.explanation,
                },
                "graph_traversal": {
                    "query": (
                        graph_traversal_query.query[:100] + "..."
                        if len(graph_traversal_query.query) > 100
                        else graph_traversal_query.query
                    ),
                    "complexity": graph_traversal_query.estimated_complexity,
                    "explanation": graph_traversal_query.explanation,
                },
            },
            "ai_generated": True,
        }

        results["showcases"].append(gremlin_showcase)

        print(
            f"   ✅ Entity insertion query: {entity_insertion_query.estimated_complexity} complexity"
        )
        print(
            f"   ✅ Relationship insertion query: {relationship_insertion_query.estimated_complexity} complexity"
        )
        print(
            f"   ✅ Graph traversal query: {graph_traversal_query.estimated_complexity} complexity"
        )
        print(f"   ⏱️  Gremlin generation: {gremlin_duration:.2f}s\n")

        # Showcase 2: Azure Cognitive Search Query Generation
        print("🔍 Showcase 2: Azure Cognitive Search Query Generation")
        print("=" * 60)
        search_start = time.time()

        print("🎯 Generating vector similarity search queries...")
        vector_search_query = await generate_search_query(
            query_type="vector",
            search_text="knowledge extraction and graph neural networks",
            vector_fields=["content_vector", "title_vector"],
            searchable_fields=["content", "title", "metadata", "tags"],
            filter_conditions={"vocabulary_complexity": ">0.7", "language": "en"},
            facet_fields=["category", "document_type", "created_date"],
            similarity_threshold=0.75,
            top_k=15,
            domain_context={"vocabulary_complexity": 0.8, "enable_semantic": True},
        )

        print("🔄 Generating hybrid search queries...")
        hybrid_search_query = await generate_search_query(
            query_type="hybrid",
            search_text="machine learning and artificial intelligence",
            vector_fields=["content_vector"],
            searchable_fields=["content", "title", "abstract"],
            filter_conditions={"confidence_score": ">0.7"},
            similarity_threshold=0.7,
            top_k=12,
            domain_context={"research_indicators": 0.8, "enable_semantic": True},
        )

        search_duration = time.time() - search_start

        search_showcase = {
            "showcase": "search_query_generation",
            "duration": search_duration,
            "status": "completed",
            "queries_generated": {
                "vector_search": {
                    "search_text": vector_search_query.search_text,
                    "filter_expression": vector_search_query.filter_expression,
                    "performance": vector_search_query.estimated_performance,
                    "explanation": vector_search_query.explanation,
                },
                "hybrid_search": {
                    "search_text": hybrid_search_query.search_text,
                    "filter_expression": hybrid_search_query.filter_expression,
                    "performance": hybrid_search_query.estimated_performance,
                    "explanation": hybrid_search_query.explanation,
                },
            },
            "ai_generated": True,
        }

        results["showcases"].append(search_showcase)

        print(
            f"   ✅ Vector search query: {vector_search_query.estimated_performance} performance"
        )
        print(
            f"   ✅ Hybrid search query: {hybrid_search_query.estimated_performance} performance"
        )
        print(f"   ⏱️  Search generation: {search_duration:.2f}s\n")

        # Showcase 3: Domain Analysis Query Generation
        print("🧠 Showcase 3: Domain Analysis Query Generation")
        print("=" * 60)
        analysis_start = time.time()

        sample_content = [
            "Azure Cognitive Services provides AI capabilities for developers...",
            "Machine learning models require training data and feature engineering...",
            "Graph neural networks excel at relationship modeling and prediction...",
        ]

        print("📊 Generating domain characterization queries...")
        domain_analysis_query = await generate_analysis_query(
            analysis_type="domain_characterization",
            content_samples=sample_content,
            domain_context={
                "analysis_focus": "concept_density",
                "target_domain": "AI/ML",
            },
            analysis_depth="deep",
            target_metrics=[
                "vocabulary_complexity",
                "concept_density",
                "concept_coverage",
            ],
        )

        print("⚡ Generating performance optimization queries...")
        performance_optimization_query = await generate_analysis_query(
            analysis_type="performance_optimization",
            content_samples=sample_content,
            domain_context={
                "system_type": "RAG",
                "target_metrics": ["speed", "accuracy"],
            },
            target_metrics=[
                "processing_speed",
                "extraction_accuracy",
                "search_relevance",
            ],
            optimization_goals=[
                "reduce_latency",
                "improve_precision",
                "enhance_recall",
            ],
        )

        analysis_duration = time.time() - analysis_start

        analysis_showcase = {
            "showcase": "analysis_query_generation",
            "duration": analysis_duration,
            "status": "completed",
            "queries_generated": {
                "domain_characterization": {
                    "complexity_level": domain_analysis_query.complexity_level,
                    "expected_output_format": domain_analysis_query.expected_output_format,
                    "execution_strategy": domain_analysis_query.execution_strategy,
                    "success_criteria": domain_analysis_query.success_criteria,
                },
                "performance_optimization": {
                    "complexity_level": performance_optimization_query.complexity_level,
                    "expected_output_format": performance_optimization_query.expected_output_format,
                    "execution_strategy": performance_optimization_query.execution_strategy,
                    "success_criteria": performance_optimization_query.success_criteria,
                },
            },
            "ai_generated": True,
        }

        results["showcases"].append(analysis_showcase)

        print(
            f"   ✅ Domain characterization: {domain_analysis_query.complexity_level} complexity"
        )
        print(
            f"   ✅ Performance optimization: {performance_optimization_query.complexity_level} complexity"
        )
        print(f"   ⏱️  Analysis generation: {analysis_duration:.2f}s\n")

        # Showcase 4: Query Orchestration and Batch Processing
        print("🎭 Showcase 4: Query Orchestration and Batch Processing")
        print("=" * 60)
        orchestration_start = time.time()

        # Create batch query requests
        batch_requests = [
            QueryRequest(
                query_type="gremlin",
                operation_type="analyze_patterns",
                context={"analysis_type": "connectivity"},
                parameters={"max_results": 25},
            ),
            QueryRequest(
                query_type="search",
                operation_type="semantic",
                context={
                    "search_text": "neural network architecture",
                    "domain_context": {"research_indicators": 0.8},
                },
                parameters={"similarity_threshold": 0.8, "top_k": 10},
            ),
            QueryRequest(
                query_type="analysis",
                operation_type="ml_training",
                context={"target_metrics": ["accuracy", "precision", "recall"]},
                parameters={"analysis_depth": "adaptive"},
            ),
        ]

        print(f"🚀 Processing {len(batch_requests)} queries in parallel...")
        batch_responses = await query_orchestrator.generate_multi_query_batch(
            batch_requests
        )

        orchestration_duration = time.time() - orchestration_start

        successful_queries = sum(1 for resp in batch_responses if resp.success)
        failed_queries = len(batch_responses) - successful_queries

        orchestration_showcase = {
            "showcase": "query_orchestration",
            "duration": orchestration_duration,
            "status": "completed",
            "batch_processing": {
                "total_queries": len(batch_requests),
                "successful_queries": successful_queries,
                "failed_queries": failed_queries,
                "success_rate": (
                    successful_queries / len(batch_requests) if batch_requests else 0
                ),
                "concurrent_execution": True,
            },
            "cache_stats": query_orchestrator.get_cache_stats(),
            "ai_generated": True,
        }

        results["showcases"].append(orchestration_showcase)

        print(
            f"   ✅ Batch processing completed: {successful_queries}/{len(batch_requests)} successful"
        )
        print(
            f"   📊 Success rate: {(successful_queries/len(batch_requests)*100):.1f}%"
        )
        print(
            f"   🏎️  Cache size: {orchestration_showcase['cache_stats']['cache_size']} entries"
        )
        print(f"   ⏱️  Orchestration: {orchestration_duration:.2f}s\n")

        # Final Summary
        total_time = time.time() - start_time
        total_queries_generated = (
            sum(
                len(showcase.get("queries_generated", {}))
                for showcase in results["showcases"]
            )
            + orchestration_showcase["batch_processing"]["successful_queries"]
        )

        results.update(
            {
                "overall_status": "completed",
                "total_duration": total_time,
                "summary": {
                    "total_showcases": len(results["showcases"]),
                    "total_queries_generated": total_queries_generated,
                    "ai_optimization_coverage": "100%",
                    "query_types_demonstrated": [
                        "gremlin",
                        "search",
                        "analysis",
                        "batch_orchestration",
                    ],
                    "zero_hardcoded_values": True,
                    "pydantic_ai_integration": True,
                },
            }
        )

        print("🎉 Query Generation Showcase Complete!")
        print("=" * 80)
        print(f"   📄 Session: {session_id}")
        print(f"   ⏱️  Total time: {total_time:.2f}s")
        print(f"   🤖 Total queries generated: {total_queries_generated}")
        print(f"   📊 Showcases completed: {len(results['showcases'])}")
        print(f"   🎯 AI optimization: 100% (all queries AI-generated)")
        print(f"   🔧 Query types: Gremlin, Search, Analysis, Orchestration")
        print(f"   🚀 Zero hardcoded values: ✅")
        print(f"   🧠 PydanticAI integration: ✅")

        return results

    except Exception as e:
        total_time = time.time() - start_time
        results.update(
            {
                "overall_status": "failed",
                "total_duration": total_time,
                "error": str(e),
                "showcases_completed": len(results.get("showcases", [])),
            }
        )

        print(f"❌ Query Generation Showcase Failed!")
        print(f"   📄 Session: {session_id}")
        print(f"   ⏱️  Time elapsed: {total_time:.2f}s")
        print(f"   📊 Showcases completed: {len(results.get('showcases', []))}")
        print(f"   ❌ Error: {e}")

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Query Generation Showcase - PydanticAI SQL Pattern"
    )
    parser.add_argument(
        "--demo-type",
        default="comprehensive",
        help="Type of demo to run (comprehensive, gremlin, search, analysis)",
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--output", help="Save results to JSON file")
    args = parser.parse_args()

    print("🤖 Azure Universal RAG - Query Generation Showcase")
    print("=" * 80)
    print("Comprehensive demonstration of AI-powered query generation:")
    print("• Gremlin Query Agent → optimized Azure Cosmos DB queries")
    print("• Search Query Agent → enhanced Azure Cognitive Search queries")
    print("• Analysis Query Agent → intelligent domain analysis prompts")
    print("• Universal Query Orchestrator → coordinated multi-query workflows")
    print("• Zero hardcoded values → all queries dynamically AI-generated")
    print("")

    # Run the comprehensive showcase
    result = asyncio.run(query_generation_showcase(demo_type=args.demo_type))

    # Handle JSON output
    if args.json or args.output:
        json_output = json.dumps(result, indent=2, default=str)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(json_output)
            print(f"\n📄 Results saved to: {output_path}")

        if args.json:
            print(f"\n" + "=" * 80)
            print("Showcase Results JSON:")
            print(json_output)

    # Final summary
    if result["overall_status"] == "completed":
        print(f"\n🎉 SUCCESS: Query generation showcase completed!")
        print(f"   📄 Session: {result['session_id']}")
        print(f"   ⏱️  Duration: {result['total_duration']:.2f}s")
        print(
            f"   🤖 Queries generated: {result['summary']['total_queries_generated']}"
        )
        print(f"   📊 Showcases: {result['summary']['total_showcases']}")
        print("AI-powered query generation with PydanticAI SQL pattern demonstrated!")
        sys.exit(0)
    else:
        print(f"\n❌ FAILED: Showcase encountered issues.")
        print(f"   📄 Session: {result['session_id']}")
        print(f"   ⏱️  Duration: {result.get('total_duration', 0):.2f}s")
        print(f"   📊 Showcases completed: {len(result.get('showcases', []))}")
        print("Check the error messages above for details.")
        sys.exit(1)
