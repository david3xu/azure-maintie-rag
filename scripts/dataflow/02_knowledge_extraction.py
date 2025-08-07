#!/usr/bin/env python3
"""
Knowledge Extraction Pipeline - PydanticAI Agent Workflow
Uses real Knowledge Extraction Agent with Azure services integration.
Agent-centric workflow following zero-domain-bias architecture.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.core.universal_deps import get_universal_deps

# Import new PydanticAI agents
from agents.domain_intelligence.agent import run_domain_analysis
from agents.knowledge_extraction.agent import run_knowledge_extraction
from agents.orchestrator import UniversalOrchestrator
from infrastructure.utilities.azure_cost_tracker import AzureServiceCostTracker


async def knowledge_extraction_pipeline(
    content: str = None,
    data_directory: str = "/workspace/azure-maintie-rag/data/raw",
    session_id: str = None,
) -> Dict[str, Any]:
    """
    PydanticAI Knowledge Extraction Pipeline
    Uses proper agent delegation and Universal RAG architecture
    """
    session_id = session_id or f"extraction_{int(time.time())}"
    print("🧠 Knowledge Extraction Pipeline - PydanticAI Architecture")
    print(f"Session: {session_id}")
    print("=" * 60)

    results = {
        "session_id": session_id,
        "data_directory": data_directory,
        "stages": [],
        "cost_tracking": {"enabled": True, "total_cost": 0.0, "breakdown": {}},
        "overall_status": "in_progress",
    }

    # Initialize cost tracker
    cost_tracker = AzureServiceCostTracker()

    start_time = time.time()

    try:
        # Initialize dependencies
        deps = await get_universal_deps()
        print(f"🔧 Initialized Universal Dependencies")
        print(f"   Available services: {', '.join(deps.get_available_services())}")

        # Use sample content for the pipeline
        sample_content = (
            content
            or """
        Azure Cosmos DB provides global distribution with multi-master replication. 
        The service offers five consistency models: Strong, Bounded Staleness, Session,
        Consistent Prefix, and Eventual consistency. Performance is optimized through 
        automatic partitioning and request unit provisioning.
        """
        )

        # Stage 1: Domain Analysis using new PydanticAI agent
        print("\n🌍 Stage 1: Domain Intelligence Agent")
        stage_start = time.time()

        try:
            domain_analysis = await run_domain_analysis(sample_content, detailed=True)
            stage_duration = time.time() - stage_start

            results["stages"].append(
                {
                    "stage": "domain_intelligence",
                    "agent": "Domain Intelligence Agent",
                    "duration": stage_duration,
                    "status": "completed",
                    "content_signature": domain_analysis.content_signature,
                    "vocabulary_complexity": domain_analysis.vocabulary_complexity,
                    "concept_density": domain_analysis.concept_density,
                }
            )

            print(f"   ✅ Content signature: {domain_analysis.content_signature}")
            print(
                f"   📊 Vocabulary complexity: {domain_analysis.vocabulary_complexity:.3f}"
            )
            print(f"   🎯 Concept density: {domain_analysis.concept_density:.3f}")
            print(f"   ⏱️  Duration: {stage_duration:.2f}s")

        except Exception as e:
            stage_duration = time.time() - stage_start
            results["stages"].append(
                {
                    "stage": "domain_intelligence",
                    "agent": "Domain Intelligence Agent",
                    "duration": stage_duration,
                    "status": "failed",
                    "error": str(e),
                }
            )
            print(f"   ❌ Domain analysis failed: {e}")
            domain_analysis = None

        # Stage 2: Knowledge Extraction using new PydanticAI agent
        print("\n📚 Stage 2: Knowledge Extraction Agent")
        extraction_start = time.time()

        try:
            extraction_result = await run_knowledge_extraction(
                sample_content, use_domain_analysis=domain_analysis is not None
            )

            extraction_duration = time.time() - extraction_start

            results["stages"].append(
                {
                    "stage": "knowledge_extraction",
                    "agent": "Knowledge Extraction Agent",
                    "duration": extraction_duration,
                    "status": "completed",
                    "entities_found": len(extraction_result.entities),
                    "relationships_found": len(extraction_result.relationships),
                    "extraction_confidence": extraction_result.extraction_confidence,
                    "processing_signature": extraction_result.processing_signature,
                }
            )

            print(f"   ✅ Entities extracted: {len(extraction_result.entities)}")
            print(f"   🔗 Relationships found: {len(extraction_result.relationships)}")
            print(
                f"   🎯 Extraction confidence: {extraction_result.extraction_confidence:.3f}"
            )
            print(
                f"   📊 Processing signature: {extraction_result.processing_signature}"
            )
            print(f"   ⏱️  Duration: {extraction_duration:.2f}s")

            # Show top entities
            if extraction_result.entities:
                print("   🏷️  Top entities:")
                for entity in extraction_result.entities[:3]:
                    print(
                        f"      - {entity.text} ({entity.type}, conf: {entity.confidence:.2f})"
                    )

            # Show relationships
            if extraction_result.relationships:
                print("   🔗 Relationships:")
                for rel in extraction_result.relationships[:3]:
                    print(
                        f"      - {rel.source_entity} -[{rel.relationship_type}]-> {rel.target_entity}"
                    )

            results["extraction_result"] = {
                "entities": [
                    {
                        "text": e.text,
                        "type": e.type,
                        "confidence": e.confidence,
                        "context": e.context[:100],
                    }
                    for e in extraction_result.entities
                ],
                "relationships": [
                    {
                        "source": r.source_entity,
                        "target": r.target_entity,
                        "type": r.relationship_type,
                        "confidence": r.confidence,
                    }
                    for r in extraction_result.relationships
                ],
            }

        except Exception as e:
            extraction_duration = time.time() - extraction_start
            results["stages"].append(
                {
                    "stage": "knowledge_extraction",
                    "agent": "Knowledge Extraction Agent",
                    "duration": extraction_duration,
                    "status": "failed",
                    "error": str(e),
                }
            )
            print(f"   ❌ Knowledge extraction failed: {e}")

        # Stage 3: Multi-Agent Orchestration Demo
        print("\n🎭 Stage 3: Multi-Agent Orchestration")
        orchestration_start = time.time()

        try:
            orchestrator = UniversalOrchestrator()
            orchestration_result = (
                await orchestrator.process_knowledge_extraction_workflow(
                    sample_content, use_domain_analysis=True
                )
            )

            orchestration_duration = time.time() - orchestration_start

            results["stages"].append(
                {
                    "stage": "orchestration",
                    "component": "UniversalOrchestrator",
                    "duration": orchestration_duration,
                    "status": (
                        "completed" if orchestration_result.success else "partial"
                    ),
                    "agents_coordinated": list(
                        orchestration_result.agent_metrics.keys()
                    ),
                    "total_processing_time": orchestration_result.total_processing_time,
                }
            )

            print(f"   ✅ Multi-agent coordination: {orchestration_result.success}")
            print(
                f"   🤖 Agents coordinated: {', '.join(orchestration_result.agent_metrics.keys())}"
            )
            print(
                f"   ⏱️  Total time: {orchestration_result.total_processing_time:.2f}s"
            )

            if orchestration_result.extraction_summary:
                print(f"   📊 Orchestrated extraction:")
                print(
                    f"      Entities: {orchestration_result.extraction_summary['entities_count']}"
                )
                print(
                    f"      Relationships: {orchestration_result.extraction_summary['relationships_count']}"
                )
                print(
                    f"      Confidence: {orchestration_result.extraction_summary['confidence']:.3f}"
                )

            results["orchestration_result"] = {
                "success": orchestration_result.success,
                "agent_metrics": orchestration_result.agent_metrics,
                "extraction_summary": orchestration_result.extraction_summary,
            }

        except Exception as e:
            orchestration_duration = time.time() - orchestration_start
            results["stages"].append(
                {
                    "stage": "orchestration",
                    "component": "UniversalOrchestrator",
                    "duration": orchestration_duration,
                    "status": "failed",
                    "error": str(e),
                }
            )
            print(f"   ❌ Orchestration failed: {e}")

        # Final Results
        total_duration = time.time() - start_time
        results["overall_status"] = "completed"
        results["total_duration"] = total_duration
        results["successful_stages"] = len(
            [s for s in results["stages"] if s["status"] == "completed"]
        )
        results["total_stages"] = len(results["stages"])

        print(f"\n✅ Pipeline Summary")
        print(f"=" * 40)
        print(f"🎯 Session: {session_id}")
        print(f"⏱️  Total Duration: {total_duration:.2f}s")
        print(
            f"📊 Stages: {results['successful_stages']}/{results['total_stages']} successful"
        )
        print(f"🏗️  Architecture: PydanticAI Multi-Agent")

        return results

    except Exception as e:
        print(f"\n❌ Pipeline Error: {e}")
        results["overall_status"] = "failed"
        results["error"] = str(e)
        results["total_duration"] = time.time() - start_time
        return results


async def main():
    """Run the knowledge extraction pipeline demonstration"""
    print("🚀 Azure Universal RAG - Knowledge Extraction Pipeline")
    print("====================================================")

    # Run the pipeline
    result = await knowledge_extraction_pipeline()

    # Display final summary
    if result["overall_status"] == "completed":
        print(f"\n🎉 Knowledge Extraction Pipeline Completed!")
        print(f"✅ Architecture: PydanticAI Multi-Agent System")
        print(f"⏱️  Total Time: {result['total_duration']:.2f}s")
        print(
            f"📊 Success Rate: {result['successful_stages']}/{result['total_stages']} stages"
        )
    else:
        print(f"\n⚠️  Pipeline completed with issues")
        if "error" in result:
            print(f"❌ Error: {result['error']}")

    return result


if __name__ == "__main__":
    asyncio.run(main())
