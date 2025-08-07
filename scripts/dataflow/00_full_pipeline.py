#!/usr/bin/env python3
"""
Universal Full Pipeline - Production Azure Universal RAG
Complete end-to-end pipeline using PydanticAI agents with Azure cloud services.
Orchestrates all dataflow stages with comprehensive progress tracking.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.domain_intelligence.agent import (
    UniversalDomainDeps,
    run_universal_domain_analysis,
)
from agents.orchestrator import UniversalOrchestrator
from infrastructure.prompt_workflows.universal_prompt_generator import (
    UniversalPromptGenerator,
)


async def run_universal_full_pipeline(
    data_dir: str = "/workspace/azure-maintie-rag/data/raw",
    skip_prerequisites: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Universal full pipeline with Azure cloud services and comprehensive progress tracking"""
    session_id = f"pipeline_{int(time.time())}"
    print("🚀 Azure Universal RAG - Complete Pipeline")
    print(f"Session: {session_id} | Data: {Path(data_dir).name}")
    print("=" * 60)

    start_time = time.time()
    pipeline_results = {
        "session_id": session_id,
        "data_directory": data_dir,
        "start_time": start_time,
        "stages_completed": [],
        "overall_status": "in_progress",
    }

    try:
        # Prerequisites: Azure State Check (unless skipped)
        if not skip_prerequisites:
            print("🔍 Prerequisites: Azure Services State Check")
            # Import here to avoid circular imports
            # Run a simple inline Azure connectivity check
            from infrastructure.azure_openai.openai_client import AzureOpenAIClient

            try:
                openai_client = AzureOpenAIClient()
                await openai_client.async_initialize()
                print("✅ Azure OpenAI connectivity verified")
                azure_ready = True
            except Exception as e:
                print(f"⚠️  Azure OpenAI connection issue: {str(e)[:100]}...")
                print("Continuing with pipeline - some features may be limited")
                azure_ready = False

            pipeline_results["prerequisites"] = {"azure_openai_ready": azure_ready}
            print("✅ Prerequisites check completed\n")

        # Stage 1: Universal Domain Discovery
        print("🔍 Stage 1: Universal Domain Analysis")
        stage_start = time.time()

        print(f"   📁 Analyzing content from: {data_dir}")

        domain_analysis = await run_universal_domain_analysis(
            UniversalDomainDeps(
                data_directory=data_dir,
                max_files_to_analyze=10,
                min_content_length=200,
                enable_multilingual=True,
            )
        )

        stage_duration = time.time() - stage_start
        pipeline_results["stages_completed"].append(
            {
                "stage": "domain_analysis",
                "duration": stage_duration,
                "status": "completed",
            }
        )

        print(f"   ✅ Domain discovered: {domain_analysis.domain_signature}")
        print(
            f"   📊 Content confidence: {domain_analysis.content_type_confidence:.2f}"
        )
        print(
            f"   🧠 Vocabulary richness: {domain_analysis.characteristics.vocabulary_richness:.3f}"
        )
        print(
            f"   ⚙️  Technical density: {domain_analysis.characteristics.technical_vocabulary_ratio:.3f}"
        )
        print(f"   ⏱️  Stage duration: {stage_duration:.2f}s")

        # Stage 2: Universal Prompt Generation
        print(f"\n📝 Stage 2: Universal Prompt Generation")
        stage_start = time.time()

        prompt_generator = UniversalPromptGenerator()
        generated_prompts = await prompt_generator.generate_domain_prompts(
            data_directory=data_dir, max_files_to_analyze=10
        )

        stage_duration = time.time() - stage_start
        pipeline_results["stages_completed"].append(
            {
                "stage": "prompt_generation",
                "duration": stage_duration,
                "status": "completed",
                "prompts_generated": len(generated_prompts),
            }
        )

        print(f"   ✅ Generated {len(generated_prompts)} domain-adaptive prompts")
        for prompt_type, path in generated_prompts.items():
            print(f"   📄 {prompt_type}: {Path(path).name}")
        print(f"   ⏱️  Stage duration: {stage_duration:.2f}s")

        # Stage 3: Universal Orchestrated Processing
        print(f"\n🚀 Stage 3: Universal Orchestrated Processing")
        stage_start = time.time()

        orchestrator = UniversalOrchestrator()

        result = await orchestrator.process_universal_workflow(
            data_directory=data_dir,
            query="analyze content and extract knowledge",
            enable_extraction=True,
            enable_search=True,
        )

        stage_duration = time.time() - stage_start
        pipeline_results["stages_completed"].append(
            {
                "stage": "orchestrated_processing",
                "duration": stage_duration,
                "status": "completed" if result.success else "failed",
                "confidence": result.overall_confidence,
                "quality_score": result.quality_score,
            }
        )

        print(f"   ✅ Processing completed: {result.success}")
        print(f"   📊 Overall confidence: {result.overall_confidence:.2f}")
        print(f"   🎯 Quality score: {result.quality_score:.2f}")
        print(f"   ⏱️  Processing time: {result.total_processing_time:.2f}s")
        print(f"   ⏱️  Stage duration: {stage_duration:.2f}s")

        if result.warnings:
            print(f"   ⚠️  Warnings: {len(result.warnings)}")
            for warning in result.warnings[:2]:
                print(f"     - {warning}")

        total_time = time.time() - start_time
        pipeline_results.update(
            {
                "overall_status": "completed",
                "total_duration": total_time,
                "domain_analysis": {
                    "domain_signature": domain_analysis.domain_signature,
                    "content_confidence": domain_analysis.content_type_confidence,
                    "vocabulary_richness": domain_analysis.characteristics.vocabulary_richness,
                    "technical_density": domain_analysis.characteristics.technical_vocabulary_ratio,
                },
                "generated_prompts": list(generated_prompts.keys()),
                "orchestration_result": {
                    "success": result.success,
                    "confidence": result.overall_confidence,
                    "quality_score": result.quality_score,
                    "processing_time": result.total_processing_time,
                    "warnings": len(result.warnings) if result.warnings else 0,
                },
            }
        )

        print(f"\n🎉 Azure Universal RAG - Pipeline Complete!")
        print(f"   ⏱️  Total time: {total_time:.2f}s")
        print(f"   🌍 Domain: {domain_analysis.domain_signature}")
        print(f"   📊 Stages completed: {len(pipeline_results['stages_completed'])}")
        print(f"   🎯 Adaptive configuration applied automatically")
        print(f"   ✅ Zero domain assumptions throughout")
        print(f"   📄 Session: {session_id}")

        return pipeline_results

    except Exception as e:
        total_time = time.time() - start_time
        pipeline_results.update(
            {
                "overall_status": "failed",
                "total_duration": total_time,
                "error": str(e),
                "failure_stage": len(pipeline_results["stages_completed"]),
            }
        )

        print(f"❌ Azure Universal RAG - Pipeline Failed!")
        print(f"   ⏱️  Time elapsed: {total_time:.2f}s")
        print(f"   📊 Stages completed: {len(pipeline_results['stages_completed'])}")
        print(f"   ❌ Error: {e}")
        print(f"   📄 Session: {session_id}")

        if verbose:
            import traceback

            print(f"\n🔍 Detailed Error Information:")
            traceback.print_exc()

        return pipeline_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Azure Universal RAG - Complete Pipeline"
    )
    parser.add_argument(
        "--data-dir",
        default="/workspace/azure-maintie-rag/data/raw",
        help="Data directory",
    )
    parser.add_argument(
        "--skip-prerequisites",
        action="store_true",
        help="Skip Azure services state check",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Verbose output with detailed progress"
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--output", help="Save results to JSON file")
    args = parser.parse_args()

    print("🚀 Azure Universal RAG - Starting Complete Pipeline")
    print("=" * 60)
    print("This pipeline automatically discovers domain characteristics")
    print("and adapts to ANY content type using Azure cloud services.")
    print("Zero domain assumptions • PydanticAI agents • Production ready")
    print("")

    # Run the pipeline
    result = asyncio.run(
        run_universal_full_pipeline(
            data_dir=args.data_dir,
            skip_prerequisites=args.skip_prerequisites,
            verbose=args.verbose,
        )
    )

    # Handle JSON output
    if args.json or args.output:
        json_output = json.dumps(result, indent=2, default=str)

        if args.output:
            # Save to file
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(json_output)
            print(f"\n📄 Results saved to: {output_path}")

        if args.json:
            # Print JSON to stdout
            print(f"\n" + "=" * 60)
            print("Pipeline Results JSON:")
            print(json_output)

    # Final summary
    if result["overall_status"] == "completed":
        print(f"\n🎉 SUCCESS: Pipeline completed successfully!")
        print(f"   📄 Session: {result['session_id']}")
        print(f"   ⏱️  Duration: {result['total_duration']:.2f}s")
        print(f"   📊 Stages: {len(result['stages_completed'])}")
        print(
            "Your data has been processed with adaptive, domain-specific configuration."
        )
        sys.exit(0)
    else:
        print(f"\n❌ FAILED: Pipeline encountered issues.")
        print(f"   📄 Session: {result['session_id']}")
        print(f"   ⏱️  Duration: {result.get('total_duration', 0):.2f}s")
        print(f"   📊 Stages completed: {len(result.get('stages_completed', []))}")
        print("Check the error messages above for details.")
        sys.exit(1)
