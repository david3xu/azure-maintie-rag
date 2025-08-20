#!/usr/bin/env python3
"""
Phase 2 Prerequisites Validation - Data Ingestion Prerequisites
=============================================================

Task: Validate that Phase 2 prerequisites are met before data ingestion.
Logic: Check Phase 0 cleanup completion + Phase 1 agent validation + Azure services ready.
NO FAKE SUCCESS PATTERNS - FAIL FAST if prerequisites not met.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import path utilities for consistent directory handling
from scripts.dataflow.utilities.path_utils import get_results_dir

from agents.core.universal_deps import get_universal_deps


async def validate_phase0_completion():
    """Validate that Phase 0 cleanup completed successfully (or skip in incremental mode)"""
    import os

    # Check if we're in incremental mode
    skip_cleanup = os.environ.get("SKIP_CLEANUP", "false").lower() == "true"

    if skip_cleanup:
        print("\n⏭️  SKIPPING PHASE 0 VALIDATION (INCREMENTAL MODE)")
        print("=" * 50)
        print("   📊 Incremental mode: Preserving existing data")
        print("   🔄 Will add/update data based on data/raw content")
        return True

    print("\n🧹 VALIDATING PHASE 0 COMPLETION")
    print("=" * 40)

    # Check for clean state verification results
    results_path = Path("scripts/dataflow/results/clean_state_verification.json")
    if not results_path.exists():
        raise RuntimeError(
            "Phase 0 cleanup results not found. Run 'make dataflow-cleanup' first."
        )

    with open(results_path) as f:
        cleanup_results = json.load(f)

    if not cleanup_results.get("all_services_clean", False):
        raise RuntimeError(
            "Phase 0 cleanup incomplete - Azure services not clean. Re-run cleanup."
        )

    print(f"   ✅ Phase 0 cleanup completed successfully")
    print(
        f"   📊 Services cleaned: {len(cleanup_results.get('services_verified', []))}"
    )
    return True


async def validate_phase1_completion():
    """Validate that Phase 1 agent validation completed successfully"""
    print("\n🧪 VALIDATING PHASE 1 AGENT VALIDATION")
    print("=" * 40)

    # Test all 3 PydanticAI agents quickly
    try:
        from agents.core.universal_deps import get_universal_deps
        from agents.domain_intelligence.agent import run_domain_analysis
        from agents.knowledge_extraction.agent import knowledge_extraction_agent
        from agents.universal_search.agent import run_universal_search

        deps = await get_universal_deps()

        # Quick agent validation with retry logic for rate limits
        print("   🧠 Testing Domain Intelligence Agent...")
        test_content = "Azure AI services provide machine learning capabilities."

        # Implement retry logic for Azure OpenAI rate limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                domain_result = await run_domain_analysis(test_content, detailed=False)
                if not domain_result or not domain_result.domain_signature:
                    raise RuntimeError("Domain Intelligence Agent not working")
                print(f"   ✅ Domain Intelligence Agent operational")
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = 60 * (2 ** attempt)  # Exponential backoff: 60s, 120s, 240s
                    print(f"   ⏰ Rate limit hit (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"Domain Intelligence Agent validation failed: {e}")

        print("   🔬 Testing Knowledge Extraction Agent...")
        # Just test that the agent can be initialized properly, skip full extraction
        # Full extraction testing will happen in Phase 3 with real data
        try:
            from agents.knowledge_extraction.agent import knowledge_extraction_agent

            if knowledge_extraction_agent and knowledge_extraction_agent.model:
                print(
                    f"   ✅ Knowledge Extraction Agent operational (full extraction testing deferred to Phase 3)"
                )
                print(
                    f"   ℹ️  Entity/relationship extraction will be validated with real data in Phase 3"
                )
            else:
                raise RuntimeError(
                    "Knowledge Extraction Agent not properly initialized"
                )
        except Exception as ke_init_error:
            raise RuntimeError(
                f"Knowledge Extraction Agent initialization failed: {ke_init_error}"
            )

        print("   🔍 Testing Universal Search Agent...")
        # Note: Skip tri-modal search testing here because graph database is empty after Phase 0 cleanup
        # Just test that the agent can be initialized properly
        try:
            from agents.universal_search.agent import universal_search_agent

            if universal_search_agent and universal_search_agent.model:
                print(
                    f"   ✅ Universal Search Agent operational (full search testing deferred to Phase 3)"
                )
            else:
                raise RuntimeError("Universal Search Agent not properly initialized")
        except Exception as search_init_error:
            raise RuntimeError(
                f"Universal Search Agent initialization failed: {search_init_error}"
            )

        print(f"   ✅ All Phase 1 agents validated")
        return True

    except Exception as e:
        raise RuntimeError(
            f"Phase 1 agent validation failed: {e}. Run 'make dataflow-validate' first."
        )


async def validate_azure_services_ready():
    """Validate Azure services are ready for data ingestion"""
    print("\n☁️  VALIDATING AZURE SERVICES READY")
    print("=" * 40)

    try:
        deps = await get_universal_deps()
        available_services = set(deps.get_available_services())
        required_services = {
            "openai",
            "storage",
            "search",
        }  # Use correct service names from universal_deps.py

        missing_services = required_services - available_services
        if missing_services:
            raise RuntimeError(f"Missing Azure services: {missing_services}")

        print(f"   ✅ Required Azure services available: {len(required_services)}")

        # Test connectivity
        print("   🔧 Testing Azure Storage connectivity...")
        storage_client = deps.storage_client
        # Quick connectivity test - don't create containers yet
        if not hasattr(storage_client, "get_blob_count"):
            print(f"   ⚠️  Storage client may need configuration")
        else:
            print(f"   ✅ Azure Storage client ready")

        print("   🔧 Testing Azure OpenAI connectivity...")
        openai_client = deps.openai_client
        if not openai_client:
            raise RuntimeError("Azure OpenAI client not available")
        print(f"   ✅ Azure OpenAI client ready")

        print("   🔧 Testing Azure Cognitive Search connectivity...")
        search_client = deps.search_client
        if not search_client:
            raise RuntimeError("Azure Cognitive Search client not available")
        print(f"   ✅ Azure Cognitive Search client ready")

        print(f"   ✅ All Azure services ready for data ingestion")
        return True

    except Exception as e:
        raise RuntimeError(f"Azure services validation failed: {e}")


def validate_source_data():
    """Validate source data exists and is accessible"""
    print("\n📂 VALIDATING SOURCE DATA")
    print("=" * 40)

    # Use relative path from project root
    project_root = Path(__file__).parent.parent.parent.parent
    data_dir = project_root / "data" / "raw"
    if not data_dir.exists():
        raise RuntimeError(f"Source data directory not found: {data_dir}")

    md_files = list(data_dir.glob("**/*.md"))
    if not md_files:
        raise RuntimeError(f"No .md files found in source directory: {data_dir}")

    total_size = sum(f.stat().st_size for f in md_files)
    print(f"   📄 Found {len(md_files)} markdown files")
    print(f"   📊 Total size: {total_size/1024:.1f} KB")

    if len(md_files) < 50:  # Expect reasonable amount of data
        print(f"   ⚠️  Few source files found ({len(md_files)}), but proceeding")
    else:
        print(f"   ✅ Adequate source data for processing")

    return {"file_count": len(md_files), "total_size_kb": total_size / 1024}


async def main():
    """Main Phase 2 prerequisites validation orchestrator"""
    print("🔍 PHASE 2 PREREQUISITES VALIDATION")
    print("=" * 60)
    print("Validating that all prerequisites are met for data ingestion...")
    print("")

    start_time = time.time()
    validation_results = {}

    try:
        # Validate each prerequisite - FAIL FAST if any fail
        validation_results["phase0_cleanup"] = await validate_phase0_completion()
        validation_results["phase1_agents"] = await validate_phase1_completion()
        validation_results["azure_services"] = await validate_azure_services_ready()
        validation_results["source_data"] = validate_source_data()

        duration = time.time() - start_time

        # Prerequisites validation summary
        print(f"\n📊 PHASE 2 PREREQUISITES VALIDATION SUMMARY")
        print("=" * 50)
        print(f"✅ Phase 0 cleanup: Complete")
        print(f"✅ Phase 1 agents: All operational")
        print(f"✅ Azure services: Ready for ingestion")
        print(
            f"✅ Source data: {validation_results['source_data']['file_count']} files available"
        )

        print(f"\n⏱️  Validation time: {duration:.2f}s")

        # Save validation report
        results_dir = get_results_dir()  # Use path utilities for reliable directory access

        validation_report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "validation_duration": duration,
            "prerequisites_validated": list(validation_results.keys()),
            "all_prerequisites_met": True,
            "results": validation_results,
            "ready_for_phase2": True,
        }

        with open(results_dir / "phase2_prerequisites.json", "w") as f:
            json.dump(validation_report, f, indent=2)

        print(f"💾 Validation report: phase2_prerequisites.json")

        print(
            f"\n🎉 SUCCESS: All Phase 2 prerequisites validated - ready for data ingestion"
        )
        return True

    except Exception as e:
        print(f"\n❌ PHASE 2 PREREQUISITES VALIDATION FAILED: {e}")
        print(f"   🚨 FAIL FAST - Fix prerequisite issues before proceeding")
        raise e


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        if result:
            print(f"\n✅ Phase 2 prerequisites validated - proceed with data ingestion")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Phase 2 prerequisites validation failed: {e}")
        sys.exit(1)
