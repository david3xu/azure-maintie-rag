#!/usr/bin/env python3
"""
Universal Full Pipeline - Zero Domain Bias
Clean pipeline using Universal RAG agents that adapt to ANY content type.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.orchestrator import UniversalOrchestrator
from agents.domain_intelligence.agent import run_universal_domain_analysis, UniversalDomainDeps
from infrastructure.prompt_workflows.universal_prompt_generator import UniversalPromptGenerator


async def run_universal_full_pipeline(data_dir: str = "/workspace/azure-maintie-rag/data/raw"):
    """Universal full pipeline that adapts to ANY content type without domain assumptions"""
    print("🌍 Universal Full Pipeline - Zero Domain Bias")
    print("===========================================")
    
    start_time = time.time()
    
    try:
        # Stage 1: Universal Domain Discovery
        print(f"🔍 Stage 1: Universal Domain Analysis")
        print(f"   📁 Analyzing content from: {data_dir}")
        
        domain_analysis = await run_universal_domain_analysis(
            UniversalDomainDeps(
                data_directory=data_dir,
                max_files_to_analyze=10,
                min_content_length=200,
                enable_multilingual=True
            )
        )
        
        print(f"   ✅ Domain discovered: {domain_analysis.domain_signature}")
        print(f"   📊 Content confidence: {domain_analysis.content_type_confidence:.2f}")
        print(f"   🧠 Vocabulary richness: {domain_analysis.characteristics.vocabulary_richness:.3f}")
        print(f"   ⚙️  Technical density: {domain_analysis.characteristics.technical_vocabulary_ratio:.3f}")
        
        # Stage 2: Universal Prompt Generation
        print(f"\n📝 Stage 2: Universal Prompt Generation")
        prompt_generator = UniversalPromptGenerator()
        generated_prompts = await prompt_generator.generate_domain_prompts(
            data_directory=data_dir,
            max_files_to_analyze=10
        )
        
        print(f"   ✅ Generated {len(generated_prompts)} domain-adaptive prompts")
        for prompt_type, path in generated_prompts.items():
            print(f"   📄 {prompt_type}: {Path(path).name}")
        
        # Stage 3: Universal Orchestrated Processing
        print(f"\n🚀 Stage 3: Universal Orchestrated Processing")
        orchestrator = UniversalOrchestrator()
        
        result = await orchestrator.process_universal_workflow(
            data_directory=data_dir,
            query="analyze content and extract knowledge",
            enable_extraction=True,
            enable_search=True
        )
        
        print(f"   ✅ Processing completed: {result.success}")
        print(f"   📊 Overall confidence: {result.overall_confidence:.2f}")
        print(f"   🎯 Quality score: {result.quality_score:.2f}")
        print(f"   ⏱️  Processing time: {result.total_processing_time:.2f}s")
        
        if result.warnings:
            print(f"   ⚠️  Warnings: {len(result.warnings)}")
            for warning in result.warnings[:2]:
                print(f"     - {warning}")
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 Universal Pipeline Complete!")
        print(f"   ⏱️  Total time: {total_time:.2f}s")
        print(f"   🌍 Domain: {domain_analysis.domain_signature}")
        print(f"   🎯 Adaptive configuration applied automatically")
        print(f"   ✅ Zero domain assumptions throughout")
        
        return {
            'success': True,
            'domain_analysis': domain_analysis,
            'generated_prompts': generated_prompts,
            'orchestration_result': result,
            'total_time': total_time
        }

    except Exception as e:
        total_time = time.time() - start_time
        print(f"❌ Universal pipeline failed: {e}")
        print(f"   ⏱️  Time elapsed: {total_time:.2f}s")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'total_time': total_time
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Universal full pipeline - adapts to ANY content type")
    parser.add_argument("--data-dir", default="/workspace/azure-maintie-rag/data/raw", help="Data directory")
    args = parser.parse_args()

    print("🌍 Starting Universal Full Pipeline...")
    print("=====================================")
    print("This pipeline automatically discovers domain characteristics")
    print("and adapts to ANY content type without hardcoded assumptions.")
    print("")
    
    result = asyncio.run(run_universal_full_pipeline(args.data_dir))
    
    if result['success']:
        print("\n🎉 SUCCESS: Universal pipeline completed successfully!")
        print("Your data has been processed with adaptive, domain-specific configuration.")
        sys.exit(0)
    else:
        print("\n❌ FAILED: Universal pipeline encountered issues.")
        print("Check the error messages above for details.")
        sys.exit(1)
