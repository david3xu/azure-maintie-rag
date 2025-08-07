#!/usr/bin/env python3
"""
Universal Knowledge Extraction - Zero Domain Bias
Clean extraction using Universal RAG agents with automatic domain adaptation.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.knowledge_extraction.agent import agent as knowledge_extraction_agent, ExtractionDeps
from agents.domain_intelligence.agent import run_universal_domain_analysis, UniversalDomainDeps
from infrastructure.prompt_workflows.universal_prompt_generator import UniversalPromptGenerator


async def universal_knowledge_extraction(
    content: str, 
    data_directory: str = "/workspace/azure-maintie-rag/data/raw"
):
    """Universal knowledge extraction that adapts to content domain automatically"""
    print("🌍 Universal Knowledge Extraction - Zero Domain Bias")
    print("===================================================")

    try:
        # Step 1: Discover domain characteristics
        print("🔍 Step 1: Universal Domain Analysis")
        print(f"   📄 Content length: {len(content)} characters")
        print(f"   📁 Reference data: {data_directory}")
        
        domain_analysis = await run_universal_domain_analysis(
            UniversalDomainDeps(
                data_directory=data_directory,
                max_files_to_analyze=5,
                min_content_length=100,
                enable_multilingual=True
            )
        )
        
        print(f"   ✅ Domain discovered: {domain_analysis.domain_signature}")
        print(f"   📊 Technical density: {domain_analysis.characteristics.technical_vocabulary_ratio:.3f}")
        
        # Step 2: Generate adaptive extraction prompts
        print("\n📝 Step 2: Adaptive Prompt Generation")
        prompt_generator = UniversalPromptGenerator()
        generated_prompts = await prompt_generator.generate_domain_prompts(
            data_directory=data_directory,
            max_files_to_analyze=5
        )
        
        print(f"   ✅ Generated domain-adaptive extraction prompts")
        print(f"   🎯 Entity confidence threshold: {domain_analysis.processing_config.entity_confidence_threshold}")
        print(f"   📏 Optimal chunk size: {domain_analysis.processing_config.optimal_chunk_size}")
        
        # Step 3: Universal Knowledge Extraction
        print("\n🧠 Step 3: Adaptive Knowledge Extraction")
        
        # Configure extraction with discovered domain characteristics
        extraction_deps = ExtractionDeps(
            confidence_threshold=domain_analysis.processing_config.entity_confidence_threshold,
            max_entities_per_chunk=25  # Could be made adaptive based on content complexity
        )
        
        print(f"   🔧 Using adaptive configuration:")
        print(f"      Confidence threshold: {extraction_deps.confidence_threshold}")
        print(f"      Max entities per chunk: {extraction_deps.max_entities_per_chunk}")
        
        # Run extraction with universal agent
        extraction_result = await knowledge_extraction_agent.run(
            f"Extract knowledge from this {domain_analysis.domain_signature} content: {content[:500]}...",
            deps=extraction_deps
        )
        
        print(f"   ✅ Extraction completed successfully")
        print(f"   📊 Entities found: {extraction_result.data.entities_count}")
        print(f"   🔗 Relationships found: {extraction_result.data.relationships_count}")
        print(f"   🎯 Extraction confidence: {extraction_result.data.overall_confidence:.2f}")
        
        # Display sample results
        if extraction_result.data.sample_entities:
            print(f"\n📋 Sample Entities (domain-specific):")
            for i, entity in enumerate(extraction_result.data.sample_entities[:3], 1):
                print(f"   {i}. {entity}")
        
        if extraction_result.data.sample_relationships:
            print(f"\n🔗 Sample Relationships (domain-specific):")
            for i, rel in enumerate(extraction_result.data.sample_relationships[:2], 1):
                print(f"   {i}. {rel}")
        
        return {
            'success': True,
            'domain_analysis': domain_analysis,
            'extraction_result': extraction_result.data,
            'adaptive_prompts': generated_prompts,
            'configuration': {
                'confidence_threshold': extraction_deps.confidence_threshold,
                'max_entities': extraction_deps.max_entities_per_chunk,
                'domain_signature': domain_analysis.domain_signature
            }
        }

    except Exception as e:
        print(f"❌ Universal extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Universal knowledge extraction - adapts to ANY content domain")
    parser.add_argument("--content", required=True, help="Content to extract knowledge from")
    parser.add_argument("--data-dir", default="/workspace/azure-maintie-rag/data/raw", help="Reference data directory for domain analysis")
    args = parser.parse_args()

    print("🧠 Starting Universal Knowledge Extraction...")
    print("===========================================")
    print("This extraction automatically adapts to your content domain")
    print("without any hardcoded domain assumptions.")
    print("")
    
    result = asyncio.run(universal_knowledge_extraction(args.content, args.data_dir))
    
    if result['success']:
        print(f"\n🎉 SUCCESS: Universal extraction completed!")
        print(f"Domain: {result['configuration']['domain_signature']}")
        print(f"Entities: {result['extraction_result'].entities_count}")
        print(f"Relationships: {result['extraction_result'].relationships_count}")
        sys.exit(0)
    else:
        print(f"\n❌ FAILED: Universal extraction encountered issues.")
        sys.exit(1)
