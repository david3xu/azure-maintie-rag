#!/usr/bin/env python3
"""
Auto Prompt Inspector Tool

Quick utility to inspect auto-generated prompts from Agent 1 → Agent 2 delegation.
Run this anytime you want to see what prompts are being generated.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.core.universal_deps import get_universal_deps
from agents.domain_intelligence.agent import domain_intelligence_agent
from agents.core.agent_toolsets import predict_entity_types, generate_extraction_prompts


class PromptInspector:
    """Utility class to inspect auto-generated prompts."""
    
    def __init__(self):
        self.deps = None
        
    async def initialize(self):
        """Initialize Azure dependencies."""
        self.deps = await get_universal_deps()
        print(f"🔧 Initialized with services: {list(self.deps.get_available_services())}")
    
    async def inspect_content_prompts(self, content: str, save_to_file: bool = False):
        """
        Inspect auto-generated prompts for given content.
        
        Args:
            content: Text content to analyze
            save_to_file: Whether to save prompts to JSON file
        """
        print("🔍 AUTO PROMPT INSPECTION")
        print("=" * 60)
        
        # Step 1: Domain Analysis
        print(f"📄 Content ({len(content)} chars): {content[:100]}...")
        print("\n1️⃣ Domain Analysis...")
        
        domain_result = await domain_intelligence_agent.run(
            f"Analyze for prompt generation: {content}",
            deps=self.deps
        )
        domain_analysis = domain_result.output
        
        print(f"   Domain signature: {domain_analysis.domain_signature}")
        print(f"   Vocabulary complexity: {domain_analysis.characteristics.vocabulary_complexity_ratio:.3f}")
        print(f"   Content patterns: {domain_analysis.characteristics.content_patterns}")
        print(f"   Key terms: {domain_analysis.characteristics.key_content_terms}")
        
        # Step 2: Auto Entity Type Prediction
        print("\n2️⃣ Auto Entity Type Prediction...")
        
        class MockRunContext:
            def __init__(self, deps): 
                self.deps = deps
        
        ctx = MockRunContext(self.deps)
        entity_predictions = await predict_entity_types(ctx, content, domain_analysis.characteristics)
        
        predicted_types = entity_predictions.get('predicted_entity_types', [])
        type_confidence = entity_predictions.get('type_confidence', {})
        
        print(f"   🤖 Generated types: {predicted_types}")
        print(f"   🎯 Confidence scores: {type_confidence}")
        print(f"   📊 Generation method: {entity_predictions.get('generation_method', 'unknown')}")
        
        # Step 3: Auto Extraction Prompts
        print("\n3️⃣ Auto Extraction Prompt Generation...")
        
        extraction_prompts = await generate_extraction_prompts(
            ctx, content, entity_predictions, domain_analysis.characteristics
        )
        
        for prompt_type, prompt_text in extraction_prompts.items():
            print(f"\n📝 {prompt_type.upper()} PROMPT:")
            print(f"   Length: {len(prompt_text)} characters")
            print(f"   Contains predicted types: {'predicted entity types' in prompt_text.lower()}")
            print("   Preview (first 200 chars):")
            print(f"   {prompt_text[:200]}...")
        
        # Step 4: Template Variables (for Jinja2 template)
        print("\n4️⃣ Template Variables (for Jinja2)...")
        
        template_vars = {
            'discovered_entity_types': predicted_types,
            'content_signature': domain_analysis.domain_signature,
            'key_content_terms': domain_analysis.characteristics.key_content_terms,
            'vocabulary_richness': domain_analysis.characteristics.vocabulary_richness,
            'concept_density': domain_analysis.characteristics.concept_density,
            'discovered_content_patterns': domain_analysis.characteristics.content_patterns,
            'entity_confidence_threshold': 0.7,
            'relationship_confidence_threshold': 0.6
        }
        
        print("   📋 Available template variables:")
        for var_name, var_value in template_vars.items():
            print(f"     {var_name}: {var_value}")
        
        # Save to file if requested
        if save_to_file:
            output_data = {
                'domain_analysis': {
                    'signature': domain_analysis.domain_signature,
                    'vocabulary_complexity': domain_analysis.characteristics.vocabulary_complexity_ratio,
                    'content_patterns': domain_analysis.characteristics.content_patterns,
                    'key_terms': domain_analysis.characteristics.key_content_terms
                },
                'entity_predictions': entity_predictions,
                'extraction_prompts': extraction_prompts,
                'template_variables': template_vars
            }
            
            timestamp = domain_analysis.domain_signature[:8]  # Use first 8 chars of signature
            output_file = Path(f"auto_prompts_inspection_{timestamp}.json")
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"\n💾 Saved inspection results to: {output_file}")
        
        print("\n" + "=" * 60)
        print("✅ Auto Prompt Inspection Complete!")
        
        return {
            'domain_analysis': domain_analysis,
            'entity_predictions': entity_predictions,
            'extraction_prompts': extraction_prompts,
            'template_vars': template_vars
        }


async def main():
    """Main function for prompt inspection."""
    inspector = PromptInspector()
    await inspector.initialize()
    
    # Option 1: Use sample content
    if len(sys.argv) < 2:
        # Load sample content from data/raw
        data_dir = Path(__file__).parent.parent.parent / "data/raw/azure-ai-services-language-service_output"
        sample_files = list(data_dir.glob("*.md"))
        
        if sample_files:
            sample_file = sample_files[0]  # Use first available file
            content = sample_file.read_text(encoding='utf-8', errors='ignore')[:800]
            print(f"📁 Using sample file: {sample_file.name}")
        else:
            content = """
            Azure Machine Learning provides comprehensive tools for building, training, and deploying 
            machine learning models at scale. It offers automated ML capabilities, model interpretability, 
            and robust MLOps features for production deployment scenarios.
            """
            print("📝 Using default sample content")
    else:
        # Option 2: Use provided content
        content = " ".join(sys.argv[1:])
        print("📝 Using provided content")
    
    # Inspect the prompts
    results = await inspector.inspect_content_prompts(content, save_to_file=True)
    
    print(f"\n🎯 SUMMARY:")
    print(f"   Domain signature: {results['domain_analysis'].domain_signature}")
    print(f"   Entity types generated: {len(results['entity_predictions'].get('predicted_entity_types', []))}")
    print(f"   Prompts generated: {len(results['extraction_prompts'])}")
    print(f"   Template variables: {len(results['template_vars'])}")


if __name__ == "__main__":
    asyncio.run(main())