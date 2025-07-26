#!/usr/bin/env python3
"""
Clean Knowledge Extraction Script (Prompt Flow Version)
Extracts only the final entities and relationships from raw text data
Uses centralized Prompt Flow templates instead of hardcoded prompts
"""

import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.prompt_flow.prompt_flow_integration import AzurePromptFlowIntegrator


def load_raw_texts(file_path: str = None) -> List[str]:
    """Load raw text data from file or use sample data"""

    if file_path and Path(file_path).exists():
        # Load from specified file
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"✅ Loaded {len(texts)} texts from: {file_path}")
        return texts

    # Sample maintenance texts for demonstration
    sample_texts = [
        "air conditioner thermostat not working",
        "air receiver safety valves to be replaced",
        "analyse failed driveline component",
        "auxiliary Cat engine lube service",
        "axle temperature sensor fault",
        "back rest unserviceable handle broken",
        "air horn not working compressor awaiting",
        "air leak near side of door",
        "brake system pressure low",
        "compressor oil level needs checking",
        "coolant temperature sensor malfunction",
        "diesel engine fuel filter clogged",
        "electrical system ground fault detected",
        "hydraulic pump pressure relief valve stuck",
        "ignition system spark plugs worn out"
    ]

    print(f"✅ Using {len(sample_texts)} sample maintenance texts")
    return sample_texts


async def extract_clean_knowledge_prompt_flow(texts: List[str], domain: str = "maintenance") -> Dict[str, Any]:
    """
    Extract clean knowledge using centralized Prompt Flow templates
    No extra files, no metadata, just pure extraction results
    """

    print(f"\n🔍 Starting clean knowledge extraction with Prompt Flow...")
    print(f"   • Domain: {domain}")
    print(f"   • Texts to process: {len(texts)}")
    print(f"   • Using centralized templates: prompt_flows/universal_knowledge_extraction/")

    # Initialize the Prompt Flow integrator
    integrator = AzurePromptFlowIntegrator(domain)

    # Extract knowledge using centralized prompts
    results = await integrator.extract_knowledge_with_prompt_flow(
        texts=texts,
        max_entities=50,
        confidence_threshold=0.7
    )

    # Transform to clean format
    clean_results = {
        "entities": results.get('entities', []),
        "relationships": results.get('relations', []),
        "summary": {
            "total_texts": len(texts),
            "total_entities": len(results.get('entities', [])),
            "total_relationships": len(results.get('relations', [])),
            "avg_entities_per_text": round(len(results.get('entities', [])) / len(texts), 2),
            "avg_relationships_per_text": round(len(results.get('relations', [])) / len(texts), 2),
            "extraction_method": "prompt_flow_centralized",
            "templates_used": [
                "prompt_flows/universal_knowledge_extraction/entity_extraction.jinja2",
                "prompt_flows/universal_knowledge_extraction/relation_extraction.jinja2"
            ]
        }
    }

    print(f"\n✅ Clean extraction completed with Prompt Flow!")
    print(f"   • Total entities: {len(clean_results['entities'])}")
    print(f"   • Total relationships: {len(clean_results['relationships'])}")
    print(f"   • Average entities per text: {clean_results['summary']['avg_entities_per_text']}")
    print(f"   • Average relationships per text: {clean_results['summary']['avg_relationships_per_text']}")
    print(f"   • Using centralized templates: ✅")

    return clean_results


def display_clean_results(results: Dict[str, Any], show_details: bool = True):
    """Display the clean extraction results"""

    print(f"\n{'='*80}")
    print("📊 CLEAN KNOWLEDGE EXTRACTION RESULTS (PROMPT FLOW)")
    print(f"{'='*80}")

    summary = results['summary']
    entities = results['entities']
    relationships = results['relationships']

    print(f"\n📈 Summary Statistics:")
    print(f"   • Texts processed: {summary['total_texts']}")
    print(f"   • Entities extracted: {summary['total_entities']}")
    print(f"   • Relationships extracted: {summary['total_relationships']}")
    print(f"   • Avg entities per text: {summary['avg_entities_per_text']}")
    print(f"   • Avg relationships per text: {summary['avg_relationships_per_text']}")
    print(f"   • Extraction method: {summary['extraction_method']}")
    print(f"   • Templates used: {len(summary['templates_used'])} centralized templates")

    if show_details:
        print(f"\n🔍 Sample Entities (first 5):")
        for i, entity in enumerate(entities[:5], 1):
            print(f"   {i}. \"{entity.get('text', entity.get('name', 'unknown'))}\" [{entity.get('entity_type', 'unknown')}]")
            print(f"      Context: \"{entity.get('context', 'N/A')}\"")
            print(f"      Source: Text {entity.get('text_id', 'N/A')}")

        print(f"\n🔗 Sample Relationships (first 5):")
        for i, relation in enumerate(relationships[:5], 1):
            source = relation.get('source_entity', relation.get('source_entity_id', 'unknown'))
            target = relation.get('target_entity', relation.get('target_entity_id', 'unknown'))
            print(f"   {i}. {source} --[{relation.get('relation_type', 'unknown')}]--> {target}")
            print(f"      Confidence: {relation.get('confidence', 'N/A')}")
            print(f"      Source: Text {relation.get('text_id', 'N/A')}")

        if len(entities) > 5:
            print(f"\n   ... and {len(entities) - 5} more entities")
        if len(relationships) > 5:
            print(f"   ... and {len(relationships) - 5} more relationships")

        print(f"\n📁 Centralized Templates Used:")
        for template in summary['templates_used']:
            print(f"   ✅ {template}")


def save_clean_results(results: Dict[str, Any], output_file: str = None) -> str:
    """Save clean results to JSON file"""

    if not output_file:
        output_file = f"clean_knowledge_extraction_prompt_flow_{len(results['entities'])}_entities_{len(results['relationships'])}_relationships.json"

    output_path = Path(__file__).parent.parent / "data" / "extraction_outputs" / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Clean results saved to: {output_path}")
    return str(output_path)


async def main():
    """Main function for clean knowledge extraction with Prompt Flow"""

    print("🧹 CLEAN KNOWLEDGE EXTRACTION (PROMPT FLOW)")
    print("=" * 60)
    print("Extracts only entities and relationships from raw text")
    print("Uses centralized Prompt Flow templates instead of hardcoded prompts")

    # Load raw texts
    import argparse
    parser = argparse.ArgumentParser(description="Clean knowledge extraction using Prompt Flow templates")
    parser.add_argument("--input", "-i", help="Input file with raw texts (one per line)")
    parser.add_argument("--domain", "-d", default="maintenance", help="Domain for extraction")
    parser.add_argument("--output", "-o", help="Output file for results")
    parser.add_argument("--no-details", action="store_true", help="Don't show detailed results")

    args = parser.parse_args()

    # Load texts
    texts = load_raw_texts(args.input)

    if not texts:
        print("❌ No texts to process!")
        return

    # Extract clean knowledge using Prompt Flow
    results = await extract_clean_knowledge_prompt_flow(texts, args.domain)

    # Display results
    display_clean_results(results, not args.no_details)

    # Save results
    output_path = save_clean_results(results, args.output)

    print(f"\n{'='*80}")
    print("🎯 CLEAN EXTRACTION WITH PROMPT FLOW COMPLETE")
    print(f"{'='*80}")
    print(f"\n📁 Results saved to: {output_path}")
    print(f"📊 Entities: {len(results['entities'])}")
    print(f"🔗 Relationships: {len(results['relationships'])}")
    print(f"\n💡 Key Benefits of Prompt Flow Version:")
    print(f"   ✅ Uses centralized templates from prompt_flows/universal_knowledge_extraction/")
    print(f"   ✅ No hardcoded prompts in code")
    print(f"   ✅ Easy to modify prompts without code changes")
    print(f"   ✅ Maintains universal principles")
    print(f"   ✅ Pure entities and relationships output")
    print(f"\n🚀 Ready for use in RAG systems, knowledge graphs, or further processing!")


if __name__ == "__main__":
    asyncio.run(main())
