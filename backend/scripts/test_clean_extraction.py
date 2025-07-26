#!/usr/bin/env python3
"""
Test Clean Knowledge Extraction Script
Extracts only the final entities and relationships from first 100 lines of raw text
Uses hardcoded prompts for reliable testing
"""

import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.azure_openai.improved_extraction_client import ImprovedKnowledgeExtractor


def load_sample_texts(file_path: str = None, max_lines: int = 100) -> List[str]:
    """Load first N lines of raw text data from file or use sample data"""

    if file_path and Path(file_path).exists():
        # Load first N lines from specified file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('='):
                    lines.append(line)
        print(f"✅ Loaded {len(lines)} texts from first {max_lines} lines of: {file_path}")
        return lines

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


async def extract_clean_knowledge_test(texts: List[str], domain: str = "maintenance") -> Dict[str, Any]:
    """
    Extract clean knowledge using hardcoded prompts (reliable for testing)
    No extra files, no metadata, just pure extraction results
    """

    print(f"\n🔍 Starting clean knowledge extraction test...")
    print(f"   • Domain: {domain}")
    print(f"   • Texts to process: {len(texts)}")
    print(f"   • Using hardcoded prompts for reliability")

    # Initialize the extractor
    extractor = ImprovedKnowledgeExtractor(domain)

    all_entities = []
    all_relationships = []

    # Process each text individually for better control
    for i, text in enumerate(texts[:10], 1):  # Limit to first 10 for testing
        try:
            print(f"   Processing text {i}/{len(texts[:10])}: {text[:50]}...")

            # Extract entities and relationships
            entities = extractor._extract_entities_with_context(text)
            relationships = extractor._extract_relations_with_linking(text, entities)

            # Add text ID to entities and relationships
            for entity in entities:
                entity['text_id'] = i
                entity['source_text'] = text[:100] + "..." if len(text) > 100 else text

            for relation in relationships:
                relation['text_id'] = i
                relation['source_text'] = text[:100] + "..." if len(text) > 100 else text

            all_entities.extend(entities)
            all_relationships.extend(relationships)

        except Exception as e:
            print(f"   ⚠️  Error processing text {i}: {e}")
            continue

    # Transform to clean format
    clean_results = {
        "entities": all_entities,
        "relationships": all_relationships,
        "summary": {
            "total_texts": len(texts[:10]),
            "total_entities": len(all_entities),
            "total_relationships": len(all_relationships),
            "avg_entities_per_text": round(len(all_entities) / len(texts[:10]), 2) if texts[:10] else 0,
            "avg_relationships_per_text": round(len(all_relationships) / len(texts[:10]), 2) if texts[:10] else 0,
            "extraction_method": "hardcoded_prompts_test",
            "processing_notes": "Limited to first 10 texts for testing"
        }
    }

    print(f"\n✅ Clean extraction test completed!")
    print(f"   • Total entities: {len(clean_results['entities'])}")
    print(f"   • Total relationships: {len(clean_results['relationships'])}")
    print(f"   • Average entities per text: {clean_results['summary']['avg_entities_per_text']}")
    print(f"   • Average relationships per text: {clean_results['summary']['avg_relationships_per_text']}")
    print(f"   • Using hardcoded prompts: ✅")

    return clean_results


def display_clean_results(results: Dict[str, Any], show_details: bool = True):
    """Display the clean extraction results"""

    print(f"\n{'='*80}")
    print("📊 CLEAN KNOWLEDGE EXTRACTION TEST RESULTS")
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
    print(f"   • Processing notes: {summary['processing_notes']}")

    if show_details and entities:
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

    elif not entities:
        print(f"\n⚠️  No entities extracted - this may indicate an issue with the extraction process")


def save_clean_results(results: Dict[str, Any], output_file: str = None) -> str:
    """Save clean results to JSON file"""

    if not output_file:
        output_file = f"test_clean_extraction_{len(results['entities'])}_entities_{len(results['relationships'])}_relationships.json"

    output_path = Path(__file__).parent.parent / "data" / "extraction_outputs" / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Test results saved to: {output_path}")
    return str(output_path)


async def main():
    """Main function for test clean knowledge extraction"""

    print("🧪 TEST CLEAN KNOWLEDGE EXTRACTION")
    print("=" * 60)
    print("Extracts only entities and relationships from first 100 lines")
    print("Uses hardcoded prompts for reliable testing")

    # Load sample texts (first 100 lines)
    import argparse
    parser = argparse.ArgumentParser(description="Test clean knowledge extraction with limited data")
    parser.add_argument("--input", "-i", help="Input file with raw texts")
    parser.add_argument("--domain", "-d", default="maintenance", help="Domain for extraction")
    parser.add_argument("--output", "-o", help="Output file for results")
    parser.add_argument("--max-lines", "-m", type=int, default=100, help="Maximum lines to process")
    parser.add_argument("--no-details", action="store_true", help="Don't show detailed results")

    args = parser.parse_args()

    # Load texts (limited to first N lines)
    texts = load_sample_texts(args.input, args.max_lines)

    if not texts:
        print("❌ No texts to process!")
        return

    # Extract clean knowledge
    results = await extract_clean_knowledge_test(texts, args.domain)

    # Display results
    display_clean_results(results, not args.no_details)

    # Save results
    output_path = save_clean_results(results, args.output)

    print(f"\n{'='*80}")
    print("🎯 TEST EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"\n📁 Results saved to: {output_path}")
    print(f"📊 Entities: {len(results['entities'])}")
    print(f"🔗 Relationships: {len(results['relationships'])}")
    print(f"\n💡 Test Features:")
    print(f"   ✅ Limited to first {args.max_lines} lines for testing")
    print(f"   ✅ Uses hardcoded prompts for reliability")
    print(f"   ✅ Processes only first 10 texts for speed")
    print(f"   ✅ Pure entities and relationships output")
    print(f"\n🚀 Ready for integration testing!")


if __name__ == "__main__":
    asyncio.run(main())
