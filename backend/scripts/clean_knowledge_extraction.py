#!/usr/bin/env python3
"""
Clean Knowledge Extraction Script
Extracts only the final entities and relationships from raw text data
No extra files, no metadata, just pure knowledge extraction results
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.azure_openai.improved_extraction_client import ImprovedKnowledgeExtractor


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


def extract_clean_knowledge(texts: List[str], domain: str = "maintenance") -> Dict[str, Any]:
    """
    Extract clean knowledge - only entities and relationships
    No extra files, no metadata, just pure extraction results
    """

    print(f"\n🔍 Starting clean knowledge extraction...")
    print(f"   • Domain: {domain}")
    print(f"   • Texts to process: {len(texts)}")

    # Initialize the improved extractor
    extractor = ImprovedKnowledgeExtractor(domain)

    # Extract knowledge from each text
    all_entities = []
    all_relationships = []

    for i, text in enumerate(texts, 1):
        print(f"   Processing text {i}/{len(texts)}: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")

        # Extract from single text
        extraction = extractor._extract_from_single_text(text)

        # Add text reference to entities and relationships
        for entity in extraction.get('entities', []):
            entity['source_text'] = text
            entity['text_id'] = i
            all_entities.append(entity)

        for relation in extraction.get('relations', []):
            relation['source_text'] = text
            relation['text_id'] = i
            all_relationships.append(relation)

    # Create clean results
    results = {
        "entities": all_entities,
        "relationships": all_relationships,
        "summary": {
            "total_texts": len(texts),
            "total_entities": len(all_entities),
            "total_relationships": len(all_relationships),
            "avg_entities_per_text": round(len(all_entities) / len(texts), 2),
            "avg_relationships_per_text": round(len(all_relationships) / len(texts), 2)
        }
    }

    print(f"\n✅ Clean extraction completed!")
    print(f"   • Total entities: {len(all_entities)}")
    print(f"   • Total relationships: {len(all_relationships)}")
    print(f"   • Average entities per text: {results['summary']['avg_entities_per_text']}")
    print(f"   • Average relationships per text: {results['summary']['avg_relationships_per_text']}")

    return results


def display_clean_results(results: Dict[str, Any], show_details: bool = True):
    """Display the clean extraction results"""

    print(f"\n{'='*80}")
    print("📊 CLEAN KNOWLEDGE EXTRACTION RESULTS")
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

    if show_details:
        print(f"\n🔍 Sample Entities (first 5):")
        for i, entity in enumerate(entities[:5], 1):
            print(f"   {i}. \"{entity['text']}\" [{entity['entity_type']}]")
            print(f"      Context: \"{entity['context']}\"")
            print(f"      Source: Text {entity['text_id']}")

        print(f"\n🔗 Sample Relationships (first 5):")
        for i, relation in enumerate(relationships[:5], 1):
            # Handle both old and new relationship formats
            source = relation.get('source_entity', relation.get('source_entity_id', 'unknown'))
            target = relation.get('target_entity', relation.get('target_entity_id', 'unknown'))
            print(f"   {i}. {source} --[{relation['relation_type']}]--> {target}")
            print(f"      Confidence: {relation['confidence']}")
            print(f"      Source: Text {relation['text_id']}")

        if len(entities) > 5:
            print(f"\n   ... and {len(entities) - 5} more entities")
        if len(relationships) > 5:
            print(f"   ... and {len(relationships) - 5} more relationships")


def save_clean_results(results: Dict[str, Any], output_file: str = None) -> str:
    """Save clean results to JSON file"""

    if not output_file:
        output_file = f"clean_knowledge_extraction_{len(results['entities'])}_entities_{len(results['relationships'])}_relationships.json"

    output_path = Path(__file__).parent.parent / "data" / "extraction_outputs" / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Clean results saved to: {output_path}")
    return str(output_path)


def main():
    """Main function for clean knowledge extraction"""

    print("🧹 CLEAN KNOWLEDGE EXTRACTION")
    print("=" * 50)
    print("Extracts only entities and relationships from raw text")
    print("No extra files, no metadata, just pure knowledge data")

    # Load raw texts
    import argparse
    parser = argparse.ArgumentParser(description="Clean knowledge extraction from raw text")
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

    # Extract clean knowledge
    results = extract_clean_knowledge(texts, args.domain)

    # Display results
    display_clean_results(results, not args.no_details)

    # Save results
    output_path = save_clean_results(results, args.output)

    print(f"\n{'='*80}")
    print("🎯 CLEAN EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"\n📁 Results saved to: {output_path}")
    print(f"📊 Entities: {len(results['entities'])}")
    print(f"🔗 Relationships: {len(results['relationships'])}")
    print(f"\n💡 The JSON file contains only the pure knowledge data:")
    print(f"   • entities: List of extracted entities")
    print(f"   • relationships: List of extracted relationships")
    print(f"   • summary: Basic statistics")
    print(f"\n🚀 Ready for use in RAG systems, knowledge graphs, or further processing!")


if __name__ == "__main__":
    main()
