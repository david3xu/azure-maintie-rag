#!/usr/bin/env python3
"""
Example Usage of Clean Knowledge Extraction
Shows how to use the clean extraction script for different scenarios
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.clean_knowledge_extraction import extract_clean_knowledge, load_raw_texts


def example_1_basic_extraction():
    """Example 1: Basic knowledge extraction"""
    print("=" * 60)
    print("EXAMPLE 1: Basic Knowledge Extraction")
    print("=" * 60)

    # Sample texts
    texts = [
        "air conditioner thermostat not working",
        "brake system pressure low",
        "compressor oil level needs checking"
    ]

    # Extract knowledge
    results = extract_clean_knowledge(texts, "maintenance")

    # Show results
    print(f"\n📊 Results:")
    print(f"   • Entities: {len(results['entities'])}")
    print(f"   • Relationships: {len(results['relationships'])}")

    # Show sample entities
    print(f"\n🔍 Sample Entities:")
    for entity in results['entities'][:3]:
        print(f"   • \"{entity['text']}\" [{entity['entity_type']}]")

    # Show sample relationships
    print(f"\n🔗 Sample Relationships:")
    for relation in results['relationships'][:3]:
        source = relation.get('source_entity_id', 'unknown')
        target = relation.get('target_entity_id', 'unknown')
        print(f"   • {source} --[{relation['relation_type']}]--> {target}")


def example_2_custom_domain():
    """Example 2: Custom domain extraction"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Custom Domain Extraction")
    print("=" * 60)

    # Medical domain texts
    medical_texts = [
        "patient has high blood pressure",
        "diabetes medication needs adjustment",
        "heart rate monitor showing irregular readings"
    ]

    # Extract with medical domain
    results = extract_clean_knowledge(medical_texts, "medical")

    print(f"\n📊 Medical Domain Results:")
    print(f"   • Entities: {len(results['entities'])}")
    print(f"   • Relationships: {len(results['relationships'])}")

    # Show medical entities
    print(f"\n🏥 Medical Entities:")
    for entity in results['entities'][:3]:
        print(f"   • \"{entity['text']}\" [{entity['entity_type']}]")


def example_3_file_input():
    """Example 3: Extract from file input"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: File Input Extraction")
    print("=" * 60)

    # Create a sample input file
    sample_file = Path(__file__).parent.parent / "data" / "sample_input.txt"
    sample_file.parent.mkdir(parents=True, exist_ok=True)

    sample_texts = [
        "engine oil filter needs replacement",
        "transmission fluid level low",
        "battery terminals corroded",
        "tire pressure sensors malfunctioning"
    ]

    with open(sample_file, 'w') as f:
        for text in sample_texts:
            f.write(text + '\n')

    print(f"📝 Created sample input file: {sample_file}")

    # Load from file
    texts = load_raw_texts(str(sample_file))

    # Extract knowledge
    results = extract_clean_knowledge(texts, "automotive")

    print(f"\n📊 File Input Results:")
    print(f"   • Entities: {len(results['entities'])}")
    print(f"   • Relationships: {len(results['relationships'])}")


def example_4_json_output_usage():
    """Example 4: Using the JSON output"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: JSON Output Usage")
    print("=" * 60)

    # Extract knowledge
    texts = ["air conditioner thermostat not working"]
    results = extract_clean_knowledge(texts, "maintenance")

    # Save to file
    output_file = Path(__file__).parent.parent / "data" / "extraction_outputs" / "example_output.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"💾 Saved results to: {output_file}")

    # Show how to use the JSON
    print(f"\n📋 JSON Structure:")
    print(f"   • entities: List of extracted entities")
    print(f"   • relationships: List of extracted relationships")
    print(f"   • summary: Statistics and metadata")

    # Show sample usage
    print(f"\n🔧 Sample Usage:")
    print(f"   # Load the JSON file")
    print(f"   with open('{output_file}', 'r') as f:")
    print(f"       data = json.load(f)")
    print(f"   ")
    print(f"   # Access entities")
    print(f"   for entity in data['entities']:")
    print(f"       print(f\"Entity: {{entity['text']}} [{{entity['entity_type']}}]\")")
    print(f"   ")
    print(f"   # Access relationships")
    print(f"   for relation in data['relationships']:")
    print(f"       print(f\"Relation: {{relation['relation_type']}}\")")


def example_5_command_line_usage():
    """Example 5: Command line usage"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Command Line Usage")
    print("=" * 60)

    print("💻 Command Line Examples:")
    print(f"   # Basic usage with sample texts")
    print(f"   python scripts/clean_knowledge_extraction.py")
    print(f"   ")
    print(f"   # Custom domain")
    print(f"   python scripts/clean_knowledge_extraction.py --domain medical")
    print(f"   ")
    print(f"   # Input from file")
    print(f"   python scripts/clean_knowledge_extraction.py --input data/sample_input.txt")
    print(f"   ")
    print(f"   # Custom output file")
    print(f"   python scripts/clean_knowledge_extraction.py --output my_results.json")
    print(f"   ")
    print(f"   # No detailed output")
    print(f"   python scripts/clean_knowledge_extraction.py --no-details")


def main():
    """Run all examples"""
    print("🧹 CLEAN KNOWLEDGE EXTRACTION EXAMPLES")
    print("=" * 60)
    print("This script demonstrates different ways to use the clean extraction")

    # Run examples
    example_1_basic_extraction()
    example_2_custom_domain()
    example_3_file_input()
    example_4_json_output_usage()
    example_5_command_line_usage()

    print(f"\n{'='*60}")
    print("🎯 ALL EXAMPLES COMPLETED")
    print(f"{'='*60}")
    print(f"\n💡 Key Benefits of Clean Extraction:")
    print(f"   ✅ No extra files or metadata")
    print(f"   ✅ Pure entities and relationships")
    print(f"   ✅ Easy to integrate into other systems")
    print(f"   ✅ Lightweight and fast")
    print(f"   ✅ Multiple domain support")
    print(f"\n🚀 Ready for RAG systems, knowledge graphs, and more!")


if __name__ == "__main__":
    main()
