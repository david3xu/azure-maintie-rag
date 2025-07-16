"""
Universal RAG Migration Demo
Demonstrates how to migrate from domain-specific RAG to Universal RAG using existing MaintIE data
"""

import logging
from pathlib import Path
from universal.maintie_data_adapter import MaintIEDataAdapter
from universal.optimized_llm_extractor import OptimizedLLMExtractor
from universal.domain_config_validator import DomainConfigValidator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_migration():
    """Demonstrate Universal RAG migration using existing MaintIE data"""

    print("🚀 Universal RAG Migration Demo")
    print("=" * 50)
    print()

    # Step 1: Preview existing MaintIE data
    print("📊 Step 1: Previewing existing MaintIE data...")
    print("-" * 30)

    try:
        adapter = MaintIEDataAdapter()

        # Preview different quality levels
        for quality in ["high", "mixed", "processed"]:
            print(f"\n🔍 Quality Filter: {quality}")

            corpus_info = adapter.create_domain_specific_corpus(
                domain_name="maintenance",
                quality_filter=quality
            )

            stats = corpus_info["statistics"]
            print(f"   📊 Total texts: {stats['total_texts']}")
            print(f"   📊 Avg length: {stats['avg_length']:.1f} chars")
            print(f"   📊 Total characters: {stats['total_characters']:,}")

        print("\n✅ Data preview complete!")

    except Exception as e:
        print(f"❌ Data preview failed: {e}")
        print("💡 Make sure you have MaintIE data in backend/data/raw/")
        return

    # Step 2: Show sample texts
    print(f"\n📝 Step 2: Sample texts from your data...")
    print("-" * 30)

    try:
        sample_texts = adapter.get_sample_texts(n_samples=3)

        for i, text in enumerate(sample_texts, 1):
            preview = text[:150] + "..." if len(text) > 150 else text
            print(f"\n{i}. {preview}")

    except Exception as e:
        print(f"❌ Sample extraction failed: {e}")

    # Step 3: Demonstrate knowledge extraction
    print(f"\n🧠 Step 3: Demonstrating LLM knowledge extraction...")
    print("-" * 30)

    try:
        # Use a small sample for demo
        corpus_info = adapter.create_domain_specific_corpus(
            domain_name="maintenance_demo",
            quality_filter="high"  # Use high quality for demo
        )

        # Limit to small sample for quick demo
        demo_texts = corpus_info["texts"][:20]  # Just 20 texts for demo

        print(f"💡 Demo: Using {len(demo_texts)} texts for quick extraction...")

        extractor = OptimizedLLMExtractor("maintenance_demo")
        extractor.max_texts_for_discovery = 10  # Small sample for demo

        # This would normally call OpenAI - for demo we'll simulate
        print("🔄 [Demo Mode] Simulating LLM knowledge extraction...")

        # Simulated result
        demo_knowledge = {
            "domain_name": "maintenance_demo",
            "entities": [
                "equipment", "pump", "motor", "bearing", "seal",
                "pressure", "temperature", "vibration", "oil", "filter"
            ],
            "relationships": [
                "has_part", "located_in", "requires", "causes", "prevents"
            ],
            "triplets": [
                ("pump", "has_part", "motor"),
                ("motor", "has_part", "bearing"),
                ("bearing", "requires", "oil"),
                ("vibration", "causes", "failure"),
                ("filter", "prevents", "contamination")
            ],
            "statistics": {
                "documents_processed": len(demo_texts),
                "entities_discovered": 10,
                "relationships_discovered": 5,
                "triplets_extracted": 5
            }
        }

        print(f"✅ Demo extraction complete!")
        print(f"   📊 Entities: {len(demo_knowledge['entities'])}")
        print(f"   📊 Relationships: {len(demo_knowledge['relationships'])}")
        print(f"   📊 Triplets: {len(demo_knowledge['triplets'])}")

        # Show extracted knowledge
        print(f"\n🏷️  Discovered Entities:")
        for entity in demo_knowledge['entities']:
            print(f"   - {entity}")

        print(f"\n🔗 Discovered Relationships:")
        for relation in demo_knowledge['relationships']:
            print(f"   - {relation}")

        print(f"\n📝 Sample Knowledge Triplets:")
        for triplet in demo_knowledge['triplets']:
            print(f"   - {triplet[0]} → {triplet[1]} → {triplet[2]}")

    except Exception as e:
        print(f"❌ Knowledge extraction demo failed: {e}")

    # Step 4: Validation demo
    print(f"\n🔍 Step 4: Demonstrating validation...")
    print("-" * 30)

    try:
        validator = DomainConfigValidator("maintenance_demo")
        is_valid, errors, warnings = validator.validate_domain_config(demo_knowledge)

        print(f"✅ Validation result: {'PASS' if is_valid else 'FAIL'}")

        if errors:
            print(f"❌ Errors: {len(errors)}")
            for error in errors:
                print(f"   - {error}")

        if warnings:
            print(f"⚠️  Warnings: {len(warnings)}")
            for warning in warnings:
                print(f"   - {warning}")

        if is_valid:
            print("🎉 Configuration is valid and ready for Universal RAG!")

    except Exception as e:
        print(f"❌ Validation demo failed: {e}")

    # Step 5: Migration summary
    print(f"\n🎯 Step 5: Migration Summary")
    print("-" * 30)

    print("""
🚀 Universal RAG Migration Benefits:

✅ ZERO manual configuration needed
✅ Uses your existing MaintIE data directly
✅ Automatic domain knowledge discovery
✅ Quality validation built-in
✅ Compatible with existing infrastructure

📋 Next Steps for Full Migration:

1. Run: python universal_rag_enhanced.py preview-maintie --name=maintenance
2. Run: python universal_rag_enhanced.py create-from-maintie --name=maintenance
3. Test with: python universal_rag_enhanced.py test-domain --name=maintenance

💡 This approach will create a Universal RAG system that:
   - Works with any domain (medical, legal, financial, etc.)
   - Uses the same infrastructure you already have
   - Requires only raw text data as input
   - Automatically discovers domain knowledge patterns
""")

def demo_cli_usage():
    """Show CLI usage examples"""

    print("\n📖 CLI Usage Examples:")
    print("-" * 30)

    examples = [
        ("Preview your MaintIE data",
         "python universal_rag_enhanced.py preview-maintie --name=maintenance --quality-filter=mixed"),

        ("Create Universal RAG from MaintIE data",
         "python universal_rag_enhanced.py create-from-maintie --name=maintenance --quality-filter=high"),

        ("Create from external corpus",
         "python universal_rag_enhanced.py create-from-file --name=medical --corpus=medical_texts.txt"),

        ("Test created domain",
         "python universal_rag_enhanced.py test-domain --name=maintenance"),
    ]

    for description, command in examples:
        print(f"\n💻 {description}:")
        print(f"   {command}")


if __name__ == "__main__":
    demo_migration()
    demo_cli_usage()

    print(f"\n🎉 Demo complete! Ready to migrate to Universal RAG!")
    print(f"📚 Check the generated output files for detailed results.")