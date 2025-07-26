#!/usr/bin/env python3
"""
Full Dataset Knowledge Extraction Script
Processes ALL texts in the dataset (not just a sample)
Extracts entities and relationships from the complete dataset
"""

import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.azure_openai.improved_extraction_client import ImprovedKnowledgeExtractor
from core.workflow.progress_tracker import create_progress_tracker, track_async_operation, track_sync_operation


def load_all_texts(file_path: str) -> List[str]:
    """Load ALL texts from the file, not just a sample"""

    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return []

    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and line.startswith('<id>'):
                # Extract the actual text content after <id>
                text_content = line.replace('<id>', '').strip()
                if text_content:
                    texts.append(text_content)

    print(f"✅ Loaded {len(texts)} texts from: {file_path}")
    return texts


async def extract_full_dataset_knowledge(texts: List[str], domain: str = "maintenance", batch_size: int = 50) -> Dict[str, Any]:
    """
    Extract knowledge from the FULL dataset in batches
    """

    # Initialize real-time progress tracker
    progress_tracker = create_progress_tracker("Full Dataset Knowledge Extraction")
    progress_tracker.start_workflow()

    print(f"\n🔍 Starting FULL dataset knowledge extraction...")
    print(f"   • Domain: {domain}")
    print(f"   • Total texts to process: {len(texts)}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Estimated batches: {len(texts) // batch_size + 1}")

    # Initialize the extractor
    extractor = ImprovedKnowledgeExtractor(domain)

    all_entities = []
    all_relationships = []
    processed_count = 0

    # Start knowledge extraction step
    progress_tracker.start_step("Knowledge Extraction", {
        "total_texts": len(texts),
        "batch_size": batch_size,
        "total_batches": (len(texts) + batch_size - 1) // batch_size
    })

    # Process in batches to avoid memory issues
    for batch_start in range(0, len(texts), batch_size):
        batch_end = min(batch_start + batch_size, len(texts))
        batch_texts = texts[batch_start:batch_end]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(texts) + batch_size - 1) // batch_size

        progress_tracker.update_step_progress("Knowledge Extraction", {
            "current_batch": batch_num,
            "total_batches": total_batches,
            "batch_texts": len(batch_texts),
            "processed_count": processed_count,
            "entities_found": len(all_entities),
            "relations_found": len(all_relationships)
        })

        print(f"\n   📦 Processing batch {batch_num}/{total_batches} (texts {batch_start+1}-{batch_end})")

        batch_entities = []
        batch_relationships = []

        for i, text in enumerate(batch_texts, 1):
            try:
                if i % 10 == 0:  # Progress indicator every 10 texts
                    print(f"      Processing text {i}/{len(batch_texts)}: {text[:50]}...")

                # Extract entities and relationships
                entities = extractor._extract_entities_with_context(text)
                relationships = extractor._extract_relations_with_linking(text, entities)

                # Add batch and text metadata
                for entity in entities:
                    entity['batch_id'] = batch_num
                    entity['text_id'] = batch_start + i
                    entity['source_text'] = text[:100] + "..." if len(text) > 100 else text

                for relation in relationships:
                    relation['batch_id'] = batch_num
                    relation['text_id'] = batch_start + i
                    relation['source_text'] = text[:100] + "..." if len(text) > 100 else text

                batch_entities.extend(entities)
                batch_relationships.extend(relationships)
                processed_count += 1

                # Update progress for individual text processing
                progress_tracker.update_step_progress("Knowledge Extraction", {
                    "current_batch": batch_num,
                    "total_batches": total_batches,
                    "batch_texts": len(batch_texts),
                    "processed_count": processed_count,
                    "entities_found": len(all_entities) + len(batch_entities),
                    "relations_found": len(all_relationships) + len(batch_relationships),
                    "current_text": i,
                    "total_texts_in_batch": len(batch_texts)
                })

            except Exception as e:
                print(f"      ⚠️  Error processing text {batch_start + i}: {e}")
                continue

        all_entities.extend(batch_entities)
        all_relationships.extend(batch_relationships)

        print(f"      ✅ Batch {batch_num} complete: {len(batch_entities)} entities, {len(batch_relationships)} relationships")

    progress_tracker.complete_step("Knowledge Extraction", success=True)

    # Transform to comprehensive format
    full_results = {
        "entities": all_entities,
        "relationships": all_relationships,
        "summary": {
            "total_texts": len(texts),
            "total_processed": processed_count,
            "total_entities": len(all_entities),
            "total_relationships": len(all_relationships),
            "avg_entities_per_text": round(len(all_entities) / processed_count, 2) if processed_count > 0 else 0,
            "avg_relationships_per_text": round(len(all_relationships) / processed_count, 2) if processed_count > 0 else 0,
            "extraction_method": "full_dataset_batch_processing",
            "batch_size": batch_size,
            "processing_notes": f"Processed {processed_count}/{len(texts)} texts in batches"
        }
    }

    print(f"\n✅ FULL dataset extraction completed!")
    print(f"   • Total texts processed: {processed_count}/{len(texts)}")
    print(f"   • Total entities: {len(full_results['entities'])}")
    print(f"   • Total relationships: {len(full_results['relationships'])}")
    print(f"   • Average entities per text: {full_results['summary']['avg_entities_per_text']}")
    print(f"   • Average relationships per text: {full_results['summary']['avg_relationships_per_text']}")

    progress_tracker.finish_workflow(success=True)

    return full_results


def display_full_results(results: Dict[str, Any], show_details: bool = True):
    """Display the full dataset extraction results"""

    print(f"\n{'='*80}")
    print("📊 FULL DATASET KNOWLEDGE EXTRACTION RESULTS")
    print(f"{'='*80}")

    summary = results['summary']
    entities = results['entities']
    relationships = results['relationships']

    print(f"\n📈 Summary Statistics:")
    print(f"   • Total texts in dataset: {summary['total_texts']}")
    print(f"   • Texts successfully processed: {summary['total_processed']}")
    print(f"   • Entities extracted: {summary['total_entities']}")
    print(f"   • Relationships extracted: {summary['total_relationships']}")
    print(f"   • Avg entities per text: {summary['avg_entities_per_text']}")
    print(f"   • Avg relationships per text: {summary['avg_relationships_per_text']}")
    print(f"   • Extraction method: {summary['extraction_method']}")
    print(f"   • Batch size used: {summary['batch_size']}")
    print(f"   • Processing notes: {summary['processing_notes']}")

    if show_details and entities:
        print(f"\n🔍 Sample Entities (first 5):")
        for i, entity in enumerate(entities[:5], 1):
            print(f"   {i}. \"{entity.get('text', entity.get('name', 'unknown'))}\" [{entity.get('entity_type', 'unknown')}]")
            print(f"      Context: \"{entity.get('context', 'N/A')}\"")
            print(f"      Source: Text {entity.get('text_id', 'N/A')} (Batch {entity.get('batch_id', 'N/A')})")

        print(f"\n🔗 Sample Relationships (first 5):")
        for i, relation in enumerate(relationships[:5], 1):
            source = relation.get('source_entity', relation.get('source_entity_id', 'unknown'))
            target = relation.get('target_entity', relation.get('target_entity_id', 'unknown'))
            print(f"   {i}. {source} --[{relation.get('relation_type', 'unknown')}]--> {target}")
            print(f"      Confidence: {relation.get('confidence', 'N/A')}")
            print(f"      Source: Text {relation.get('text_id', 'N/A')} (Batch {relation.get('batch_id', 'N/A')})")

        if len(entities) > 5:
            print(f"\n   ... and {len(entities) - 5} more entities")
        if len(relationships) > 5:
            print(f"   ... and {len(relationships) - 5} more relationships")

    elif not entities:
        print(f"\n⚠️  No entities extracted - this may indicate an issue with the extraction process")


def save_full_results(results: Dict[str, Any], output_file: str = None) -> str:
    """Save full results to JSON file"""

    if not output_file:
        output_file = f"full_dataset_extraction_{len(results['entities'])}_entities_{len(results['relationships'])}_relationships.json"

    output_path = Path(__file__).parent.parent / "data" / "extraction_outputs" / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Full dataset results saved to: {output_path}")
    return str(output_path)


async def main():
    """Main function for full dataset knowledge extraction"""

    print("🌐 FULL DATASET KNOWLEDGE EXTRACTION")
    print("=" * 60)
    print("Processes ALL texts in the dataset (not just a sample)")
    print("Uses batch processing for memory efficiency")

    # Load arguments
    import argparse
    parser = argparse.ArgumentParser(description="Full dataset knowledge extraction")
    parser.add_argument("--input", "-i", default="data/raw/maintenance_all_texts.md", help="Input file with raw texts")
    parser.add_argument("--domain", "-d", default="maintenance", help="Domain for extraction")
    parser.add_argument("--output", "-o", help="Output file for results")
    parser.add_argument("--batch-size", "-b", type=int, default=50, help="Batch size for processing")
    parser.add_argument("--no-details", action="store_true", help="Don't show detailed results")

    args = parser.parse_args()

    try:
        # Load ALL texts
        texts = load_all_texts(args.input)

        if not texts:
            print("❌ No texts to process!")
            return

        # Extract knowledge from full dataset
        results = await extract_full_dataset_knowledge(texts, args.domain, args.batch_size)

        # Display results
        display_full_results(results, not args.no_details)

        # Save results
        output_path = save_full_results(results, args.output)

        print(f"\n{'='*80}")
        print("🎯 FULL DATASET EXTRACTION COMPLETE")
        print(f"{'='*80}")
        print(f"\n📁 Results saved to: {output_path}")
        print(f"📊 Entities: {len(results['entities'])}")
        print(f"🔗 Relationships: {len(results['relationships'])}")
        print(f"\n💡 Full Dataset Features:")
        print(f"   ✅ Processes ALL {len(texts)} texts (not just a sample)")
        print(f"   ✅ Uses batch processing ({args.batch_size} texts per batch)")
        print(f"   ✅ Memory efficient processing")
        print(f"   ✅ Complete knowledge extraction")
        print(f"\n🚀 Ready for comprehensive knowledge graph construction!")

    except Exception as e:
        print(f"❌ Full dataset extraction failed: {e}")
        return 1


if __name__ == "__main__":
    asyncio.run(main())
