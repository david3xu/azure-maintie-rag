#!/usr/bin/env python3
"""
Finalize Extraction Results
Consolidate and validate completed knowledge extraction
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

sys.path.append(str(Path(__file__).parent.parent))

def finalize_results():
    """Finalize and validate extraction results"""
    
    print("🏁 FINALIZING EXTRACTION RESULTS")
    print("=" * 50)
    
    # Check progress
    progress_dir = Path(__file__).parent.parent / "data" / "extraction_progress"
    progress_file = progress_dir / "extraction_progress.json"
    entities_file = progress_dir / "entities_accumulator.jsonl"
    relationships_file = progress_dir / "relationships_accumulator.jsonl"
    
    if not progress_file.exists():
        print("❌ No extraction progress found")
        return
    
    # Load progress metadata
    with open(progress_file, 'r') as f:
        progress = json.load(f)
    
    # Load all entities
    entities = []
    if entities_file.exists():
        with open(entities_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entities.append(json.loads(line))
    
    # Load all relationships  
    relationships = []
    if relationships_file.exists():
        with open(relationships_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    relationships.append(json.loads(line))
    
    # Validation
    total_texts = 3083
    completed_texts = len(progress.get("completed_text_ids", []))
    completion_rate = (completed_texts / total_texts) * 100
    
    print(f"📊 EXTRACTION SUMMARY:")
    print(f"   • Texts processed: {completed_texts:,}/{total_texts:,} ({completion_rate:.1f}%)")
    print(f"   • Entities extracted: {len(entities):,}")
    print(f"   • Relationships extracted: {len(relationships):,}")
    
    if completed_texts > 0:
        entities_per_text = len(entities) / completed_texts
        relationships_per_text = len(relationships) / completed_texts
        print(f"   • Quality: {entities_per_text:.1f} entities/text, {relationships_per_text:.1f} relationships/text")
    
    # Create final consolidated file
    final_results = {
        "extraction_metadata": {
            "approach": "context_aware_extraction",
            "completion_date": datetime.now().isoformat(),
            "total_texts_available": total_texts,
            "total_texts_processed": completed_texts,
            "completion_rate_percent": round(completion_rate, 2),
            "total_entities": len(entities),
            "total_relationships": len(relationships),
            "entities_per_text": round(len(entities) / max(completed_texts, 1), 2),
            "relationships_per_text": round(len(relationships) / max(completed_texts, 1), 2),
            "processing_summary": progress.get("batch_summaries", [])
        },
        "entities": entities,
        "relationships": relationships
    }
    
    # Save final results
    output_dir = Path(__file__).parent.parent / "data" / "extraction_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_file = output_dir / f"final_context_aware_extraction_{timestamp}.json"
    
    with open(final_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Final results saved: {final_file}")
    
    # Quality analysis
    print(f"\n📈 QUALITY ANALYSIS:")
    
    # Entity types
    entity_types = {}
    for entity in entities:
        entity_type = entity.get("entity_type", "unknown")
        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
    
    print(f"   Top Entity Types:")
    for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"     • {entity_type}: {count:,}")
    
    # Relationship types
    relation_types = {}
    for relation in relationships:
        relation_type = relation.get("relation_type", "unknown")
        relation_types[relation_type] = relation_types.get(relation_type, 0) + 1
    
    print(f"   Top Relationship Types:")
    for relation_type, count in sorted(relation_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"     • {relation_type}: {count:,}")
    
    # Context preservation check
    entities_with_context = sum(1 for e in entities if e.get("context", "").strip())
    context_rate = (entities_with_context / len(entities)) * 100 if entities else 0
    print(f"   Context Preservation: {context_rate:.1f}% ({entities_with_context:,}/{len(entities):,})")
    
    # Validation checks
    print(f"\n🔍 VALIDATION CHECKS:")
    
    # Check for duplicate entities
    entity_texts = [e.get("text", "") for e in entities]
    unique_entities = len(set(entity_texts))
    print(f"   • Entity uniqueness: {unique_entities:,}/{len(entities):,} unique")
    
    # Check relationship integrity
    entity_ids = {e.get("entity_id", "") for e in entities}
    valid_relations = 0
    for rel in relationships:
        source_id = rel.get("source_entity_id", "")
        target_id = rel.get("target_entity_id", "")
        if source_id in entity_ids and target_id in entity_ids:
            valid_relations += 1
    
    relation_integrity = (valid_relations / len(relationships)) * 100 if relationships else 0
    print(f"   • Relationship integrity: {relation_integrity:.1f}% ({valid_relations:,}/{len(relationships):,})")
    
    # Overall assessment
    if completion_rate >= 95 and context_rate >= 95 and relation_integrity >= 90:
        status = "✅ EXCELLENT - Ready for Azure upload and GNN training"
    elif completion_rate >= 80 and context_rate >= 80 and relation_integrity >= 80:
        status = "✅ GOOD - Ready for Azure upload with minor issues"
    else:
        status = "⚠️  NEEDS REVIEW - Check quality metrics"
    
    print(f"\n🎯 OVERALL ASSESSMENT: {status}")
    
    # Next steps
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Upload to Azure Cosmos DB:")
    print(f"      python scripts/upload_knowledge_to_azure.py")
    print(f"   2. Validate Azure data integrity:")
    print(f"      python scripts/validate_azure_knowledge_data.py")
    print(f"   3. Prepare GNN training features:")
    print(f"      python scripts/prepare_gnn_training_features.py")
    
    return final_file

def main():
    """Main finalization process"""
    try:
        final_file = finalize_results()
        print(f"\n🏆 Extraction finalization completed!")
        print(f"📄 Results: {final_file}")
    except Exception as e:
        print(f"❌ Finalization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()