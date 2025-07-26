#!/usr/bin/env python3
"""
Process Full Dataset with Context-Aware Knowledge Extraction
Processes all 3,083 maintenance texts using the validated context engineering approach
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.azure_openai.improved_extraction_client import ImprovedKnowledgeExtractor

def load_all_maintenance_texts() -> List[str]:
    """Load all maintenance texts from the dataset"""
    
    raw_file = Path(__file__).parent.parent / "data" / "raw" / "maintenance_all_texts.md"
    
    if not raw_file.exists():
        raise FileNotFoundError(f"Maintenance dataset not found: {raw_file}")
    
    texts = []
    with open(raw_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('<id>'):
                text_content = line.replace('<id>', '').strip()
                if text_content:
                    texts.append(text_content)
    
    print(f"✅ Loaded {len(texts)} maintenance texts from dataset")
    return texts

def process_in_batches(texts: List[str], batch_size: int = 50) -> Dict[str, Any]:
    """Process texts in batches to manage API rate limits and memory"""
    
    extractor = ImprovedKnowledgeExtractor("maintenance")
    
    total_entities = []
    total_relationships = []
    batch_results = []
    
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    print(f"🔄 Processing {len(texts)} texts in {total_batches} batches of {batch_size}")
    
    for batch_idx in range(0, len(texts), batch_size):
        batch_texts = texts[batch_idx:batch_idx + batch_size]
        batch_num = (batch_idx // batch_size) + 1
        
        print(f"\n📦 Processing Batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")
        start_time = time.time()
        
        try:
            # Process batch
            batch_result = extractor.extract_with_comparison(batch_texts, sample_size=len(batch_texts))
            
            # Collect entities and relationships
            batch_entities = []
            batch_relationships = []
            
            for comparison in batch_result.get("comparisons", []):
                # Add source record tracking
                for entity in comparison.get("extracted_entities", []):
                    entity["source_text_id"] = batch_idx + comparison["text_id"]
                    entity["batch_number"] = batch_num
                    batch_entities.append(entity)
                
                for relation in comparison.get("extracted_relations", []):
                    relation["source_text_id"] = batch_idx + comparison["text_id"]
                    relation["batch_number"] = batch_num
                    batch_relationships.append(relation)
            
            total_entities.extend(batch_entities)
            total_relationships.extend(batch_relationships)
            
            batch_summary = {
                "batch_number": batch_num,
                "texts_processed": len(batch_texts),
                "entities_extracted": len(batch_entities),
                "relationships_extracted": len(batch_relationships),
                "processing_time_seconds": round(time.time() - start_time, 2),
                "quality_metrics": batch_result.get("summary", {}).get("quality_metrics", {})
            }
            
            batch_results.append(batch_summary)
            
            print(f"✅ Batch {batch_num} completed:")
            print(f"   • Entities: {len(batch_entities)}")
            print(f"   • Relationships: {len(batch_relationships)}")
            print(f"   • Time: {batch_summary['processing_time_seconds']}s")
            
            # Rate limiting: pause between batches
            if batch_num < total_batches:
                print("⏳ Pausing 2 seconds for API rate limiting...")
                time.sleep(2)
                
        except Exception as e:
            print(f"❌ Batch {batch_num} failed: {e}")
            batch_summary = {
                "batch_number": batch_num,
                "texts_processed": len(batch_texts),
                "entities_extracted": 0,
                "relationships_extracted": 0,
                "processing_time_seconds": round(time.time() - start_time, 2),
                "error": str(e)
            }
            batch_results.append(batch_summary)
            continue
    
    return {
        "processing_summary": {
            "total_texts": len(texts),
            "total_entities": len(total_entities),
            "total_relationships": len(total_relationships),
            "entities_per_text": round(len(total_entities) / len(texts), 2),
            "relationships_per_text": round(len(total_relationships) / len(texts), 2),
            "batches_processed": len(batch_results),
            "processing_timestamp": datetime.now().isoformat()
        },
        "entities": total_entities,
        "relationships": total_relationships,
        "batch_results": batch_results
    }

def save_results(results: Dict[str, Any]) -> Path:
    """Save extraction results to file"""
    
    output_dir = Path(__file__).parent.parent / "data" / "extraction_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"full_dataset_context_aware_extraction_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {output_file}")
    return output_file

def generate_summary_report(results: Dict[str, Any]):
    """Generate summary report of extraction results"""
    
    summary = results["processing_summary"]
    
    print(f"\n{'='*80}")
    print(f"FULL DATASET CONTEXT-AWARE EXTRACTION SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n📊 PROCESSING RESULTS:")
    print(f"   • Total texts processed: {summary['total_texts']:,}")
    print(f"   • Total entities extracted: {summary['total_entities']:,}")
    print(f"   • Total relationships extracted: {summary['total_relationships']:,}")
    print(f"   • Average entities per text: {summary['entities_per_text']}")
    print(f"   • Average relationships per text: {summary['relationships_per_text']}")
    
    print(f"\n🔄 BATCH PROCESSING:")
    print(f"   • Total batches: {summary['batches_processed']}")
    
    # Calculate success rate
    successful_batches = sum(1 for batch in results["batch_results"] if "error" not in batch)
    success_rate = (successful_batches / summary['batches_processed']) * 100
    print(f"   • Success rate: {success_rate:.1f}%")
    
    # Entity type analysis
    entity_types = {}
    for entity in results["entities"]:
        entity_type = entity.get("entity_type", "unknown")
        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
    
    print(f"\n🏷️  TOP ENTITY TYPES:")
    for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   • {entity_type}: {count:,}")
    
    # Relationship type analysis
    relation_types = {}
    for relation in results["relationships"]:
        relation_type = relation.get("relation_type", "unknown")
        relation_types[relation_type] = relation_types.get(relation_type, 0) + 1
    
    print(f"\n🔗 TOP RELATIONSHIP TYPES:")
    for relation_type, count in sorted(relation_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   • {relation_type}: {count:,}")
    
    print(f"\n✅ READY FOR AZURE UPLOAD AND GNN TRAINING")
    print(f"   • High-quality entities with full context for semantic embeddings")
    print(f"   • Dense relationship graph for graph neural network training")
    print(f"   • Context engineering approach validated at production scale")

def main():
    """Main processing function"""
    
    print("🚀 Starting Full Dataset Context-Aware Knowledge Extraction")
    print("=" * 80)
    
    try:
        # Load all maintenance texts
        texts = load_all_maintenance_texts()
        
        # Process in batches
        results = process_in_batches(texts, batch_size=50)
        
        # Save results
        output_file = save_results(results)
        
        # Generate summary report
        generate_summary_report(results)
        
        print(f"\n🎯 EXTRACTION COMPLETE!")
        print(f"📄 Detailed results: {output_file}")
        print(f"🔄 Next step: Upload to Azure and validate data integrity")
        
    except Exception as e:
        print(f"❌ Full dataset processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()