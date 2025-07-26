#!/usr/bin/env python3
"""
Optimized Full Dataset Extraction with Real-time Saving
- Real-time progress saving to prevent data loss
- Resume capability from last saved state
- Smaller batch sizes and better error handling
- Azure rate limit compliance
"""

import sys
import json
import time
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.azure_openai.improved_extraction_client import ImprovedKnowledgeExtractor

class OptimizedFullExtractor:
    """Optimized extractor with real-time saving and resume capability"""
    
    def __init__(self, batch_size: int = 10, save_every: int = 5):
        self.batch_size = batch_size  # Smaller batches for stability
        self.save_every = save_every  # Save progress every N batches
        self.extractor = ImprovedKnowledgeExtractor("maintenance")
        
        # Progress tracking
        self.progress_dir = Path(__file__).parent.parent / "data" / "extraction_progress"
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        
        self.progress_file = self.progress_dir / "extraction_progress.json"
        self.entities_file = self.progress_dir / "entities_accumulator.jsonl"
        self.relationships_file = self.progress_dir / "relationships_accumulator.jsonl"
        
        print(f"🔧 Optimizer initialized: batch_size={batch_size}, save_every={save_every}")

    def load_progress(self) -> Dict[str, Any]:
        """Load previous progress or create new"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
            print(f"📄 Resuming from batch {progress.get('last_completed_batch', 0) + 1}")
            return progress
        else:
            progress = {
                "start_time": datetime.now().isoformat(),
                "last_completed_batch": 0,
                "total_entities": 0,
                "total_relationships": 0,
                "completed_text_ids": [],
                "failed_batches": [],
                "batch_summaries": []
            }
            self.save_progress(progress)
            print("🆕 Starting new extraction session")
            return progress

    def save_progress(self, progress: Dict[str, Any]):
        """Save current progress"""
        progress["last_update"] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

    def append_entities(self, entities: List[Dict[str, Any]]):
        """Append entities to JSONL file for real-time saving"""
        with open(self.entities_file, 'a', encoding='utf-8') as f:
            for entity in entities:
                f.write(json.dumps(entity, ensure_ascii=False) + '\n')

    def append_relationships(self, relationships: List[Dict[str, Any]]):
        """Append relationships to JSONL file for real-time saving"""
        with open(self.relationships_file, 'a', encoding='utf-8') as f:
            for relationship in relationships:
                f.write(json.dumps(relationship, ensure_ascii=False) + '\n')

    def load_all_texts(self) -> List[str]:
        """Load all maintenance texts"""
        raw_file = Path(__file__).parent.parent / "data" / "raw" / "maintenance_all_texts.md"
        
        if not raw_file.exists():
            raise FileNotFoundError(f"Dataset not found: {raw_file}")
        
        texts = []
        with open(raw_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('<id>'):
                    text_content = line.replace('<id>', '').strip()
                    if text_content:
                        texts.append(text_content)
        
        print(f"📚 Loaded {len(texts)} maintenance texts")
        return texts

    def process_single_text(self, text: str, text_id: int) -> Dict[str, Any]:
        """Process a single text with error handling"""
        try:
            start_time = time.time()
            
            # Extract from single text
            extraction_result = self.extractor._extract_from_single_text(text)
            
            # Add metadata
            for entity in extraction_result.get("entities", []):
                entity["global_text_id"] = text_id
                entity["extraction_timestamp"] = datetime.now().isoformat()
            
            for relation in extraction_result.get("relations", []):
                relation["global_text_id"] = text_id
                relation["extraction_timestamp"] = datetime.now().isoformat()
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "text_id": text_id,
                "text": text,
                "entities": extraction_result.get("entities", []),
                "relationships": extraction_result.get("relations", []),
                "processing_time": round(processing_time, 2)
            }
            
        except Exception as e:
            print(f"❌ Failed processing text {text_id}: {e}")
            return {
                "success": False,
                "text_id": text_id,
                "text": text,
                "error": str(e),
                "entities": [],
                "relationships": []
            }

    def process_with_realtime_saving(self, texts: List[str]) -> Dict[str, Any]:
        """Process all texts with real-time progress saving"""
        
        progress = self.load_progress()
        start_batch = progress["last_completed_batch"]
        
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        print(f"🚀 Processing {len(texts)} texts in {total_batches} batches")
        print(f"📊 Starting from batch {start_batch + 1}/{total_batches}")
        
        for batch_idx in range(start_batch * self.batch_size, len(texts), self.batch_size):
            batch_num = (batch_idx // self.batch_size) + 1
            batch_texts = texts[batch_idx:batch_idx + self.batch_size]
            
            print(f"\n🔄 Processing Batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")
            batch_start_time = time.time()
            
            batch_entities = []
            batch_relationships = []
            batch_successes = 0
            batch_failures = 0
            
            # Process each text in the batch
            for i, text in enumerate(batch_texts):
                text_id = batch_idx + i
                
                print(f"   📝 Text {i+1}/{len(batch_texts)}: {text[:50]}{'...' if len(text) > 50 else ''}")
                
                result = self.process_single_text(text, text_id)
                
                if result["success"]:
                    batch_entities.extend(result["entities"])
                    batch_relationships.extend(result["relationships"])
                    batch_successes += 1
                    progress["completed_text_ids"].append(text_id)
                    
                    print(f"      ✅ {len(result['entities'])} entities, {len(result['relationships'])} relationships")
                else:
                    batch_failures += 1
                    print(f"      ❌ Failed: {result.get('error', 'Unknown error')}")
                
                # Small delay between texts for API rate limiting
                time.sleep(0.5)
            
            # Real-time save of extracted data
            if batch_entities:
                self.append_entities(batch_entities)
                progress["total_entities"] += len(batch_entities)
            
            if batch_relationships:
                self.append_relationships(batch_relationships)
                progress["total_relationships"] += len(batch_relationships)
            
            # Update progress
            batch_time = time.time() - batch_start_time
            batch_summary = {
                "batch_number": batch_num,
                "texts_processed": len(batch_texts),
                "successes": batch_successes,
                "failures": batch_failures,
                "entities_extracted": len(batch_entities),
                "relationships_extracted": len(batch_relationships),
                "processing_time_seconds": round(batch_time, 2)
            }
            
            progress["batch_summaries"].append(batch_summary)
            progress["last_completed_batch"] = batch_num
            
            print(f"   ✅ Batch {batch_num} completed: {batch_successes}/{len(batch_texts)} texts successful")
            print(f"   📊 Cumulative: {progress['total_entities']} entities, {progress['total_relationships']} relationships")
            
            # Save progress every N batches
            if batch_num % self.save_every == 0 or batch_num == total_batches:
                self.save_progress(progress)
                print(f"   💾 Progress saved at batch {batch_num}")
            
            # Rate limiting between batches
            if batch_num < total_batches:
                print(f"   ⏳ Pausing 3 seconds for API rate limiting...")
                time.sleep(3)
        
        return progress

    def finalize_results(self, progress: Dict[str, Any]) -> Path:
        """Create final consolidated results file"""
        
        # Load all entities and relationships from JSONL files
        entities = []
        relationships = []
        
        if self.entities_file.exists():
            with open(self.entities_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entities.append(json.loads(line))
        
        if self.relationships_file.exists():
            with open(self.relationships_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        relationships.append(json.loads(line))
        
        # Create final results
        final_results = {
            "extraction_metadata": {
                "approach": "context_aware_extraction",
                "start_time": progress.get("start_time"),
                "completion_time": datetime.now().isoformat(),
                "total_texts_processed": len(progress.get("completed_text_ids", [])),
                "total_entities": len(entities),
                "total_relationships": len(relationships),
                "entities_per_text": round(len(entities) / max(len(progress.get("completed_text_ids", [])), 1), 2),
                "relationships_per_text": round(len(relationships) / max(len(progress.get("completed_text_ids", [])), 1), 2),
                "batch_processing_summary": progress.get("batch_summaries", [])
            },
            "entities": entities,
            "relationships": relationships,
            "processing_progress": progress
        }
        
        # Save final results
        output_dir = Path(__file__).parent.parent / "data" / "extraction_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"optimized_full_extraction_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Final results saved: {output_file}")
        return output_file

    def cleanup_progress_files(self):
        """Clean up progress files after successful completion"""
        try:
            if self.progress_file.exists():
                self.progress_file.unlink()
            if self.entities_file.exists():
                self.entities_file.unlink()
            if self.relationships_file.exists():
                self.relationships_file.unlink()
            print("🧹 Progress files cleaned up")
        except Exception as e:
            print(f"⚠️  Could not clean up progress files: {e}")

def main():
    """Main execution with optimized processing"""
    
    print("🚀 OPTIMIZED FULL DATASET EXTRACTION")
    print("=" * 60)
    print("Features:")
    print("  ✅ Real-time progress saving")
    print("  ✅ Resume from last saved state")
    print("  ✅ Small batch sizes for stability")
    print("  ✅ Individual text error handling")
    print("  ✅ Azure API rate limit compliance")
    print("=" * 60)
    
    try:
        # Initialize optimizer
        extractor = OptimizedFullExtractor(batch_size=10, save_every=5)
        
        # Load texts
        texts = extractor.load_all_texts()
        
        # Process with real-time saving
        progress = extractor.process_with_realtime_saving(texts)
        
        # Finalize results
        output_file = extractor.finalize_results(progress)
        
        # Show summary
        print(f"\n{'='*60}")
        print(f"EXTRACTION COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"📊 Final Results:")
        print(f"   • Texts processed: {len(progress.get('completed_text_ids', []))}")
        print(f"   • Total entities: {progress['total_entities']}")
        print(f"   • Total relationships: {progress['total_relationships']}")
        print(f"   • Success rate: {len(progress.get('completed_text_ids', []))/len(texts)*100:.1f}%")
        print(f"📄 Results file: {output_file}")
        
        # Clean up progress files
        extractor.cleanup_progress_files()
        
        print(f"\n🎯 READY FOR AZURE UPLOAD AND GNN TRAINING!")
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\n💡 Progress has been saved. You can resume by running this script again.")

if __name__ == "__main__":
    main()