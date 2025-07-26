#!/usr/bin/env python3
"""
Continue/Resume Extraction Process
Designed for long-running extraction with better progress tracking
"""

import sys
import json
import time
import signal
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.optimized_full_extraction import OptimizedFullExtractor

class ContinuousExtractor:
    """Enhanced extractor with better progress tracking and graceful shutdown"""
    
    def __init__(self):
        self.should_stop = False
        self.extractor = OptimizedFullExtractor(batch_size=5, save_every=3)  # Smaller batches
        
        # Set up graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n🛑 Shutdown signal received. Saving progress...")
        self.should_stop = True
    
    def run_with_progress_updates(self):
        """Run extraction with enhanced progress tracking"""
        
        print("🚀 CONTINUOUS EXTRACTION WITH PROGRESS TRACKING")
        print("=" * 60)
        print("Features:")
        print("  ✅ Graceful shutdown (Ctrl+C)")
        print("  ✅ Enhanced progress tracking")
        print("  ✅ Real-time status updates")
        print("  ✅ Automatic resume capability")
        print("=" * 60)
        
        try:
            # Load texts
            texts = self.extractor.load_all_texts()
            total_texts = len(texts)
            
            # Load existing progress
            progress = self.extractor.load_progress()
            start_batch = progress["last_completed_batch"]
            
            # Calculate remaining work
            completed_texts = len(progress.get("completed_text_ids", []))
            remaining_texts = total_texts - completed_texts
            
            print(f"📊 Dataset Status:")
            print(f"   • Total texts: {total_texts:,}")
            print(f"   • Completed: {completed_texts:,}")
            print(f"   • Remaining: {remaining_texts:,}")
            print(f"   • Resuming from batch: {start_batch + 1}")
            
            if remaining_texts == 0:
                print("✅ All texts already processed!")
                return self.extractor.finalize_results(progress)
            
            # Process remaining texts
            batch_size = self.extractor.batch_size
            total_batches = (total_texts + batch_size - 1) // batch_size
            
            for batch_idx in range(start_batch * batch_size, total_texts, batch_size):
                if self.should_stop:
                    print("🛑 Graceful shutdown requested")
                    break
                
                batch_num = (batch_idx // batch_size) + 1
                batch_texts = texts[batch_idx:batch_idx + batch_size]
                
                print(f"\n🔄 Processing Batch {batch_num}/{total_batches}")
                print(f"   📝 Texts in batch: {len(batch_texts)}")
                
                batch_start_time = time.time()
                batch_entities = []
                batch_relationships = []
                batch_successes = 0
                
                # Process each text
                for i, text in enumerate(batch_texts):
                    if self.should_stop:
                        break
                        
                    text_id = batch_idx + i
                    print(f"      {i+1}/{len(batch_texts)}: Processing text {text_id}")
                    
                    result = self.extractor.process_single_text(text, text_id)
                    
                    if result["success"]:
                        batch_entities.extend(result["entities"])
                        batch_relationships.extend(result["relationships"])
                        batch_successes += 1
                        progress["completed_text_ids"].append(text_id)
                        print(f"         ✅ {len(result['entities'])} entities, {len(result['relationships'])} relationships")
                    else:
                        print(f"         ❌ Failed: {result.get('error', 'Unknown')}")
                    
                    # Small delay for API rate limiting
                    time.sleep(1.0)
                
                # Save batch results
                if batch_entities:
                    self.extractor.append_entities(batch_entities)
                    progress["total_entities"] += len(batch_entities)
                
                if batch_relationships:
                    self.extractor.append_relationships(batch_relationships)
                    progress["total_relationships"] += len(batch_relationships)
                
                # Update progress
                batch_time = time.time() - batch_start_time
                batch_summary = {
                    "batch_number": batch_num,
                    "texts_processed": len(batch_texts),
                    "successes": batch_successes,
                    "failures": len(batch_texts) - batch_successes,
                    "entities_extracted": len(batch_entities),
                    "relationships_extracted": len(batch_relationships),
                    "processing_time_seconds": round(batch_time, 2),
                    "timestamp": datetime.now().isoformat()
                }
                
                progress["batch_summaries"].append(batch_summary)
                progress["last_completed_batch"] = batch_num
                
                # Save progress
                self.extractor.save_progress(progress)
                
                # Progress report
                completed_now = len(progress["completed_text_ids"])
                overall_progress = (completed_now / total_texts) * 100
                
                print(f"   ✅ Batch {batch_num} completed:")
                print(f"      • Success rate: {batch_successes}/{len(batch_texts)}")
                print(f"      • Entities: {len(batch_entities)}")
                print(f"      • Relationships: {len(batch_relationships)}")
                print(f"      • Overall progress: {completed_now:,}/{total_texts:,} ({overall_progress:.1f}%)")
                print(f"      • Cumulative: {progress['total_entities']} entities, {progress['total_relationships']} relationships")
                
                # Rate limiting between batches
                if batch_num < total_batches and not self.should_stop:
                    print(f"   ⏳ Pausing 5 seconds...")
                    time.sleep(5)
            
            # Finalize if completed
            if not self.should_stop:
                print(f"\n🎉 EXTRACTION COMPLETED!")
                return self.extractor.finalize_results(progress)
            else:
                print(f"\n💾 Progress saved. Resume with same command.")
                return None
                
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main execution"""
    extractor = ContinuousExtractor()
    result = extractor.run_with_progress_updates()
    
    if result:
        print(f"🎯 Final results available!")
        print(f"📄 Use monitor script to check: python scripts/monitor_extraction_progress.py")

if __name__ == "__main__":
    main()