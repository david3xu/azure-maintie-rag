#!/usr/bin/env python3
"""
Monitor Full Dataset Extraction Progress
Real-time monitoring of the ongoing extraction process
"""

import json
import time
from pathlib import Path
from datetime import datetime

def monitor_progress():
    """Monitor current extraction progress"""
    
    progress_dir = Path(__file__).parent.parent / "data" / "extraction_progress"
    progress_file = progress_dir / "extraction_progress.json"
    entities_file = progress_dir / "entities_accumulator.jsonl"
    relationships_file = progress_dir / "relationships_accumulator.jsonl"
    
    if not progress_file.exists():
        print("❌ No extraction process found. Start with:")
        print("   python scripts/optimized_full_extraction.py")
        return
    
    # Load progress metadata
    with open(progress_file, 'r') as f:
        progress = json.load(f)
    
    # Count accumulated data
    entity_count = 0
    if entities_file.exists():
        with open(entities_file, 'r') as f:
            entity_count = sum(1 for line in f if line.strip())
    
    relationship_count = 0
    if relationships_file.exists():
        with open(relationships_file, 'r') as f:
            relationship_count = sum(1 for line in f if line.strip())
    
    # Calculate progress
    total_texts = 3083  # Known dataset size
    completed_texts = len(progress.get("completed_text_ids", []))
    progress_percent = (completed_texts / total_texts) * 100
    
    # Time analysis
    start_time = datetime.fromisoformat(progress["start_time"])
    current_time = datetime.now()
    elapsed_time = current_time - start_time
    
    # Estimate completion
    if completed_texts > 0:
        texts_per_minute = completed_texts / (elapsed_time.total_seconds() / 60)
        remaining_texts = total_texts - completed_texts
        estimated_minutes_remaining = remaining_texts / texts_per_minute if texts_per_minute > 0 else 0
    else:
        texts_per_minute = 0
        estimated_minutes_remaining = 0
    
    print("📊 EXTRACTION PROGRESS MONITOR")
    print("=" * 50)
    print(f"⏰ Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⌛ Elapsed: {str(elapsed_time).split('.')[0]}")
    print(f"📈 Progress: {completed_texts:,}/{total_texts:,} texts ({progress_percent:.1f}%)")
    print(f"🎯 Entities: {entity_count:,}")
    print(f"🔗 Relationships: {relationship_count:,}")
    print(f"📊 Last Batch: {progress.get('last_completed_batch', 0)}")
    print(f"⚡ Speed: {texts_per_minute:.1f} texts/minute")
    
    if estimated_minutes_remaining > 0:
        hours = int(estimated_minutes_remaining // 60)
        minutes = int(estimated_minutes_remaining % 60)
        print(f"⏱️  ETA: ~{hours}h {minutes}m remaining")
    
    # Recent batch summaries
    batch_summaries = progress.get("batch_summaries", [])
    if batch_summaries:
        print(f"\n📦 Recent Batches:")
        for batch in batch_summaries[-3:]:  # Last 3 batches
            success_rate = batch["successes"] / batch["texts_processed"] * 100
            print(f"   Batch {batch['batch_number']}: {batch['successes']}/{batch['texts_processed']} ({success_rate:.0f}%) - {batch['entities_extracted']} entities")
    
    # Quality metrics
    if completed_texts > 0:
        entities_per_text = entity_count / completed_texts
        relationships_per_text = relationship_count / completed_texts
        print(f"\n📈 Quality Metrics:")
        print(f"   • Entities per text: {entities_per_text:.1f}")
        print(f"   • Relationships per text: {relationships_per_text:.1f}")
    
    print(f"\n💾 Data Files:")
    print(f"   • Progress: {progress_file}")
    print(f"   • Entities: {entities_file} ({entity_count:,} records)")
    print(f"   • Relationships: {relationships_file} ({relationship_count:,} records)")
    
    # Status
    last_update = datetime.fromisoformat(progress["last_update"])
    minutes_since_update = (current_time - last_update).total_seconds() / 60
    
    if minutes_since_update < 5:
        status = "🟢 ACTIVE"
    elif minutes_since_update < 15:
        status = "🟡 POSSIBLY STALLED"
    else:
        status = "🔴 LIKELY STOPPED"
    
    print(f"\n🚦 Status: {status}")
    print(f"   Last update: {minutes_since_update:.1f} minutes ago")
    
    if status != "🟢 ACTIVE":
        print(f"\n💡 To resume extraction:")
        print(f"   python scripts/optimized_full_extraction.py")

def main():
    """Main monitoring function"""
    try:
        monitor_progress()
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")

if __name__ == "__main__":
    main()