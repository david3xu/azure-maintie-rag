#!/usr/bin/env python3
"""
Test Optimized Extraction with Small Sample
Verify the optimized approach works before running on full dataset
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.optimized_full_extraction import OptimizedFullExtractor

def main():
    """Test with small sample"""
    
    print("🧪 Testing Optimized Extraction Approach")
    print("=" * 50)
    
    # Test with small sample
    sample_texts = [
        "air conditioner thermostat not working",
        "bearing on air conditioner compressor unserviceable", 
        "blown o-ring off steering hose",
        "brake system pressure low",
        "coolant temperature sensor malfunction",
        "diesel engine fuel filter clogged",
        "hydraulic pump pressure relief valve stuck",
        "transmission fluid leak at left hand side"
    ]
    
    print(f"📝 Testing with {len(sample_texts)} sample texts")
    
    try:
        # Initialize with small batches for testing
        extractor = OptimizedFullExtractor(batch_size=3, save_every=2)
        
        # Process sample
        progress = extractor.process_with_realtime_saving(sample_texts)
        
        # Finalize
        output_file = extractor.finalize_results(progress)
        
        print(f"\n✅ TEST COMPLETED SUCCESSFULLY!")
        print(f"📊 Results: {progress['total_entities']} entities, {progress['total_relationships']} relationships")
        print(f"📄 Output: {output_file}")
        
        # Clean up test files
        extractor.cleanup_progress_files()
        
        print(f"\n🎯 Optimized extraction approach validated!")
        print(f"🚀 Ready to run on full dataset: python scripts/optimized_full_extraction.py")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()