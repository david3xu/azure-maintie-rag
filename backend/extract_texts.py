#!/usr/bin/env python3
"""
Simple Text Extraction Runner
Extracts pure text data from MaintIE files for Universal RAG processing
"""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / "scripts"))

from extract_text_data import TextDataExtractor

def main():
    """Extract text data with default settings"""

    print("🚀 Extracting text data from MaintIE files...")
    print("=" * 50)

    try:
        extractor = TextDataExtractor()
        metadata = extractor.extract_all_text_data(
            min_length=30,
            create_combined=True,
            create_separate=True
        )

        print("\n🎯 Quick Start Commands:")
        print("# Create Universal RAG from high-quality texts:")
        print("python universal_rag_enhanced.py create-from-file \\")
        print("  --name=maintenance \\")
        print("  --corpus=data/raw/maintenance_high_quality_texts.txt")

        print("\n# Or use all texts:")
        print("python universal_rag_enhanced.py create-from-file \\")
        print("  --name=maintenance \\")
        print("  --corpus=data/raw/maintenance_all_texts.txt")

        return metadata

    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        print("\n💡 Make sure you have MaintIE data files:")
        print("   - data/raw/gold_release.json")
        print("   - data/raw/silver_release.json")
        sys.exit(1)

if __name__ == "__main__":
    main()