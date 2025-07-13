#!/usr/bin/env python3
"""
Simple script to extract knowledge from MaintIE data
Follows the practical approach outlined in Predetermined-Knowledge-Fixed.md
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.knowledge.simple_extraction import SimpleMaintIEExtractor, quick_equipment_extraction, run_full_extraction


def main():
    """Main extraction script"""
    print("🔍 MaintIE Knowledge Extraction Script")
    print("=" * 50)

    # Check if MaintIE data exists
    data_dir = Path("data/processed")
    if not data_dir.exists():
        print("❌ No processed data directory found")
        print("   Please run the data transformer first to process MaintIE data")
        return

    # Check for required files
    entities_file = data_dir / "maintenance_entities.json"
    documents_file = data_dir / "maintenance_documents.json"

    if not entities_file.exists() and not documents_file.exists():
        print("❌ No MaintIE processed data found")
        print("   Please ensure the following files exist:")
        print(f"   - {entities_file}")
        print(f"   - {documents_file}")
        return

    print("✅ Found MaintIE processed data")

    # Run quick equipment extraction
    print("\n1️⃣ Running quick equipment extraction...")
    quick_equipment_extraction()

    # Run full knowledge extraction
    print("\n2️⃣ Running full knowledge extraction...")
    run_full_extraction()

    # Show final stats
    print("\n3️⃣ Final extraction statistics:")
    extractor = SimpleMaintIEExtractor()
    stats = extractor.get_extraction_stats()

    if "error" in stats:
        print(f"❌ Error getting stats: {stats['error']}")
    else:
        print(f"   Equipment terms: {stats['equipment_terms']}")
        print(f"   Abbreviations: {stats['abbreviations']}")
        print(f"   Failure terms: {stats['failure_terms']}")
        print(f"   Procedure terms: {stats['procedure_terms']}")
        print(f"   Total extracted: {stats['total_extracted']}")

    print("\n✅ Knowledge extraction complete!")
    print("   Updated config/domain_knowledge.json with extracted knowledge")


if __name__ == "__main__":
    main()
