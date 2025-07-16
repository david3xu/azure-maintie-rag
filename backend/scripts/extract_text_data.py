#!/usr/bin/env python3
"""
Text Data Extraction Script
Extracts pure text data from MaintIE gold and silver annotation files
Creates clean text files for Universal RAG processing
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextDataExtractor:
    """Extract clean text data from MaintIE annotation files"""

    def __init__(self, raw_data_dir: Optional[Path] = None, output_dir: Optional[Path] = None):
        """Initialize extractor with data directories"""
        self.raw_data_dir = raw_data_dir or settings.raw_data_dir
        self.output_dir = output_dir or self.raw_data_dir

        # Input files
        self.gold_file = self.raw_data_dir / settings.gold_data_filename
        self.silver_file = self.raw_data_dir / settings.silver_data_filename

        # Output files
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"TextDataExtractor initialized")
        logger.info(f"Input dir: {self.raw_data_dir}")
        logger.info(f"Output dir: {self.output_dir}")

    def extract_all_text_data(self, min_length: int = 30,
                             create_combined: bool = True,
                             create_separate: bool = True) -> Dict[str, Any]:
        """Extract all text data in different formats"""

        logger.info("🚀 Starting text data extraction...")
        start_time = datetime.now()

        # Extract from gold and silver
        gold_texts, gold_stats = self._extract_texts_from_file(self.gold_file, "gold", min_length)
        silver_texts, silver_stats = self._extract_texts_from_file(self.silver_file, "silver", min_length)

        # Create output files
        outputs = {}

        if create_separate:
            # Save separate files
            gold_path = self._save_texts_to_file(gold_texts, "maintenance_gold_texts.txt", "Gold quality maintenance texts")
            silver_path = self._save_texts_to_file(silver_texts, "maintenance_silver_texts.txt", "Silver quality maintenance texts")
            outputs["gold_file"] = str(gold_path)
            outputs["silver_file"] = str(silver_path)

        if create_combined:
            # Save combined file
            combined_texts = gold_texts + silver_texts
            combined_path = self._save_texts_to_file(combined_texts, "maintenance_all_texts.txt", "Combined maintenance texts (gold + silver)")
            outputs["combined_file"] = str(combined_path)

        # Create high-quality subset (gold only)
        high_quality_path = self._save_texts_to_file(gold_texts, "maintenance_high_quality_texts.txt", "High quality maintenance texts (gold only)")
        outputs["high_quality_file"] = str(high_quality_path)

        # Generate metadata
        metadata = {
            "extraction_date": start_time.isoformat(),
            "source_files": {
                "gold": str(self.gold_file),
                "silver": str(self.silver_file)
            },
            "statistics": {
                "gold": gold_stats,
                "silver": silver_stats,
                "combined": {
                    "total_texts": len(gold_texts) + len(silver_texts),
                    "total_characters": sum(len(t) for t in gold_texts + silver_texts),
                    "avg_length": (sum(len(t) for t in gold_texts + silver_texts) /
                                 (len(gold_texts) + len(silver_texts))) if (gold_texts + silver_texts) else 0
                }
            },
            "output_files": outputs,
            "processing_settings": {
                "min_text_length": min_length,
                "create_combined": create_combined,
                "create_separate": create_separate
            }
        }

        # Save metadata
        metadata_path = self.output_dir / "text_extraction_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        elapsed_time = datetime.now() - start_time

        # Print summary
        self._print_extraction_summary(metadata, elapsed_time)

        return metadata

    def _extract_texts_from_file(self, file_path: Path, source_name: str,
                                min_length: int) -> tuple[List[str], Dict[str, Any]]:
        """Extract texts from a single MaintIE file"""

        logger.info(f"📖 Extracting texts from {source_name} file: {file_path}")

        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return [], {"error": "file_not_found", "total_documents": 0, "texts_extracted": 0}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return [], {"error": str(e), "total_documents": 0, "texts_extracted": 0}

        texts = []
        skipped = 0
        char_counts = []

        for item in data:
            # Extract text field
            text = item.get("text", "")

            if not text or not isinstance(text, str):
                skipped += 1
                continue

            # Clean text
            clean_text = self._clean_text(text)

            # Check minimum length
            if len(clean_text) < min_length:
                skipped += 1
                continue

            texts.append(clean_text)
            char_counts.append(len(clean_text))

        stats = {
            "total_documents": len(data),
            "texts_extracted": len(texts),
            "texts_skipped": skipped,
            "total_characters": sum(char_counts),
            "avg_length": sum(char_counts) / len(char_counts) if char_counts else 0,
            "min_length": min(char_counts) if char_counts else 0,
            "max_length": max(char_counts) if char_counts else 0
        }

        logger.info(f"✅ {source_name}: extracted {len(texts)} texts, skipped {skipped}")
        return texts, stats

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""

        # Basic cleaning
        clean_text = text.strip()

        # Remove excessive whitespace
        clean_text = " ".join(clean_text.split())

        # Remove control characters but keep newlines
        clean_text = "".join(char for char in clean_text if ord(char) >= 32 or char in ['\n', '\t'])

        return clean_text

    def _save_texts_to_file(self, texts: List[str], filename: str, description: str) -> Path:
        """Save texts to file with metadata header"""

        output_path = self.output_dir / filename

        logger.info(f"💾 Saving {len(texts)} texts to {filename}")

        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write(f"# {description}\n")
            f.write(f"# Extracted on: {datetime.now().isoformat()}\n")
            f.write(f"# Total texts: {len(texts)}\n")
            f.write(f"# Source: MaintIE dataset\n")
            f.write("#" + "="*60 + "\n\n")

            # Write texts (one per line, separated by double newline)
            for i, text in enumerate(texts):
                # Escape internal newlines for single-line format
                escaped_text = text.replace('\n', '\\n').replace('\r', '\\r')
                f.write(f"{escaped_text}\n\n")

        logger.info(f"✅ Saved to {output_path}")
        return output_path

    def _print_extraction_summary(self, metadata: Dict[str, Any], elapsed_time) -> None:
        """Print extraction summary"""

        print("\n" + "="*60)
        print("📊 TEXT DATA EXTRACTION SUMMARY")
        print("="*60)

        stats = metadata["statistics"]

        print(f"\n📁 Gold Data:")
        if "error" not in stats["gold"]:
            print(f"   📄 Documents processed: {stats['gold']['total_documents']}")
            print(f"   ✅ Texts extracted: {stats['gold']['texts_extracted']}")
            print(f"   ❌ Texts skipped: {stats['gold']['texts_skipped']}")
            print(f"   📏 Average length: {stats['gold']['avg_length']:.1f} chars")
        else:
            print(f"   ❌ Error: {stats['gold']['error']}")

        print(f"\n📁 Silver Data:")
        if "error" not in stats["silver"]:
            print(f"   📄 Documents processed: {stats['silver']['total_documents']}")
            print(f"   ✅ Texts extracted: {stats['silver']['texts_extracted']}")
            print(f"   ❌ Texts skipped: {stats['silver']['texts_skipped']}")
            print(f"   📏 Average length: {stats['silver']['avg_length']:.1f} chars")
        else:
            print(f"   ❌ Error: {stats['silver']['error']}")

        print(f"\n📊 Combined Statistics:")
        print(f"   📄 Total texts: {stats['combined']['total_texts']}")
        print(f"   📏 Total characters: {stats['combined']['total_characters']:,}")
        print(f"   📏 Average length: {stats['combined']['avg_length']:.1f} chars")

        print(f"\n📁 Output Files Created:")
        for file_type, file_path in metadata["output_files"].items():
            print(f"   📄 {file_type}: {file_path}")

        print(f"\n⏱️  Processing time: {elapsed_time.total_seconds():.2f} seconds")
        print(f"📍 Metadata saved to: {self.output_dir / 'text_extraction_metadata.json'}")

        print("\n🎯 Usage Examples:")
        print("   # Universal RAG with high quality texts:")
        print("   python universal_rag_enhanced.py create-from-file \\")
        print("     --name=maintenance \\")
        print(f"     --corpus={self.output_dir}/maintenance_high_quality_texts.txt")

        print("\n   # Universal RAG with all texts:")
        print("   python universal_rag_enhanced.py create-from-file \\")
        print("     --name=maintenance \\")
        print(f"     --corpus={self.output_dir}/maintenance_all_texts.txt")

        print("\n✅ Text extraction complete! Your project can now rely purely on text data.")
        print("="*60)

    def create_sample_file(self, n_samples: int = 50) -> Path:
        """Create a small sample file for testing"""

        logger.info(f"📝 Creating sample file with {n_samples} texts...")

        # Extract some texts
        gold_texts, _ = self._extract_texts_from_file(self.gold_file, "gold", 30)

        if len(gold_texts) < n_samples:
            sample_texts = gold_texts
        else:
            # Take evenly spaced samples
            step = len(gold_texts) // n_samples
            sample_texts = [gold_texts[i * step] for i in range(n_samples)]

        sample_path = self._save_texts_to_file(
            sample_texts,
            "maintenance_sample_texts.txt",
            f"Sample maintenance texts ({len(sample_texts)} texts for testing)"
        )

        logger.info(f"✅ Sample file created: {sample_path}")
        return sample_path


def main():
    """Main script execution"""

    parser = argparse.ArgumentParser(description="Extract text data from MaintIE annotation files")
    parser.add_argument("--min-length", type=int, default=30,
                       help="Minimum text length (default: 30)")
    parser.add_argument("--no-combined", action="store_true",
                       help="Skip creating combined file")
    parser.add_argument("--no-separate", action="store_true",
                       help="Skip creating separate gold/silver files")
    parser.add_argument("--sample-only", type=int,
                       help="Create only a sample file with N texts")
    parser.add_argument("--output-dir", type=Path,
                       help="Output directory (default: same as raw data)")

    args = parser.parse_args()

    # Initialize extractor
    extractor = TextDataExtractor(output_dir=args.output_dir)

    try:
        if args.sample_only:
            # Create sample file only
            extractor.create_sample_file(args.sample_only)
        else:
            # Full extraction
            extractor.extract_all_text_data(
                min_length=args.min_length,
                create_combined=not args.no_combined,
                create_separate=not args.no_separate
            )

    except Exception as e:
        logger.error(f"❌ Extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()