"""
MaintIE Data Adapter for Universal RAG
Extracts raw text data from existing MaintIE structure for Universal RAG processing
Auto-detects and prefers pure text files when available
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

from src.knowledge.data_transformer import MaintIEDataTransformer
from src.models.maintenance_models import MaintenanceDocument
from config.settings import settings

logger = logging.getLogger(__name__)


class MaintIEDataAdapter:
    """Adapter to extract text corpus from existing MaintIE data for Universal RAG"""

    def __init__(self):
        """Initialize adapter with existing MaintIE infrastructure"""
        self.data_transformer = MaintIEDataTransformer()
        self.raw_data_dir = settings.raw_data_dir
        self.processed_data_dir = settings.processed_data_dir

        # Check for pure text files
        self.pure_text_files = self._detect_pure_text_files()

        logger.info("MaintIEDataAdapter initialized")
        if self.pure_text_files:
            logger.info("🎯 Pure text files detected - will use text-only mode")
            for file_type, file_path in self.pure_text_files.items():
                logger.info(f"   📄 {file_type}: {file_path}")
        else:
            logger.info("📊 No pure text files found - will extract from MaintIE annotations")

    def _detect_pure_text_files(self) -> Dict[str, Path]:
        """Detect if pure text files exist"""
        text_files = {}

        # Check for extracted text files
        candidates = {
            "high_quality": "maintenance_high_quality_texts.txt",
            "gold": "maintenance_gold_texts.txt",
            "silver": "maintenance_silver_texts.txt",
            "combined": "maintenance_all_texts.txt",
            "sample": "maintenance_sample_texts.txt"
        }

        for file_type, filename in candidates.items():
            file_path = self.raw_data_dir / filename
            if file_path.exists():
                text_files[file_type] = file_path

        return text_files

    def extract_text_corpus_from_raw_data(self, min_text_length: int = 50) -> Dict[str, List[str]]:
        """Extract text corpus from raw MaintIE data files"""

        # Check if pure text files are available
        if self.pure_text_files:
            logger.info("🎯 Using pure text files instead of MaintIE annotations")
            return self._extract_from_pure_text_files(min_text_length)

        logger.info("📊 Extracting text corpus from raw MaintIE annotation files...")

        corpus_data = {
            "gold_texts": [],
            "silver_texts": [],
            "combined_texts": [],
            "statistics": {}
        }

        # Load raw data using existing infrastructure
        raw_data = self.data_transformer.load_raw_data()

        # Extract texts from gold data
        gold_texts = self._extract_texts_from_dataset(
            raw_data["gold"],
            source="gold",
            min_length=min_text_length
        )
        corpus_data["gold_texts"] = gold_texts

        # Extract texts from silver data
        silver_texts = self._extract_texts_from_dataset(
            raw_data["silver"],
            source="silver",
            min_length=min_text_length
        )
        corpus_data["silver_texts"] = silver_texts

        # Combine all texts
        corpus_data["combined_texts"] = gold_texts + silver_texts

        # Generate statistics
        corpus_data["statistics"] = {
            "gold_documents": len(raw_data["gold"]),
            "silver_documents": len(raw_data["silver"]),
            "gold_texts_extracted": len(gold_texts),
            "silver_texts_extracted": len(silver_texts),
            "total_texts": len(corpus_data["combined_texts"]),
            "avg_text_length": sum(len(text) for text in corpus_data["combined_texts"]) / len(corpus_data["combined_texts"]) if corpus_data["combined_texts"] else 0,
            "data_source": "maintie_annotations"
        }

        logger.info(f"Text extraction complete: {corpus_data['statistics']}")
        return corpus_data

    def _extract_from_pure_text_files(self, min_text_length: int = 50) -> Dict[str, List[str]]:
        """Extract text corpus from pure text files"""

        corpus_data = {
            "gold_texts": [],
            "silver_texts": [],
            "combined_texts": [],
            "statistics": {}
        }

        # Load from high quality or gold file
        if "high_quality" in self.pure_text_files:
            gold_texts = self._load_pure_text_file(self.pure_text_files["high_quality"], min_text_length)
        elif "gold" in self.pure_text_files:
            gold_texts = self._load_pure_text_file(self.pure_text_files["gold"], min_text_length)
        else:
            gold_texts = []

        corpus_data["gold_texts"] = gold_texts

        # Load from silver file if available
        if "silver" in self.pure_text_files:
            silver_texts = self._load_pure_text_file(self.pure_text_files["silver"], min_text_length)
        else:
            silver_texts = []

        corpus_data["silver_texts"] = silver_texts

        # Load from combined file if available and no separate files
        if "combined" in self.pure_text_files and not gold_texts and not silver_texts:
            combined_texts = self._load_pure_text_file(self.pure_text_files["combined"], min_text_length)
            corpus_data["combined_texts"] = combined_texts
        else:
            corpus_data["combined_texts"] = gold_texts + silver_texts

        # Generate statistics
        corpus_data["statistics"] = {
            "gold_texts_extracted": len(gold_texts),
            "silver_texts_extracted": len(silver_texts),
            "total_texts": len(corpus_data["combined_texts"]),
            "avg_text_length": sum(len(text) for text in corpus_data["combined_texts"]) / len(corpus_data["combined_texts"]) if corpus_data["combined_texts"] else 0,
            "data_source": "pure_text_files",
            "source_files": {k: str(v) for k, v in self.pure_text_files.items()}
        }

        logger.info(f"Pure text extraction complete: {corpus_data['statistics']}")
        return corpus_data

    def _load_pure_text_file(self, file_path: Path, min_length: int) -> List[str]:
        """Load texts from a pure text file"""

        logger.info(f"📄 Loading pure text file: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Skip header lines (start with #)
            lines = content.split('\n')
            text_lines = [line for line in lines if not line.strip().startswith('#')]

            # Join and split by double newlines (our format)
            text_content = '\n'.join(text_lines)
            raw_texts = text_content.split('\n\n')

            # Clean and filter texts
            texts = []
            for text in raw_texts:
                # Unescape newlines
                clean_text = text.replace('\\n', '\n').replace('\\r', '\r').strip()

                if clean_text and len(clean_text) >= min_length:
                    texts.append(clean_text)

            logger.info(f"✅ Loaded {len(texts)} texts from {file_path}")
            return texts

        except Exception as e:
            logger.error(f"❌ Error loading pure text file {file_path}: {e}")
            return []

    def extract_text_corpus_from_processed_data(self, min_text_length: int = 50) -> Dict[str, List[str]]:
        """Extract text corpus from processed MaintIE data (if available)"""

        # If pure text files exist, use those instead
        if self.pure_text_files:
            logger.info("🎯 Using pure text files instead of processed data")
            return self._extract_from_pure_text_files(min_text_length)

        logger.info("📊 Extracting text corpus from processed MaintIE data...")

        corpus_data = {
            "document_texts": [],
            "entity_contexts": [],
            "combined_texts": [],
            "statistics": {}
        }

        try:
            # Try to load existing processed documents
            if self.data_transformer.load_existing_processed_data():
                logger.info("Using existing processed data")

                # Extract texts from documents
                document_texts = []
                for doc_id, document in self.data_transformer.documents.items():
                    doc_text = self._clean_and_validate_text(document.text, min_text_length)
                    if doc_text:
                        # Combine title and text for richer context
                        full_text = f"{document.title or ''} {doc_text}".strip()
                        document_texts.append(full_text)

                corpus_data["document_texts"] = document_texts

                # Extract entity contexts for additional training data
                entity_contexts = []
                for entity_id, entity in self.data_transformer.entities.items():
                    if entity.context and len(entity.context) >= min_text_length:
                        entity_contexts.append(entity.context)

                corpus_data["entity_contexts"] = entity_contexts

                # Combine all texts
                corpus_data["combined_texts"] = document_texts + entity_contexts

                # Generate statistics
                corpus_data["statistics"] = {
                    "total_documents": len(self.data_transformer.documents),
                    "total_entities": len(self.data_transformer.entities),
                    "document_texts_extracted": len(document_texts),
                    "entity_contexts_extracted": len(entity_contexts),
                    "total_texts": len(corpus_data["combined_texts"]),
                    "avg_text_length": sum(len(text) for text in corpus_data["combined_texts"]) / len(corpus_data["combined_texts"]) if corpus_data["combined_texts"] else 0,
                    "data_source": "processed_maintie_data"
                }

            else:
                logger.warning("No processed data available, falling back to raw data extraction")
                return self.extract_text_corpus_from_raw_data(min_text_length)

        except Exception as e:
            logger.error(f"Error extracting from processed data: {e}")
            logger.info("Falling back to raw data extraction")
            return self.extract_text_corpus_from_raw_data(min_text_length)

        logger.info(f"Processed data extraction complete: {corpus_data['statistics']}")
        return corpus_data

    def _extract_texts_from_dataset(self, dataset: List[Dict[str, Any]],
                                   source: str, min_length: int) -> List[str]:
        """Extract clean texts from a dataset (gold or silver)"""

        texts = []
        skipped = 0

        for doc_data in dataset:
            # Get document text
            doc_text = doc_data.get("text", "")

            # Clean and validate text
            clean_text = self._clean_and_validate_text(doc_text, min_length)
            if clean_text:
                texts.append(clean_text)
            else:
                skipped += 1

        logger.info(f"Extracted {len(texts)} texts from {source} data (skipped {skipped} short/empty texts)")
        return texts

    def _clean_and_validate_text(self, text: str, min_length: int) -> Optional[str]:
        """Clean and validate text for Universal RAG processing"""

        if not text or not isinstance(text, str):
            return None

        # Basic cleaning
        clean_text = text.strip()

        # Remove excessive whitespace
        clean_text = " ".join(clean_text.split())

        # Check minimum length
        if len(clean_text) < min_length:
            return None

        # Additional cleaning if needed
        # Remove very short sentences, excessive punctuation, etc.

        return clean_text

    def create_domain_specific_corpus(self, domain_name: str = "maintenance",
                                    quality_filter: str = "high") -> Dict[str, Any]:
        """Create a domain-specific corpus for Universal RAG"""

        logger.info(f"Creating {domain_name} domain corpus with {quality_filter} quality filter")

        corpus_info = {
            "domain_name": domain_name,
            "quality_filter": quality_filter,
            "texts": [],
            "metadata": {},
            "statistics": {}
        }

        # Determine data source strategy
        if self.pure_text_files:
            corpus_info["metadata"]["data_mode"] = "pure_text_files"
            corpus_data = self._extract_from_pure_text_files()
        else:
            corpus_info["metadata"]["data_mode"] = "maintie_annotations"

            if quality_filter == "high":
                # Use only gold data for high quality
                corpus_data = self.extract_text_corpus_from_raw_data()
                corpus_info["texts"] = corpus_data["gold_texts"]
                corpus_info["metadata"]["source"] = "gold_data_only"

            elif quality_filter == "mixed":
                # Use both gold and silver data
                corpus_data = self.extract_text_corpus_from_raw_data()
                corpus_info["texts"] = corpus_data["combined_texts"]
                corpus_info["metadata"]["source"] = "gold_and_silver_data"

            elif quality_filter == "processed":
                # Use processed data (includes entity contexts)
                corpus_data = self.extract_text_corpus_from_processed_data()
                corpus_info["texts"] = corpus_data["combined_texts"]
                corpus_info["metadata"]["source"] = "processed_data_with_contexts"

        # If using pure text files, apply quality filter logic
        if self.pure_text_files:
            if quality_filter == "high" and corpus_data["gold_texts"]:
                corpus_info["texts"] = corpus_data["gold_texts"]
                corpus_info["metadata"]["source"] = "pure_text_gold_only"
            elif quality_filter == "mixed" or quality_filter == "processed":
                corpus_info["texts"] = corpus_data["combined_texts"]
                corpus_info["metadata"]["source"] = "pure_text_combined"
            else:
                corpus_info["texts"] = corpus_data["combined_texts"]
                corpus_info["metadata"]["source"] = "pure_text_all_available"

        # Add domain metadata
        corpus_info["metadata"].update({
            "domain": domain_name,
            "extraction_date": str(Path(__file__).stat().st_mtime),
            "original_data_source": "maintie_dataset",
            "pure_text_available": bool(self.pure_text_files),
            "pure_text_files": {k: str(v) for k, v in self.pure_text_files.items()} if self.pure_text_files else None
        })

        # Calculate statistics
        texts = corpus_info["texts"]
        corpus_info["statistics"] = {
            "total_texts": len(texts),
            "avg_length": sum(len(text) for text in texts) / len(texts) if texts else 0,
            "min_length": min(len(text) for text in texts) if texts else 0,
            "max_length": max(len(text) for text in texts) if texts else 0,
            "total_characters": sum(len(text) for text in texts)
        }

        # Add source statistics if available
        if corpus_data.get("statistics"):
            corpus_info["statistics"]["source_statistics"] = corpus_data["statistics"]

        logger.info(f"Domain corpus created: {corpus_info['statistics']}")
        return corpus_info

    def save_corpus_to_file(self, corpus_info: Dict[str, Any],
                           output_path: Optional[Path] = None) -> Path:
        """Save extracted corpus to file for Universal RAG processing"""

        if output_path is None:
            output_path = self.processed_data_dir / f"{corpus_info['domain_name']}_corpus.txt"

        # Write texts to file (one per line for easy processing)
        with open(output_path, 'w', encoding='utf-8') as f:
            for text in corpus_info["texts"]:
                # Escape newlines within texts and write one text per line
                escaped_text = text.replace('\n', '\\n').replace('\r', '\\r')
                f.write(escaped_text + '\n')

        # Save metadata
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": corpus_info["metadata"],
                "statistics": corpus_info["statistics"]
            }, f, indent=2)

        logger.info(f"Corpus saved to {output_path} (metadata: {metadata_path})")
        return output_path

    def get_sample_texts(self, n_samples: int = 5) -> List[str]:
        """Get sample texts for preview/testing"""

        # If pure text files exist and include a sample file, use that
        if "sample" in self.pure_text_files:
            sample_texts = self._load_pure_text_file(self.pure_text_files["sample"], 30)
            if len(sample_texts) <= n_samples:
                return sample_texts
            # Return evenly spaced samples from sample file
            step = len(sample_texts) // n_samples
            return [sample_texts[i * step] for i in range(n_samples)]

        # Otherwise use existing logic
        if self.pure_text_files:
            corpus_data = self._extract_from_pure_text_files()
        else:
            corpus_data = self.extract_text_corpus_from_processed_data()

        texts = corpus_data["combined_texts"]

        if len(texts) <= n_samples:
            return texts

        # Return evenly spaced samples
        step = len(texts) // n_samples
        return [texts[i * step] for i in range(n_samples)]

    def check_pure_text_status(self) -> Dict[str, Any]:
        """Check status of pure text files"""

        status = {
            "pure_text_available": bool(self.pure_text_files),
            "files_found": {},
            "recommendations": []
        }

        if self.pure_text_files:
            for file_type, file_path in self.pure_text_files.items():
                file_size = file_path.stat().st_size if file_path.exists() else 0
                with open(file_path, 'r') as f:
                    # Count non-header lines
                    lines = [line for line in f.readlines() if not line.strip().startswith('#')]
                    text_count = len([line for line in lines if line.strip()])

                status["files_found"][file_type] = {
                    "path": str(file_path),
                    "size_bytes": file_size,
                    "estimated_texts": text_count // 2,  # Rough estimate (double newline separated)
                    "exists": file_path.exists()
                }

            status["recommendations"].append("✅ Pure text files detected - system will use text-only mode")
            status["recommendations"].append("🎯 Faster processing and better Universal RAG compatibility")
        else:
            status["recommendations"].append("💡 Consider running: python extract_texts.py")
            status["recommendations"].append("📊 This will create pure text files for better performance")

        return status