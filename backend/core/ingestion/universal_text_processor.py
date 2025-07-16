"""
Universal Text Processor
Handles text corpus ingestion and processing for any domain
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TextDocument:
    """Universal text document representation"""
    id: str
    text: str
    metadata: Dict[str, Any]
    quality_score: float
    source: str
    processed_at: str


@dataclass
class CorpusStatistics:
    """Statistics about a text corpus"""
    total_documents: int
    total_characters: int
    avg_length: float
    min_length: int
    max_length: int
    quality_distribution: Dict[str, int]
    source_distribution: Dict[str, int]


class UniversalTextProcessor:
    """Universal text processor that works with any domain"""

    def __init__(self, min_length: int = 10, max_length: int = 10000):
        """Initialize universal text processor"""
        self.min_length = min_length
        self.max_length = max_length
        self.processors = self._build_processors()

        logger.info(f"Universal text processor initialized (min_length={min_length}, max_length={max_length})")

    def process_text_corpus(self, texts: List[str], source: str = "unknown") -> Tuple[List[TextDocument], CorpusStatistics]:
        """Process a corpus of texts from any domain"""

        logger.info(f"Processing text corpus with {len(texts)} texts from source: {source}")

        documents = []
        processed_count = 0
        skipped_count = 0

        for i, text in enumerate(texts):
            try:
                # Clean and validate text
                cleaned_text = self._clean_text(text)

                if not self._is_valid_text(cleaned_text):
                    skipped_count += 1
                    continue

                # Calculate quality score
                quality_score = self._calculate_quality_score(cleaned_text)

                # Create document
                doc = TextDocument(
                    id=f"{source}_{i:06d}",
                    text=cleaned_text,
                    metadata={
                        "original_length": len(text),
                        "cleaned_length": len(cleaned_text),
                        "source_index": i
                    },
                    quality_score=quality_score,
                    source=source,
                    processed_at=datetime.now().isoformat()
                )

                documents.append(doc)
                processed_count += 1

            except Exception as e:
                logger.warning(f"Failed to process text {i}: {e}")
                skipped_count += 1
                continue

        # Generate statistics
        statistics = self._generate_statistics(documents)

        logger.info(f"Text processing completed: {processed_count} processed, {skipped_count} skipped")
        return documents, statistics

    def process_file(self, file_path: Path, source: str = None) -> Tuple[List[TextDocument], CorpusStatistics]:
        """Process text from a file"""

        if source is None:
            source = file_path.stem

        logger.info(f"Processing file: {file_path}")

        # Detect file format and read appropriately
        if file_path.suffix.lower() == '.txt':
            texts = self._read_text_file(file_path)
        elif file_path.suffix.lower() == '.json':
            texts = self._read_json_file(file_path)
        else:
            # Try to read as plain text
            texts = self._read_text_file(file_path)

        return self.process_text_corpus(texts, source)

    def filter_by_quality(self, documents: List[TextDocument], min_quality: float = 0.7) -> List[TextDocument]:
        """Filter documents by quality score"""
        filtered = [doc for doc in documents if doc.quality_score >= min_quality]
        logger.info(f"Quality filtering: {len(filtered)}/{len(documents)} documents retained (min_quality={min_quality})")
        return filtered

    def filter_by_length(self, documents: List[TextDocument], min_length: int = None, max_length: int = None) -> List[TextDocument]:
        """Filter documents by text length"""
        min_len = min_length or self.min_length
        max_len = max_length or self.max_length

        filtered = [doc for doc in documents if min_len <= len(doc.text) <= max_len]
        logger.info(f"Length filtering: {len(filtered)}/{len(documents)} documents retained (length={min_len}-{max_len})")
        return filtered

    def sample_documents(self, documents: List[TextDocument], sample_size: int, strategy: str = "random") -> List[TextDocument]:
        """Sample documents using different strategies"""

        if len(documents) <= sample_size:
            return documents

        if strategy == "random":
            import random
            return random.sample(documents, sample_size)
        elif strategy == "quality":
            # Sample highest quality documents
            sorted_docs = sorted(documents, key=lambda x: x.quality_score, reverse=True)
            return sorted_docs[:sample_size]
        elif strategy == "diverse":
            # Sample to maximize diversity (simple length-based diversity)
            sorted_docs = sorted(documents, key=lambda x: len(x.text))
            step = len(sorted_docs) // sample_size
            return [sorted_docs[i * step] for i in range(sample_size)]
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

    def _clean_text(self, text: str) -> str:
        """Universal text cleaning"""

        # Apply all cleaning processors
        cleaned = text
        for processor_name, processor_func in self.processors.items():
            try:
                cleaned = processor_func(cleaned)
            except Exception as e:
                logger.warning(f"Cleaning processor {processor_name} failed: {e}")
                continue

        return cleaned.strip()

    def _build_processors(self) -> Dict[str, callable]:
        """Build universal text cleaning processors"""
        return {
            "normalize_whitespace": self._normalize_whitespace,
            "remove_special_chars": self._remove_special_chars,
            "normalize_case": self._normalize_case,
            "remove_extra_spaces": self._remove_extra_spaces
        }

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters"""
        # Replace various whitespace with standard space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _remove_special_chars(self, text: str) -> str:
        """Remove or normalize special characters"""
        # Keep alphanumeric, spaces, and basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
        return text

    def _normalize_case(self, text: str) -> str:
        """Normalize text case (keep original for now)"""
        # For universal processing, we might want to preserve original case
        # Different domains may have different case conventions
        return text

    def _remove_extra_spaces(self, text: str) -> str:
        """Remove extra spaces"""
        return re.sub(r'\s+', ' ', text).strip()

    def _is_valid_text(self, text: str) -> bool:
        """Check if text meets validity criteria"""

        # Length check
        if len(text) < self.min_length or len(text) > self.max_length:
            return False

        # Content check - must have some alphabetic characters
        if not re.search(r'[a-zA-Z]', text):
            return False

        # Not just punctuation or numbers
        alpha_ratio = len(re.findall(r'[a-zA-Z]', text)) / len(text)
        if alpha_ratio < 0.3:  # At least 30% alphabetic
            return False

        return True

    def _calculate_quality_score(self, text: str) -> float:
        """Calculate text quality score (0.0 to 1.0)"""

        score = 0.0

        # Length factor (prefer moderate lengths)
        length = len(text)
        if 50 <= length <= 500:
            score += 0.3
        elif 20 <= length <= 1000:
            score += 0.2
        elif length >= 10:
            score += 0.1

        # Alphabetic content ratio
        alpha_ratio = len(re.findall(r'[a-zA-Z]', text)) / len(text)
        score += alpha_ratio * 0.3

        # Word count (prefer texts with multiple words)
        word_count = len(text.split())
        if word_count >= 5:
            score += 0.2
        elif word_count >= 2:
            score += 0.1

        # Sentence structure (presence of punctuation)
        if re.search(r'[\.!?]', text):
            score += 0.1

        # Capitalization (proper sentence structure)
        if re.search(r'^[A-Z]', text.strip()):
            score += 0.1

        return min(1.0, score)

    def _read_text_file(self, file_path: Path) -> List[str]:
        """Read texts from a plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split by double newlines (paragraph separation)
            if '\n\n' in content:
                texts = [t.strip() for t in content.split('\n\n') if t.strip()]
            else:
                # Split by single newlines
                texts = [t.strip() for t in content.split('\n') if t.strip()]

            return texts

        except Exception as e:
            logger.error(f"Failed to read text file {file_path}: {e}")
            return []

    def _read_json_file(self, file_path: Path) -> List[str]:
        """Read texts from a JSON file"""
        try:
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            texts = []

            # Handle different JSON structures
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        texts.append(item)
                    elif isinstance(item, dict):
                        # Look for common text fields
                        for key in ['text', 'content', 'body', 'message', 'description']:
                            if key in item and isinstance(item[key], str):
                                texts.append(item[key])
                                break
            elif isinstance(data, dict):
                # Single document or structured data
                for key in ['texts', 'documents', 'content']:
                    if key in data and isinstance(data[key], list):
                        texts.extend([str(t) for t in data[key]])
                        break

            return texts

        except Exception as e:
            logger.error(f"Failed to read JSON file {file_path}: {e}")
            return []

    def _generate_statistics(self, documents: List[TextDocument]) -> CorpusStatistics:
        """Generate corpus statistics"""

        if not documents:
            return CorpusStatistics(
                total_documents=0,
                total_characters=0,
                avg_length=0.0,
                min_length=0,
                max_length=0,
                quality_distribution={},
                source_distribution={}
            )

        lengths = [len(doc.text) for doc in documents]

        # Quality distribution
        quality_dist = {"high": 0, "medium": 0, "low": 0}
        for doc in documents:
            if doc.quality_score >= 0.8:
                quality_dist["high"] += 1
            elif doc.quality_score >= 0.6:
                quality_dist["medium"] += 1
            else:
                quality_dist["low"] += 1

        # Source distribution
        source_dist = {}
        for doc in documents:
            source_dist[doc.source] = source_dist.get(doc.source, 0) + 1

        return CorpusStatistics(
            total_documents=len(documents),
            total_characters=sum(lengths),
            avg_length=sum(lengths) / len(lengths),
            min_length=min(lengths),
            max_length=max(lengths),
            quality_distribution=quality_dist,
            source_distribution=source_dist
        )